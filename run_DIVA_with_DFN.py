import logging
import os
import sys
import random
import datetime
import builtins
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import numpy as np
import torch
import transformers
from transformers.trainer_utils import set_seed
from transformers import HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version
from trainer import CustomTrainer
from arguments import DataTrainingArguments, ModelArguments, TrainingArguments
import csv
from tqdm import tqdm
from PIL import Image
import json
from open_clip import create_model_from_pretrained, get_tokenizer 
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore")


def random_seed(seed=42, rank=0):
    set_seed(seed)
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    try:
        import deepspeed
        deepspeed.runtime.utils.set_random_seed(seed + rank)
    except:
        print("deepspeed.runtime.utils.set_random_seed is not available")


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')
            builtin_print(*args, **kwargs)

    builtins.print = print


def setup_wandb_env(wandb_api_key=None):
    os.environ["WANDB_API_KEY"] = wandb_api_key or ''
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    os.environ["WANDB_CONFIG_DIR"] = "./wandb"


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.ddp_find_unused_parameters = True
    training_args.multiple_optimizer_training = False
    training_args.one_minus_one_data_transform = data_args.one_minus_one_data_transform
    training_args.cost_gradient_penalty = model_args.cost_gradient_penalty
    setup_wandb_env(training_args.wandb_api_key)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # data_args.data_seed
    random_seed(training_args.seed)
    data_args.seed = training_args.seed
    training_args.model_type = "image"

    from models.SD_with_DFN import SDModel
    from config import SDConfig
    
    config = SDConfig()
    config.tta.gradient_descent.train_steps = training_args.train_steps
    config.visual_pattern = training_args.visual_pattern
    config.clip_image_size = training_args.clip_image_size
    model = SDModel(config)
    
    # print model parameters
    logger.info(f"{str(model)}")
    model.cuda()

    from data import get_cc3m_wds_dataset_and_collator
    wds_dataset, wds_collator = get_cc3m_wds_dataset_and_collator(data_args, model_args)

    if config.model.freeze_class_embeds:
        params = []
        for key,parm in model.named_parameters():
            if 'final_fc' not in key:
                params.append(parm)
        
    optimizer = torch.optim.SGD(
        params, lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        momentum=config.tta.gradient_descent.optimizer_momentum
    )
    scheduler = None

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=wds_dataset,
        data_collator=wds_collator,
        optimizers=(optimizer, scheduler)
    )
    setup_for_distributed(torch.distributed.get_rank() == 0)
    
    from callbacks import ModelCallback
    trainer.add_callback(ModelCallback)

    # Evaluation
    if training_args.local_rank == 0:
        print("CLIP's Performance on MMVP-VLM —— Before Generative Fine-tuning")
        results_before = official_evaluation(model.class_model.model, config)
        print(results_before)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(output_dir=training_args.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        
    # Evaluation
    if training_args.local_rank == 0:
        print("CLIP's Performance on MMVP-VLM —— After Generative Fine-tuning")
        model_weight_save_path = os.path.join(training_args.output_dir, 'CLIP_after_GenFT.pth')
        torch.save(trainer.model.state_dict(), model_weight_save_path)
        results_final_after = official_evaluation(trainer.model.class_model.model, config)
        print(results_final_after)
        save_results(results_before, results_final_after, output_dir=training_args.output_dir)


def benchmark_model(model, benchmark_dir, device = "cpu", config=None):
    if config.clip_image_size == 224:
        _, preprocess = create_model_from_pretrained(model_name='ViT-H-14-quickgelu', pretrained="pretrained_weights/CLIP/DFN5B-CLIP-ViT-H-14/open_clip_pytorch_model.bin", device=device)
    if config.clip_image_size == 378:
        _, preprocess = create_model_from_pretrained(model_name='ViT-H-14-378-quickgelu', pretrained="pretrained_weights/CLIP/DFN5B-CLIP-ViT-H-14-378/open_clip_pytorch_model.bin", device=device)
    
    tokenizer = get_tokenizer('ViT-H-14')

    image_dir = os.path.join(benchmark_dir, 'MLLM_VLM_Images')
    csv_file = os.path.join(benchmark_dir, 'Questions.csv')

    csv_outfile = open('Prediction_Results_DFN.csv', 'w', newline='')
    csv_writer = csv.writer(csv_outfile)
    csv_writer.writerow(['qid1', 'qid2', 'pred1', 'pred2', 'gt1', 'gt2', 'q1score', 'q2score'])  # header

    categories = [
        'Orientation and Direction', 'Presence of Specific Features', 
        'State and Condition', 'Quantity and Count', 
        'Positional and Relational Context', 'Color and Appearance',
        'Structural Characteristics', 'Texts',
        'Viewpoint and Perspective'
    ]

    pair_accuracies = {category: 0 for category in categories}
    num_pairs = 0
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for i, row in tqdm(enumerate(reader)):
            qid1, qtype1, statement1 = row
        
            # Get next row for the pair
            row = next(reader, None)
            if not row:
                break
            qid2, qtype2, statement2 = row
            
            qid1, qid2 = int(qid1), int(qid2)
            
            img1 = Image.open(os.path.join(image_dir, qtype1, f'{qid1}.jpg'))
            img2 = Image.open(os.path.join(image_dir, qtype1, f'{qid2}.jpg'))

            text1 = 'a photo of ' + statement1
            text2 = 'a photo of ' + statement2

            text1 = tokenizer(text1).to(device)
            text2 = tokenizer(text2).to(device)
            
            img1 = preprocess(img1).unsqueeze(0).to(device)
            img2 = preprocess(img2).unsqueeze(0).to(device)
            imgs = torch.cat((img1, img2), dim=0)

            with torch.no_grad(), torch.cuda.amp.autocast():
                model.eval().float()
                
                # original code
                # image_features = model.encode_image(imgs)

                # ours
                if config.clip_image_size == 224:
                    image_features = model.encode_image(imgs, normalize=True)[:,0,:]
                if config.clip_image_size == 378:
                    global_image_features = model.encode_image(imgs, normalize=True)[:,0,:]
                    local_image_features = model.encode_image(imgs, normalize=True)[:,1:,:].mean(dim=1)
                    image_features = global_image_features + local_image_features

                text1_features = model.encode_text(text1, normalize=True)
                text2_features = model.encode_text(text2, normalize=True)
                logits_per_image1 = model.logit_scale.exp() * image_features @ text1_features.T
                logits_per_text1 = logits_per_image1.T
                logits_per_image2 = model.logit_scale.exp() * image_features @ text2_features.T
                logits_per_text2 = logits_per_image2.T
                probs1 = logits_per_text1.softmax(dim=-1).cpu().numpy()
                probs2 = logits_per_text2.softmax(dim=-1).cpu().numpy()


            img1_score1 = probs1[0][0]
            img1_score2 = probs2[0][0]
            
            pred1 = "img1" if img1_score1 > 0.5 else "img2"
            pred2 = "img1" if img1_score2 > 0.5 else "img2"

            gt1 = "img1" if qid1 % 2 == 1 else "img2"
            gt2 = "img1" if qid2 % 2 == 1 else "img2"

            csv_writer.writerow([qid1, qid2, pred1, pred2, gt1, gt2, img1_score1, img1_score2])
                
            current_category = categories[num_pairs // 15]
            if pred1 == gt1 and pred2 == gt2:
                pair_accuracies[current_category] += 1
            num_pairs += 1
            
        csv_outfile.close()

    # Calculate percentage accuracies
    Category_Score_List = []
    
    for category in pair_accuracies:
        pair_accuracies[category] = (pair_accuracies[category] / (num_pairs // len(categories))) * 100
        Category_Score_List.append(pair_accuracies[category])
        
    pair_accuracies['average_score'] = sum(Category_Score_List)/len(Category_Score_List)

    return pair_accuracies

def official_evaluation(clip_model, config):
    
    with torch.no_grad():
        clip_model.eval()

        # models
        data = "dataset/MMVP_VLM"
        clip_model_device = next(clip_model.parameters()).device
        if config.clip_image_size == 224:
            results_openai = {f'DFN5B-CLIP-ViT-H-14': benchmark_model(clip_model, data, clip_model_device, config)}
        if config.clip_image_size == 378:
            results_openai = {f'DFN5B-CLIP-ViT-H-14-378': benchmark_model(clip_model, data, clip_model_device, config)}

        # Merge results
        results = {**results_openai}

        # Convert results to format suitable for star plot
        categories = results[list(results.keys())[0]].keys()
        data = {'Categories': list(categories)}
        for model in list(results_openai.keys()):
            data[model] = [results[model][category] for category in categories]

        return results
    
def save_results(results_before, results_final_after, output_dir, filename='pred_result.json'):
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_data = {
        'results_before': results_before,
        'results_final_after': results_final_after
    }
    
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)


if __name__ == "__main__":
    main()
