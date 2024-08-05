import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline, DDPMScheduler, EulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from .utils import VQVAEUnNormalize
from .CLIP_bank import OpenAICLIP, DFN, SigLIP, MetaCLIP

def load_sd_model(config):
    """Load Stable Diffusion model"""
    dtype = torch.float32
    image_renormalizer = VQVAEUnNormalize(
        mean=config.input.mean, std=config.input.std
    )
    if config.model.sd_version == '1-4':
        if config.model.use_flash:
            model_id = "CompVis/stable-diffusion-v1-4"
            scheduler = EulerDiscreteScheduler.from_pretrained(
                model_id, subfolder="scheduler"
            )
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id, scheduler=scheduler, torch_dtype=dtype
            ).cuda()
            pipe.enable_xformers_memory_efficient_attention()
            vae = pipe.vae.cuda()
            tokenizer = pipe.tokenizer
            text_encoder = pipe.text_encoder.cuda()
            unet = pipe.unet.cuda()
        else:
            vae = AutoencoderKL.from_pretrained(
                f"CompVis/stable-diffusion-v{config.model.sd_version}",
                subfolder="vae", torch_dtype=dtype
            ).cuda()
            tokenizer = CLIPTokenizer.from_pretrained(
                "/share/project/wangwenxuan/projects/Overcome_VS/MMVP_Test/openai/clip-vit-large-patch14"
            )
            text_encoder = CLIPTextModel.from_pretrained(
                "/share/project/wangwenxuan/projects/Overcome_VS/MMVP_Test/openai/clip-vit-large-patch14", torch_dtype=dtype
            ).cuda()
            unet = UNet2DConditionModel.from_pretrained(
                f"CompVis/stable-diffusion-v{config.model.sd_version}",
                subfolder="unet", torch_dtype=dtype
            ).cuda()
            scheduler_config = get_scheduler_config(config)
            scheduler = DDPMScheduler(
                num_train_timesteps=scheduler_config['num_train_timesteps'],
                beta_start=scheduler_config['beta_start'],
                beta_end=scheduler_config['beta_end'],
                beta_schedule=scheduler_config['beta_schedule']
            )
    elif config.model.sd_version == '2-1':
        
        model_id = "pretrained_weights/SD/stable-diffusion-2-1-base"
        print(f'model_id:{model_id}')
        
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=dtype)
        pipe.to(dtype)
        
        pipe.enable_xformers_memory_efficient_attention()
        vae = pipe.vae.cuda()
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder.cuda()
        unet = pipe.unet.cuda()

    if config.model.adapt_only_classifier:
        for m in [vae, text_encoder, unet]:
            for param in m.parameters():
                param.requires_grad = False
    for m in [vae, text_encoder]:
        for param in m.parameters():
            param.requires_grad = False

    return (model_id, pipe, vae, tokenizer, text_encoder, unet, scheduler, image_renormalizer)


def get_scheduler_config(config):
    assert config.model.sd_version in {'1-4', '2-1'}
    if config.model.sd_version == '1-4':
        schedule_config = {
            "_class_name": "PNDMScheduler",
            "_diffusers_version": "0.7.0.dev0",
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "num_train_timesteps": 1000,
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,
            "trained_betas": None,
            "clip_sample": False
        }
    elif config.model.sd_version == '2-1':
        schedule_config = {
            "_class_name": "EulerDiscreteScheduler",
            "_diffusers_version": "0.10.2",
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "clip_sample": False,
            "num_train_timesteps": 1000,
            "prediction_type": "epsilon",
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,  # todo
            "trained_betas": None
        }
    else:
        raise NotImplementedError

    return schedule_config


def load_clip_model_OpenAICLIP(config):

    class_model = OpenAICLIP(config)
    class_model.to(torch.float32)

    return class_model


def load_clip_model_DFN(config):

    class_model = DFN(config)
    class_model.to(torch.float32)

    return class_model


def load_clip_model_SigLIP(config):

    class_model = SigLIP(config)
    class_model.to(torch.float32)

    return class_model


def load_clip_model_MetaCLIP(config):

    class_model = MetaCLIP(config)
    class_model.to(torch.float32)

    return class_model