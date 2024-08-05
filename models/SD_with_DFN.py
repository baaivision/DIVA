from dataclasses import dataclass
from typing import Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import torchvision
from transformers.utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from .build import load_sd_model, load_clip_model_DFN
from .utils import initiate_time_steps, prepare_class_text_embeddings
import torchvision.transforms as transforms
from open_clip import get_tokenizer 

@dataclass
class SDOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None

class SDModel(PreTrainedModel):
    def __init__(
        self,
        config = None,
    ):
        super().__init__(config)
        
        self.model_id, self.pipe, self.vae, self.tokenizer, self.text_encoder, self.unet, self.scheduler, self.image_renormalizer = load_sd_model(config)
        self.text_encoder.eval()
        self.vae.eval()
        self.unet.eval()
        
        self.pattern_dictionary={'None':['']}
        self.config.actual_bs = len(self.pattern_dictionary[self.config.visual_pattern])
        self.class_model = load_clip_model_DFN(config)
        self.class_model.eval()
        self.config = config
        discrimi_size = self.config.clip_image_size
        self.resize_transform_discrimi = transforms.Resize((discrimi_size, discrimi_size))
        self.visual_proj = nn.Linear(1024, 1024)


    def classify(self, image, classes):

        image_features, logits = self.class_model(image)

        if classes is not None:
            logits = logits[:, classes]

        probs = logits.softmax(-1)
        max_idx = probs.argmax(-1)
        K = probs.shape[-1] if self.config.tta.adapt_topk == -1 else self.config.tta.adapt_topk
        topk_idx = probs.argsort(descending=True)[:, :K]

        if classes is not None:
            classes = torch.tensor(classes).to(logits.device)
            max_class_idx = classes[max_idx.flatten()].view(max_idx.shape)
            topk_class_idx = classes[topk_idx.flatten()].view(topk_idx.shape)
        else:
            max_class_idx, topk_class_idx = max_idx, topk_idx

        return image_features, logits, topk_idx, max_class_idx, topk_class_idx
    
    def _unet_pred_noise(self, x_start, t, noise, context):

        _,c,h,w = x_start.shape
        device = t.device
        nt = t.shape[0]

        x_start = x_start.unsqueeze(1)
        x_start = x_start.expand(-1, nt//x_start.shape[0], -1, -1, -1)
        x_start = x_start.reshape(-1,c,h,w)

        alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        noised_latent = (
            x_start * (alphas_cumprod[t]**0.5).view(-1, 1, 1, 1).to(device)
            + noise * ((1 - alphas_cumprod[t])**0.5).view(-1, 1, 1, 1).to(device)
        )
        pred_noise = self.unet(noised_latent, t, encoder_hidden_states=context.expand(nt, -1, -1)).sample

        return pred_noise

    def zeroshot_classifier_DFN(self, classnames, templates, model):
        with torch.no_grad():
            zeroshot_weights = []
            tokenizer = get_tokenizer('ViT-H-14')
            for classname in classnames:
                texts = [template.format(classname) for template in templates]
                texts = tokenizer(texts, context_length=model.context_length).cuda()
                class_embeddings = model.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights

    def forward(
        self,
        image: torch.Tensor = None,
        text = None
    ) -> SDOutput:
        
        text = self.pattern_dictionary[self.config.visual_pattern]
        with torch.no_grad():
            imagenet_templates = ['{}',]
            zeroshot_weights = self.zeroshot_classifier_DFN(text, imagenet_templates, self.class_model.model.float()).float()
            
        self.class_model.final_fc.weight.data = zeroshot_weights.T
        self.class_model.final_fc.weight.data = self.class_model.final_fc.weight.data.contiguous()
        classes = [i for i in range(len(text))]
        
        discrimi_image = self.resize_transform_discrimi(image)
        genera_image = image
        real_BS = image.shape[0]
        after_DF_expand_BS = real_BS*self.config.input.batch_size
        
        # prepare_vae_latent
        self.vae, self.text_encoder, self.unet = self.vae.to(torch.float32), self.text_encoder.to(torch.float32), self.unet.to(torch.float32)
        renormed_image = self.image_renormalizer(genera_image).detach()
        x0 = self.vae.encode(renormed_image).latent_dist.mean.float()
        latent = x0 * 0.18215
        
        # prepare_total_timesteps
        total_timestep = self.scheduler.num_train_timesteps
        
        for step in range(self.config.tta.gradient_descent.train_steps):
            # Initiate timesteps and noise
            timesteps = initiate_time_steps(step, total_timestep, after_DF_expand_BS, self.config).long()
            timesteps = timesteps.cuda()

            c, h, w = latent.shape[1:]
            if not self.config.tta.use_same_noise_among_timesteps:
                noise = torch.randn((real_BS* self.config.input.batch_size, c, h, w)).cuda()
            else:
                noise = torch.randn((1, c, h, w)).cuda()
                noise = noise.repeat(real_BS* self.config.input.batch_size, 1, 1, 1)

            if self.config.tta.adapt_topk == -1:
                image_features, logits, _, _, _ = self.classify(discrimi_image, classes)
                pred_top_idx = None
            else:
                image_features, logits, pred_top_idx, _, _ = self.classify(discrimi_image, classes)
            real_BS, C = logits.shape[:2]

            # Pick top-K predictions
            if pred_top_idx is not None:
                pred_top_idx = pred_top_idx.squeeze(0)
            else:
                pred_top_idx = torch.arange(C).cuda()

            logits = logits[:, pred_top_idx]

            class_text_embeddings = prepare_class_text_embeddings(self.tokenizer, self.text_encoder, class_names=text)
            class_text_embeddings = class_text_embeddings.detach()
            class_text_embeddings = class_text_embeddings[pred_top_idx, :]

            # Compute conditional text embeddings using weighted-summed predictions
            probs = logits.softmax(-1)
            probs = probs[:, :, None, None]
            class_text_embeddings = (class_text_embeddings.unsqueeze(0).repeat(after_DF_expand_BS, 1, 1, 1))
            _, word_num, _, _ = probs.shape
            probs = probs.unsqueeze(1).repeat(1,self.config.input.batch_size,1,1,1).reshape(-1,word_num,1,1)
            context = (probs * class_text_embeddings).sum(1)
            image_features = self.visual_proj(image_features)
            context = context.mean(dim=1).unsqueeze(1) + image_features
                    
            # Predict noise with the diffusion model
            pred_noise = self._unet_pred_noise(x_start=latent, t=timesteps, noise=noise, context=context).float()

            # Compute diffusion loss
            if self.config.tta.loss == "l1":
                loss = torch.nn.functional.l1_loss(pred_noise, noise)
            else:
                loss = torch.nn.functional.mse_loss(pred_noise, noise)
            
            if step != (self.config.tta.gradient_descent.train_steps-1):
                loss.backward()

        return SDOutput(loss=loss)