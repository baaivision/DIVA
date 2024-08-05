import clip
import torch.nn as nn
from open_clip import create_model_from_pretrained, create_model_and_transforms


class OpenAICLIP(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.clip_image_size == 224:
            model, _ = clip.load("pretrained_weights/CLIP/ViT-L-14.pt", jit=False)
        if config.clip_image_size == 336:
            model, _ = clip.load("pretrained_weights/CLIP/ViT-L-14-336px.pt",jit=False)

        self.final_fc = nn.Linear(768, config.actual_bs, bias=False)
        self.model = model
        self.config = config

    def forward(self, images):
        
        image_features = self.model.encode_image(images).float()
        logits = 100. * self.final_fc(image_features[:,0,:]).float()

        return image_features, logits


class DFN(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.clip_image_size == 224:
            model, _ = create_model_from_pretrained(model_name='ViT-H-14-quickgelu', pretrained="pretrained_weights/CLIP/DFN5B-CLIP-ViT-H-14/open_clip_pytorch_model.bin")
        if config.clip_image_size == 378:
            model, _ = create_model_from_pretrained(model_name='ViT-H-14-378-quickgelu', pretrained="pretrained_weights/CLIP/DFN5B-CLIP-ViT-H-14-378/open_clip_pytorch_model.bin")
        
        self.final_fc = nn.Linear(1024, config.actual_bs, bias=False)
        self.model = model
        self.config = config

    def forward(self, images):
        
        image_features = self.model.encode_image(images).float()
        logits = 100. * self.final_fc(image_features[:,0,:]).float()    

        return image_features, logits
    
    
class SigLIP(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.clip_image_size == 224:
            model, _ = create_model_from_pretrained(model_name='ViT-SO400M-14-SigLIP', pretrained="pretrained_weights/CLIP/ViT-SO400M-14-SigLIP/open_clip_pytorch_model.bin",
                                                    image_mean=([0.5,0.5,0.5]), image_std=([0.5,0.5,0.5]), image_interpolation="bicubic", image_resize_mode="squash")
        if config.clip_image_size == 384:
            model, _ = create_model_from_pretrained(model_name='ViT-SO400M-14-SigLIP-384', pretrained="pretrained_weights/CLIP/ViT-SO400M-14-SigLIP-384/open_clip_pytorch_model.bin",
                                                     image_mean=([0.5,0.5,0.5]), image_std=([0.5,0.5,0.5]), image_interpolation="bicubic", image_resize_mode="squash")

        self.final_fc = nn.Linear(1152, config.actual_bs, bias=False)
        self.model = model
        self.config = config

    def forward(self, images):
        
        image_features = self.model.encode_image(images).float()
        logits = 100. * self.final_fc(image_features[:,0,:]).float()    

        return image_features, logits


class MetaCLIP(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.metaclip_version == "large":
            model, _, _ = create_model_and_transforms(model_name='ViT-L-14-quickgelu', pretrained="pretrained_weights/CLIP/MetaCLIP/l14_fullcc2.5b.pt")
            self.final_fc = nn.Linear(768, config.actual_bs, bias=False)
        if config.metaclip_version == "huge":
            model, _, _ = create_model_and_transforms(model_name='ViT-H-14-quickgelu', pretrained="pretrained_weights/CLIP/MetaCLIP/h14_fullcc2.5b.pt")
            self.final_fc = nn.Linear(1024, config.actual_bs, bias=False)

        self.model = model
        self.config = config

    def forward(self, images):
        
        image_features = self.model.encode_image(images).float()
        logits = 100. * self.final_fc(image_features[:,0,:]).float()    

        return image_features, logits
