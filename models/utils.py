"""Utility functions"""
import importlib
import random

import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf, open_dict


class UnNormalize(object):
    """Unformalize image as: image = (image * std) + mean
    """
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, tensor):
        """
        Args:
            tensor: A tensor of shape [C, H, W] or [N, C, H, W]

        Returns:
            tensor: A tensor of shape [C, H, W] or [N, C, H, W]
        """

        std = self.std.to(tensor.device)
        mean = self.mean.to(tensor.device)
        if tensor.ndim == 3:
            std, mean = std.view(-1, 1, 1), mean.view(-1, 1, 1)
        elif tensor.ndim == 4:
            std, mean = std.view(1, -1, 1, 1), mean.view(1, -1, 1, 1)
        tensor = (tensor * std) + mean
        return tensor


class VQVAEUnNormalize(UnNormalize):
    """Unformalize image as:
    First: image = (image * std) + mean
    Second: image = (image * 2) - 1
    """
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) or (N, C, H, W)
                             to be unnormalized.
        Returns:
            Tensor: UnNormalized image.
        """
        tensor = super().__call__(tensor)
        tensor = 2 * tensor - 1
        return tensor


def mean_list(l):
    l = [int(_l) for _l in l]
    return float(sum(l)) / len(l)


def segment_mean(x, index):
    """Function as tf.segment_mean.
    """
    x = x.view(-1, x.shape[-1])
    index = index.view(-1)

    max_index = index.max() + 1
    sum_x = torch.zeros((max_index, x.shape[-1]),
                        dtype=x.dtype,
                        device=x.device)
    num_index = torch.zeros((max_index,),
                            dtype=x.dtype,
                            device=x.device)

    num_index = num_index.scatter_add_(
        0, index, torch.ones_like(index, dtype=x.dtype))
    num_index = torch.where(torch.eq(num_index, 0),
                            torch.ones_like(num_index, dtype=x.dtype),
                            num_index)

    index_2d = index.view(-1, 1).expand(-1, x.shape[-1])
    sum_x = sum_x.scatter_add_(0, index_2d, x)
    mean_x = sum_x.div_(num_index.view(-1, 1))

    return mean_x


def get_class_sd_features(tokenizer, text_encoder, input, device=None):
    """Prepare class text embeddings for Stable Diffusion

    Args:
        tokenizer: A nn.Module object of tokenizer.
        text_encoder: A nn.Module object of text encoder.
        input: A string
        device: GPU/CPU device
    """
    with torch.no_grad():    
        input_val = f'a photo of a {input}.',
        # Tokenize the text
        text_input = tokenizer(input_val, padding="max_length",
                               max_length=tokenizer.model_max_length,
                               truncation=True,
                               return_tensors="pt")
        # Get the text embeddings
        text_embeddings = text_encoder(text_input.input_ids.cuda())[0]

    return text_embeddings


def prepare_class_text_embeddings(tokenizer=None, text_encoder=None, class_names=None):
    
    text_features = []
    for class_name in class_names:
        text_features.append(
            get_class_sd_features(tokenizer, text_encoder, class_name)
        )
    text_features = torch.cat(text_features, dim=0)

    return text_features


def initiate_time_steps(step, total_timestep, batch_size, config):
    """A helper function to initiate time steps for the diffusion model.

    Args:
        step: An integer of the constant step
        total_timestep: An integer of the total timesteps of the diffusion model
        batch_size: An integer of the batch size
        config: A config object

    Returns:
        timesteps: A tensor of shape [batch_size,] of the time steps
    """
    if config.tta.rand_timestep_equal_int:
        interval_val = total_timestep // batch_size
        start_point = random.randint(0, interval_val - 1)
        timesteps = torch.tensor(
            list(range(start_point, total_timestep, interval_val))
        ).long()
        return timesteps
    elif config.tta.random_timestep_per_iteration:
        return torch.randint(0, total_timestep, (batch_size,)).long()          #default
    else:
        return torch.tensor([step] * batch_size).long()


def instantiate_from_config(config):
    """A helper function to instantiate a class from a config object.
    See https://github.com/CompVis/stable-diffusion/blob/main/ldm/util.py
    """
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    """A helper function to instantiate a class from a config object.
    See https://github.com/CompVis/stable-diffusion/blob/main/ldm/util.py
    """
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)