from typing import Optional, Sequence, Dict, Union, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from torchvision import transforms as T
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

def to_tensor(image):
    try:
        image = ToTensor()(image)
    except Exception as e:
        image = image.float()
        print(e)
    return image

def _convert_to_rgb(image):
    try:
        image = image.convert('RGB')
    except Exception as e:
        print(e)
    return image

def image_transform_original_resolution(
    image,
    patch_size: int,
    max_size:int = 2048
):
    """accept a pil image and transform into torch.tensor"""
    w, h = map(lambda x: x // patch_size * patch_size, image.size)
    if max(w, h) > max_size:
        if w > h:
            h = int(h / (w / max_size) // patch_size * patch_size)
            w = max_size
        else:
            w = int(w / (h / max_size) // patch_size * patch_size)
            h = max_size

    def _convert_to_rgb(image):
        return image.convert('RGB')

    normalize = Normalize(
        mean=OPENAI_DATASET_MEAN, 
        std=OPENAI_DATASET_STD
    )
    transform = Compose([
        CenterCrop((h, w)),
        _convert_to_rgb,
        to_tensor,
        normalize,
    ])
    ph, pw = h // patch_size, w // patch_size
    return transform(image), (ph, pw)

def image_transform_original_resolution_test(
        image,
        patch_size: int,
    ):
    w, h = map(lambda x: x // patch_size * patch_size, image.size)
    normalize = Normalize(
        mean=OPENAI_DATASET_MEAN, 
        std=OPENAI_DATASET_STD
    )
    transform = Compose([
        Resize((h, w), interpolation=InterpolationMode.BICUBIC),
        _convert_to_rgb,
        to_tensor,
        normalize,
    ])
    return transform(image)

def image_transform(
    image_size: Union[int, Tuple[int, int]],
    is_train: bool,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    normalize = Normalize(mean=mean, std=std)

    if is_train:
        return Compose([
            RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
            _convert_to_rgb,
            # ToTensor(),
            to_tensor,
            normalize,
        ])
    else:
        return Compose([
            Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            _convert_to_rgb,
            # ToTensor(),
            to_tensor,
            normalize,
        ])

def norm_img_vq(img):
    arr = np.array(img)
    arr = arr.astype(np.float32) / 127.5 - 1
    img = torch.from_numpy(np.transpose(arr, [2, 0, 1]))
    return img


def prepare_image(img):
    """ Transform and normalize PIL Image to tensor. """
    transform = Compose([
            RandomResizedCrop(512, scale=(1., 1.), ratio=(1., 1.), interpolation=InterpolationMode.BICUBIC),
        ])
    pil_image = transform(img)
    arr = np.array(pil_image.convert("RGB"))
    arr = arr.astype(np.float32) / 127.5 - 1
    return torch.from_numpy(np.transpose(arr, [2, 0, 1]))


def image_transform_vq(
    image_size: Union[int, Tuple[int, int]],
    is_train: bool,
):

    # if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
    #     # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
    #     image_size = image_size[0]
    def _convert_to_rgb(image):
        return image.convert('RGB')

    return Compose([
        RandomResizedCrop(image_size, scale=(1., 1.), ratio=(1., 1.), interpolation=InterpolationMode.BICUBIC),
        _convert_to_rgb,
        norm_img_vq
    ])


def DiffAugment(x, policy='color,translation,cutout', is_tensor=True, channels_first=True):
    if policy:
        if not is_tensor and not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not is_tensor and not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    return x


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}