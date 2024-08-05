import copy
import io
import os
import json
from functools import partial
from typing import Sequence, Dict, Union, Tuple
from dataclasses import dataclass
import numpy as np
from einops import rearrange
from PIL import Image
import random
import torch
import torchvision
from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision import transforms

from .constants import ASPECT_RATIO_256, ASPECT_RATIO_512, ASPECT_RATIO_1024, DEFAULT_IMAGE_FILE_SUFFIX
from .transform import image_transform, image_transform_original_resolution, image_transform_original_resolution_test


aspect_ratio_database = {
    256: ASPECT_RATIO_256,
    512: ASPECT_RATIO_512,
    1024: ASPECT_RATIO_1024
}

def ratio_sample(ratio, aspect_ratios=ASPECT_RATIO_1024):
        closest_ratio = min(aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))
        return closest_ratio

def find_image(sample):
    for suffix in DEFAULT_IMAGE_FILE_SUFFIX:
        if suffix in sample.keys():
            sample['0.jpg'] = sample[suffix]
            break
    return sample

def get_cc3m_wds_dataset_and_collator(data_args, model_args):
    img_size = model_args.image_size
    train_processor = image_transform(img_size, is_train=True)
    val_processor = image_transform(img_size, is_train=False)

    data = load_dataset("webdataset", data_dir=data_args.dataset_path, split="train", streaming=True)
    data = data.shuffle(buffer_size=2_000, seed=data_args.seed)

    def decode(sample, img_processor):
        sample = find_image(sample)
        sample['image'] = img_processor(sample['jpg'])
        sample['text'] = sample['txt']
        return sample
    data = data.map(
        partial(decode, img_processor=train_processor),
        remove_columns=['__key__', '__url__']
    )
    data = data.filter(lambda sample: 'image' in sample and 'text' in sample) # filter return samples that match the given condition
    data_collator = CC3M_WebdatasetCollator(model_args.patch_size)

    return data, data_collator

def get_wds_dataset_and_collator(data_args, model_args):
    img_size = model_args.image_size

    train_processor = image_transform(img_size, is_train=True) if model_args.fixed_image_size else image_transform
    data = load_dataset("webdataset", data_dir=data_args.dataset_path, split="train", streaming=True)
    data = data.shuffle(buffer_size=2_000, seed=data_args.seed)

    def decode(sample, img_processor):
        sample = find_image(sample)
        if model_args.fixed_image_size:
            sample['0.jpg'] = img_processor(sample['0.jpg'])
        return sample

    data = data.map(
        partial(decode, img_processor=train_processor),
        remove_columns=['__key__', '__url__']
    )
    data = data.filter(lambda sample: '0.jpg' in sample)
    data = data.rename_columns({'0.jpg': 'image'})

    aspect_ratios = aspect_ratio_database.get(model_args.image_size, None)
    aspect_ratios = aspect_ratios or ASPECT_RATIO_512

    data_collator = WebdatasetCollator(model_args.fixed_image_size, model_args.patch_size, aspect_ratios)
    import ipdb
    ipdb.set_trace()
    return data, data_collator

def get_wds_dataset_and_collator_arbitrary_resolution(data_args, model_args):

    data = load_dataset("webdataset", data_dir=data_args.dataset_path, split="train", streaming=True)
    data = data.shuffle(buffer_size=2_000, seed=data_args.seed)

    def decode_sample(sample, img_processor):
        sample = find_image(sample)
        sample['0.jpg'], sample['size'] = img_processor(sample['0.jpg'])
        return sample
    
    data = data.map(
        partial(
            decode_sample, 
            img_processor=partial(image_transform_original_resolution, patch_size=model_args.patch_size)
        ),
        remove_columns=['__key__', '__url__']
    )
    data = data.filter(lambda sample: '0.jpg' in sample and sample['0.jpg'].ndim == 3 and sample['0.jpg'].shape[-1] > 0 and sample['0.jpg'].shape[-2] > 0) # filter return samples that match the given condition
    data = data.rename_columns({'0.jpg': 'image'})
    data_collator = WebdatasetCollator(model_args.patch_size)
    
    return data, data_collator

def dataset_test(data_args, model_args):
    from datasets import load_dataset
    import numpy as np
    from functools import partial
    from torchvision import transforms
    OPENAI_DATASET_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
    OPENAI_DATASET_STD = np.array([0.26862954, 0.26130258, 0.27577711])
    DEFAULT_IMAGE_FILE_SUFFIX = ['jpg', '0.jpg', '0.png', 'png', 'jpeg', '0.jpeg', 'webp']
    data = load_dataset("webdataset", data_dir="/share/project/datasets/laion-high-resolution/*/*.tar", split="train", streaming=True)
    data = data.shuffle(buffer_size=2_000, seed=1)
    
    data_iter = iter(data)

    def decode_sample(sample, img_processor):
        sample = find_image(sample)
        sample['0.jpg'], sample['size'] = img_processor(sample['0.jpg'])
        return sample

    data = data.map(
        partial(
            decode_sample, 
            img_processor=partial(image_transform_original_resolution_test, patch_size=16)
        ),
        remove_columns=['__key__', '__url__']
    )
    data = data.filter(lambda sample: '0.jpg' in sample) # filter return samples that match the given condition
    data = data.rename_columns({'0.jpg': 'image'})
    data_collator = WebdatasetCollator()
    
    return data, data_collator

def collate_anyres(images, sizes, patch_size, max_size=2048):
    """
    Args:
    * images: list of images
    * sizes: list of image sizes in (ph, pw), i.e., number of patches in h and w
    
    Return: args accepted by VQModel
    * pixel_values: packed images
    * cu_seqlens_img
    * max_seqlen_img
    * grid_hw
    * image_sizes
    """
    b, c = len(images), images[0].shape[0]
    max_patch_num = max_size // patch_size

    image_sizes = torch.tensor([(image.shape[1], image.shape[2]) for image in images])
    H, W = image_sizes.max(dim=0).values
    padded_images = images[0].new_zeros(size=(b, c, H.item(), W.item()))

    h, w = torch.tensor(sizes).max(dim=0).values
    padding_masks = torch.zeros(size=(b, h.item(), w.item()), dtype=torch.bool)

    for i, (image, mask_size) in enumerate(zip(images, sizes)):
        padded_images[i, :, : image.shape[1], : image.shape[2]].copy_(image)
        padding_masks[i, : mask_size[0], : mask_size[1]] = 1

    padded_images = padded_images.reshape(b, c, h, patch_size, w, patch_size)
    padded_images = torch.einsum("nchpwq->nhwpqc", padded_images)
    padded_images = padded_images.reshape(b, h, w, -1)
    packed_images = padded_images[padding_masks]

    seq_lens = padding_masks.flatten(1, 2).sum(dim=-1)
    cu_seqlens_img = torch.nn.functional.pad(
        torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0)
    )
    max_seqlen_img = seq_lens.max()

    grid_h = torch.arange(0, h)[None, :, None].repeat(b, 1, w)
    grid_w = torch.arange(0, w)[None, None, :].repeat(b, h, 1)
    grid_hw = grid_h[padding_masks] * max_patch_num + grid_w[padding_masks]
    
    return packed_images, cu_seqlens_img, max_seqlen_img, grid_hw, torch.tensor(sizes)
@dataclass
class CC3M_WebdatasetCollator:
    def __init__(self, patch_size: int = 1):
        self.patch_size = patch_size
        self.count = 0

    def __call__(
        self, 
        samples: Sequence[Dict],
        ) -> Dict[str, torch.Tensor]:

        self.count += 1
        images = [sample["image"] for sample in samples]
        texts = [sample["text"] for sample in samples]

        if "size" in samples[0]:
            sizes = [sample['size'] for sample in samples]

        batch = {}

        if all(x is not None and x.shape == images[0].shape for x in images):
            batch['image'] = torch.stack(images)
        else:
            if "size" in samples[0]:
                batch['image'], batch['cu_seqlens_img'], \
                    batch['max_seqlen_img'], batch['grid_hw'], \
                        batch['image_sizes'] = collate_anyres(images, sizes, self.patch_size)
            else:
                batch['image'] = images
        batch['text'] = texts
        return batch

@dataclass
class WebdatasetCollator:
    def __init__(self,fixed_image_size: bool = True, patch_size: int = 8, aspect_ratios=ASPECT_RATIO_512):
        self.fixed_image_size = fixed_image_size
        self.patch_size = patch_size
        self.aspect_ratios = aspect_ratios
        self.count = 0

    def __call__(
        self, 
        samples: Sequence[Dict],
        ) -> Dict[str, torch.Tensor]:

        self.count += 1
        images = [sample["image"] for sample in samples]

        if "size" in samples[0]:
            sizes = [sample['size'] for sample in samples]

        if "size" not in samples[0] and not self.fixed_image_size:
            np.random.seed(self.count)

            aspect_ratio = np.random.choice(list(self.aspect_ratios.keys()))
            image_sizes = [int(x) for x in self.aspect_ratios[aspect_ratio]]
            img_processor = image_transform(image_sizes, is_train=True)
            images = [img_processor(image) for image in images]

        batch = {}

        if all(x is not None and x.shape == images[0].shape for x in images):
            batch['pixel_values'] = torch.stack(images)
        else:
            if "size" in samples[0]:
                batch['pixel_values'], batch['cu_seqlens_img'], \
                    batch['max_seqlen_img'], batch['grid_hw'], \
                        batch['image_sizes'] = collate_anyres(images, sizes, self.patch_size)
            else:
                batch['pixel_values'] = images
        return batch

def anyres_process_images_for_model(image_path=None, pil_image=None, patch_size=32):
    """
    given a list of image_path or pil_image, transform to input to model
    """
    if image_path is not None:
        assert pil_image is None
        if not isinstance(image_path, list):
            image_path = [image_path]
        pil_image = []
        for p in image_path:
            pil_image.append(Image.open(p).convert('RGB'))
    if not isinstance(pil_image, list):
        pil_image = [pil_image]
    
    if len(pil_image) % 2 != 0:
        pil_image.append(pil_image[-1])
    
    image_tensors, sizes = [], []
    for pil_i in pil_image:
        image_tensor, size = image_transform_original_resolution(image=pil_i, patch_size=patch_size)
        image_tensors.append(image_tensor)
        sizes.append(size)
    
    pixel_values, cu_seqlens_img, max_seqlen_img, grid_hw, image_sizes = collate_anyres(image_tensors, sizes, patch_size)
    
    return {
        'pixel_values': pixel_values,
        'cu_seqlens_img': cu_seqlens_img,
        'max_seqlen_img': max_seqlen_img,
        'grid_hw': grid_hw, 
        'image_sizes': image_sizes
    }

def get_in1k_val_dataset(data_args, model_args):
    import torchvision
    transform = image_transform(model_args.image_size, is_train=False)
    dataset = torchvision.datasets.ImageFolder(root="/share/project/datasets/ImageNet/val", transform=transform)
    def in1k_collator(samples):
        if model_args.gan_loss_weight:
            return {"pixel_values": torch.stack([sample[0] for sample in samples]), "optimizer_idx": 0}
        return {"pixel_values": torch.stack([sample[0] for sample in samples])}
    def in1k_collator_anyres(samples):
        images = [sample[0] for sample in samples]
        sizes = [[image.shape[1] // model_args.patch_size, image.shape[2] // model_args.patch_size] for image in images]
        b, c = len(images), images[0].shape[0]
        max_patch_num = 1024 // model_args.patch_size
        
        image_sizes = torch.tensor([(image.shape[1], image.shape[2]) for image in images])
        H, W = image_sizes.max(dim=0).values
        padded_images = images[0].new_zeros(size=(b, c, H.item(), W.item()))

        h, w = torch.tensor(sizes).max(dim=0).values
        padding_masks = torch.zeros(size=(b, h.item(), w.item()), dtype=torch.bool)
        
        for i, (image, mask_size) in enumerate(zip(images, sizes)):
            padded_images[i, :, : image.shape[1], : image.shape[2]].copy_(image)
            padding_masks[i, : mask_size[0], : mask_size[1]] = 1
        
        padded_images = padded_images.reshape(b, c, h, model_args.patch_size, w, model_args.patch_size)
        padded_images = torch.einsum("nchpwq->nhwpqc", padded_images)
        padded_images = padded_images.reshape(b, h, w, -1)
        packed_images = padded_images[padding_masks]
        
        seq_lens = padding_masks.flatten(1, 2).sum(dim=-1)
        cu_seqlens_img = torch.nn.functional.pad(
            torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0)
        )
        max_seqlen_img = seq_lens.max()
        
        grid_h = torch.arange(0, h)[None, :, None].repeat(b, 1, w)
        grid_w = torch.arange(0, w)[None, None, :].repeat(b, h, 1)
        grid_hw = grid_h[padding_masks] * max_patch_num + grid_w[padding_masks]
        
        batch = {}
        batch['pixel_values'] = packed_images
        batch['cu_seqlens_img'] = cu_seqlens_img
        batch['max_seqlen_img'] = max_seqlen_img
        batch['grid_hw'] = grid_hw
        batch['image_sizes'] = torch.tensor(sizes)
        if model_args.gan_loss_weight:
            batch["optimizer_idx"] = 0
        return batch
    
    return dataset, in1k_collator_anyres if data_args.arbitrary_resolution else in1k_collator
    

def split_val_set():
    data = load_dataset("webdataset", data_dir="/share/project/datasets/laion-high-resolution/*/*.tar", split="train", streaming=True)
    data = data.shuffle(buffer_size=100_000, seed=100)
    val_set = []
    val_ids = set()
    for i, item in enumerate(data):
        item_id = item['__url__'] + item['__key__']
        if i == 50_000:
            break
        val_set.append(item)
        val_ids.add(item_id)
    with open("/share/project/datasets/laion-high-resolution/50k_eval_ids.pkl", "wb") as f:
        pickle.dump(val_ids, f)
        
    import webdataset as wds
    from PIL import Image
    from pathlib import Path
    for i in range(50):
        sink = wds.TarWriter(f"/share/project/datasets/laion-high-resolution/eval/eval_{i}.tar")
        for sample in val_set[i * 1000: (i + 1) * 1000]:
            sink.write(sample)
        sink.close()

def preprocess_val_data():
    data = load_dataset("webdataset", data_dir="/share/project/datasets/laion-high-resolution/eval/*.tar", split="train", streaming=True)
    save_dir = "/share/project/datasets/laion-high-resolution/eval/"
    info = []
    def decode_sample(sample):
        sample = find_image(sample)
        sample['0.jpg'], sample['size'] = image_transform_original_resolution(sample['0.jpg'], patch_size=32)
        return sample
    for i, item in tqdm(enumerate(data)):
        image_file = save_dir + f"image_{i}.pkl"
        # process image -> size, image pkl
        item = decode_sample(item)
        with open(image_file, "wb") as f:
            pickle.dump(item['0.jpg'], f)
        info.append({
            "image_path": image_file,
            "size": item['size']
        })
        print(i)
    with open(save_dir + f"image_info.pkl", "wb") as f:
        pickle.dump(info, f)

class HighresEvalDataset(Dataset):
    def __init__(self):
        with open("/share/project/datasets/laion-high-resolution/eval/image_info.pkl", "rb") as f:
            self.info = pickle.load(f)
    
    def __getitem__(self, index):
        info = self.info[index]
        image_path, size = info['image_path'], info['size']
        with open(image_path, "rb") as f:
            image = pickle.load(f)
        return {"image": image, "size": size}
    
    def __len__(self):
        return len(self.info)

def get_highres_eval_dataset(data_args, model_args):
    data = HighresEvalDataset()
    data_collator = WebdatasetCollator(model_args.patch_size)
    
    return data, data_collator