<div align='center'>

<h2><a href="https://arxiv.org/abs/2407.20171">Diffusion Feedback Helps CLIP See Better</a></h2>

[Wenxuan Wang](https://scholar.google.com/citations?user=75OyC-oAAAAJ&hl=zh-CN)<sup>1,2,3*</sup>, [Quan Sun](https://scholar.google.cz/citations?user=pVKiHdEAAAAJ&hl=zh-CN&oi=ao)<sup>3*</sup>, [Fan Zhang](https://scholar.google.cz/citations?hl=zh-CN&user=VsJ39HMAAAAJ&view_op=list_works&sortby=pubdate)<sup>3</sup>, [Yepeng Tang](https://scholar.google.cz/citations?user=CAC_4OUAAAAJ&hl=zh-CN&oi=ao)<sup>4</sup>, [Jing Liu](https://scholar.google.com/citations?user=sOI-S7oAAAAJ&hl=zh-CN)<sup>1,2</sup>, [Xinlong Wang](https://scholar.google.com/citations?hl=zh-CN&user=DPz0DjYAAAAJ&view_op=list_works&sortby=pubdate/)<sup>3</sup>
 
<sup>1</sup>[CASIA](http://english.ia.cas.cn/), <sup>2</sup>[UCAS](https://english.ucas.ac.cn/), <sup>3</sup>[BAAI](https://www.baai.ac.cn/english.html), <sup>4</sup>[BJTU](https://en.bjtu.edu.cn/) <br><sup>*</sup> Equal Contribution <br>


</div>


## ⏰ Schedule

### [2025-01-23] Our [paper](https://arxiv.org/abs/2407.20171) is accepted by ICLR 2025 ! 💥
### [2024-08-07] We release [CLIP model weights](https://huggingface.co/BAAI/DIVA) ! 💥  
### [2024-08-05] We release [training & evaluation code](https://github.com/baaivision/DIVA) ! 💥  
### [2024-07-30] Our [paper](https://arxiv.org/abs/2407.20171) is released on arXiv ! 💥


## 💡 Motivation

<p align="center">
    <img src="assets/introduction.png" alt="overview" width="800" />
</p>

In this work, we present a simple post-training approach for CLIP models, which largely overcomes its visual shortcomings via a self-supervised diffusion process. We introduce DIVA, which uses the DIffusion model as a Visual Assistant for CLIP. Specifically, DIVA leverages generative feedback from text-to-image diffusion models to optimize CLIP representations, with only images (w/o corresponding text). We demonstrate that DIVA improves CLIP's performance on the challenging MMVP-VLM benchmark which assesses fine-grained visual abilities to a large extent (e.g., 3-7% ↑), and enhances the performance of MLLMs and vision models on multimodal understanding and segmentation tasks. Extensive evaluation on 29 image classification and retrieval benchmarks confirms that DIVA preserves CLIP's strong zero-shot capabilities.


## 🤖 Architecture

<p align="center">
    <img src="assets/methodology.png" alt="overview" width="800" />
</p>

Given an image, the CLIP model encodes the visual features as the main part of condition, then the generative diffusion model predicts the added noise taking the noisy image and condition as input. We optimize the CLIP's representation by maximizing the image likelihood with the diffusion loss via generative feedback.


## 🔨 Installation
Clone this repository and install the required packages:

```shell
git clone https://github.com/baaivision/DIVA.git
cd DIVA
mkdir -p outputs logs datasets pretrained_weights/CLIP pretrained_weights/SD

conda create -n diva python=3.9
conda activate diva
pip install -r requirements.txt
```
Core packages: 
- [Pytorch](https://pytorch.org/) version 2.0.0
- [open-clip-torch](https://github.com/mlfoundations/open_clip) version 2.24.0
- [timm](https://github.com/rwightman/pytorch-image-models) version 0.9.8


## 🍹 Preparation for DIVA's Generative Fine-tuning

### Data Acquisition
For data preparation, please refer to [image2dataset](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md) and [MMVP](https://github.com/tsb0601/MMVP/tree/main) for the employed training and evaluation data in this work. After collecting the corresponding datasets, directly put them into the `dataset/` folder path. 

### Pre-trained Weight Downloading
As for pre-trained weight preparation, please refer to [OpenAI ViT-L-14/224&336](https://github.com/openai/CLIP/blob/main/clip/clip.py), [MetaCLIP ViT-L/H-14](https://github.com/facebookresearch/metaclip), [SigLIP ViT-SO-14/224](https://huggingface.co/timm/ViT-SO400M-14-SigLIP), [SigLIP ViT-SO-14/384](https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384), [DFN ViT-H-14/224](https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14), [DFN ViT-H-14/378](https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378) and [SD-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) to acquire the model weights for discriminative CLIP models and the leveraged diffusion model that provides generative feedback. After downloading all these necessary weights, move them respectively to the corresponding folder path `pretrained_weights/CLIP/` and `pretrained_weights/SD/`.

### Code Modification
For the preparation for our DIVA's condition design, some source code in the installed [CLIP](https://github.com/openai/CLIP) and [OpenCLIP](https://github.com/mlfoundations/open_clip) packages need to be modified.

For OpenAI CLIP, use the content in our provided `condition/OpenAICLIP_for_clip_model.py` to replace the content in `Your Conda Installation Path/anaconda3/envs/diva/lib/python3.9/site-packages/clip/model.py`.

For MetaCLIP and DFN, use the content in our provided `condition/MetaCLIP_for_openclip_transformer.py` and `condition/DFN_for_openclip_transformer.py` to replace the content in `Your Conda Installation Path/anaconda3/envs/diva/lib/python3.9/site-packages/open_clip/transformer.py`, respectively.

For SigLIP, use the content in our provided `condition/SigLIP_for_timm_models_visiontransformer.py` to replace the content in `Your Conda Installation Path/anaconda3/envs/diva/lib/python3.9/site-packages/timm/models/vision_transformer.py`.


## 🍻 Quick Start for Training & Evaluation

After all the above preparation steps, you can simply start training for our DIVA with the following command: 
```shell
# For OpenAICLIP
bash DIVA_for_OpenAICLIP.sh

# For MetaCLIP
bash DIVA_for_MetaCLIP.sh

# For SigLIP
bash DIVA_for_SigLIP.sh

# For DFN
bash DIVA_for_DFN.sh
```

##  Model Zoo

| Method               | Image Size | Params (M) | Average Score |
|----------------------|------------|------------|---------------|
| [OpenAI ViT-L-14](https://huggingface.co/BAAI/DIVA/blob/main/OpenAICLIP/OpenAI-ViT-L-14-224.pth)      | 224²       | 427.6      | 25.9 (+6.6)   |
| [OpenAI ViT-L-14](https://huggingface.co/BAAI/DIVA/blob/main/OpenAICLIP/OpenAI-ViT-L-14-336.pth)      | 336²       | 427.9      | 25.2 (+5.2)   |
| [MetaCLIP ViT-L-14](https://huggingface.co/BAAI/DIVA/blob/main/MetaCLIP/MetaCLIP-ViT-L-14.pth)    | 224²       | 427.6      | 27.4 (+3.7)   |
| [MetaCLIP ViT-H-14](https://huggingface.co/BAAI/DIVA/blob/main/MetaCLIP/MetaCLIP-ViT-H-14.pth)    | 224²       | 986.1      | 31.9 (+6.7)   |
| [SigLIP ViT-SO-14](https://huggingface.co/BAAI/DIVA/blob/main/SigLIP/SigLIP-ViT-SO-14-224.pth)     | 224²       | 877.4      | 40.7 (+2.9)   |
| [SigLIP ViT-SO-14](https://huggingface.co/BAAI/DIVA/blob/main/SigLIP/SigLIP-ViT-SO-14-384.pth)     | 384²       | 878.0      | 38.5 (+1.5)   |
| [DFN ViT-H-14](https://huggingface.co/BAAI/DIVA/blob/main/DFN/DFN-ViT-H-14-224.pth)        | 224²       | 986.1      | 43.7 (+4.4)   |
| [DFN ViT-H-14](https://huggingface.co/BAAI/DIVA/blob/main/DFN/DFN-ViT-H-14-378.pth)         | 378²       | 986.7      | 37.8 (+3.0)   |


It is worth noting that, due to the randomness among the introduced condition design during the training phase and the selection of local patch tokens during the inference phase for OpenAI CLIP, the obtained scores on MMVP_VLM benchmark using our provided OpenAI CLIP weights might not be the same as the reported results in our paper. At this time, we recommend trying different random seeds multiple times if the scores do not meet expectations. 

## 🎨 Visualization

<p align="center">
    <img src="assets/qualitative_mmvp.png" alt="scene" width="900" />
</p>


## 💙 Acknowledgement
DIVA is built upon the awesome [Diffusion-TTA](https://github.com/mihirp1998/Diffusion-TTA), [MMVP](https://github.com/tsb0601/MMVP), [CLIP](https://github.com/openai/CLIP), [OpenCLIP](https://github.com/mlfoundations/open_clip), [timm](https://github.com/huggingface/pytorch-image-models/). 

## 📝 Citation
```bib
@article{wang2024diffusion,
      title={Diffusion Feedback Helps CLIP See Better},
      author={Wang, Wenxuan and Sun, Quan and Zhang, Fan and Tang, Yepeng and Liu, Jing and Wang, Xinlong},
      journal={arXiv preprint arXiv:2407.20171},
      year={2024}
}
```
