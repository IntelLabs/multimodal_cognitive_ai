# CLIP-InterpreT
This repository contains the code for CVPR 2024 demo [CLIP-InterpreT: An interpretability Tool for CLIP-like Models
](https://intellabs.github.io/multimodal_cognitive_ai/CLIP-InterpreT/). CLIP-InterpreT is an interpretability tool for exploring the inner workings of CLIP-like foundational models. CLIP is one of the most popular vision-language foundational models and is heavily utilized as a base model when developing new models for tasks such as video retrieval, image generation, and visual navigation. Hence, it is critical to understand the inner workings of CLIP. This tool supports a wide-range of CLIP-like models and provides five types of interpretability analyses: 

+ Property-based nearest neighbors search
+ Topic Segmentation
+ Contrastive Segmentation
+ Nearest neighbors of an image
+ Nearest neighbors for a text input.

# Install
1. Clone this repository and navigate to CLIP-InterpreT folder
```
git clone https://github.com/intel-sandbox/CLIP-InterpreT/
cd CLIP-InterpreT
```
2. Install required packages
```
conda env create -f environments.yml
pip install -r requirements.txt
```
# Running the script
## Step-1
We need to compute image representations of last 4 years of the CLIP model. You need to do this just once. To do so run:
```
cd src
python get_image_rep.py --modelname ViT-B-16_openai --imagepath data/ --batchsize 1
```
For our experiments we used images of [ImageNet validation data](https://www.image-net.org/). You can use any other bigger dataset to get even more fine-grained results. We used six CLIP models:
+ ViT-B-16_laion2b_s34b_b88k
+ ViT-B-16_openai
+ ViT-B-32_datacomp_m_s128m_b4k
+ ViT-B-32_openai
+ ViT-L-14_laion2b_s32b_b82k
+ ViT-L-14_openai

**Note: You need attention weights of all these models to run the application.**
## Step-2
To run the app:
```
cd src
python main.py
```
