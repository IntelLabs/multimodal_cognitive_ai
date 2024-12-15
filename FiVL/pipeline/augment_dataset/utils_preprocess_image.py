# PATH_TO_LLAVA='/home/gbenmele/lvlm/LLaVA_gabi' #"/home/gbenmele/lvlm/llava_updated/LLaVA"  #
import sys, os
# sys.path.append(PATH_TO_LLAVA)
import json
from PIL import Image
import torch
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from io import BytesIO
import base64
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
from transformers import StoppingCriteria


# def load_image_from_base64(image):
#     return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, image_aspect_ratio):
    #image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image_processor.do_normalize=False # otherwise the colors are off (reconstruct image)
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images

def save_process_img_for_llava(image_path, output_dir, image_aspect_ratio="pad"):
    model_path= "openai/clip-vit-large-patch14-336"
    model_base=None
    model_name=model_path
    load_8bit=True
    load_4bit=False
    device='cuda'
    image_processor = CLIPProcessor.from_pretrained(pretrained_model_name_or_path=model_path, device=device)
    image = Image.open(image_path).convert("RGB") 
    images = [image]
    images = process_images(images, image_processor.image_processor, image_aspect_ratio)
    pil_image = to_pil_image(images[0])

    return pil_image

# def convert_mask_image_to_mask_vector(mask_image_path, num_patches_y = 24, num_patches_x = 24):
#     image = Image.open(mask_image_path).convert('L')
#     image_array = np.array(image)
#     height, width = image_array.shape
#     assert height == 336 & width == 336, print("image size is not 336 x 336")
#     patch_height = height // num_patches_y
#     patch_width = width // num_patches_x
#     patch_means = np.zeros((num_patches_y, num_patches_x))
#     for i in range(num_patches_y):
#         for j in range(num_patches_x):
#             start_y = i * patch_height
#             end_y = start_y + patch_height
#             start_x = j * patch_width
#             end_x = start_x + patch_width
#             patch = image_array[start_y:end_y, start_x:end_x]
#             patch_mean = np.mean(patch)/255.0
#             patch_means[i, j] = patch_mean
#     return np.array(patch_means)

def reshape_mask_to_mask_vector(image_array, num_patches_y = 24, num_patches_x = 24):
    height, width = image_array.shape
    assert height == 336 & width == 336, print("image size is not 336 x 336")
    patch_height = height // num_patches_y
    patch_width = width // num_patches_x
    patch_means = np.zeros((num_patches_y, num_patches_x))
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            start_y = i * patch_height
            end_y = start_y + patch_height
            start_x = j * patch_width
            end_x = start_x + patch_width
            patch = image_array[start_y:end_y, start_x:end_x]
            patch_mean = np.mean(patch)
            patch_means[i, j] = patch_mean
    return torch.tensor(patch_means)