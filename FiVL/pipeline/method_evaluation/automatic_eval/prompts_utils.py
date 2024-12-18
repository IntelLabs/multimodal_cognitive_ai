from mimetypes import guess_type
import base64
from PIL import Image
import os
import numpy as np
import sys
import json
import torch.nn.functional as F
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel, StoppingCriteria

def get_response(client, model, prompt, data_url, inference_length=10):
    response = client.chat.completions.create(
    model=model,
    messages=[
        { "role": "user", "content": [  
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": data_url
                }
            }
        ] }
    ],
    max_tokens=inference_length
)
    response_content = response.choices[0].message.content
    return response_content


def local_image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream' 
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"


def convert_img_path_to_url(sample, image_dir):
    image_path = os.path.join(image_dir, sample['image'])
    image = Image.open(image_path)
    data_url = local_image_to_data_url(image_path)
    return data_url



def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


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


def process_images(images, image_processor, image_aspect_ratio=None):
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image_processor.do_normalize=False # otherwise the colors are off (reconstruct image)
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        image_processor.do_normalize=False # otherwise the colors are off (reconstruct image)
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images

def save_process_img_for_llava(image_path, output_dir, mask_rel_path, image_processor, image_aspect_ratio="pad", inverse_mask=False):   
    image = Image.open(image_path).convert("RGB") 
    images = [image]
    images = process_images(images, image_processor.image_processor, image_aspect_ratio)
    mask_rel_torch = torch.load(mask_rel_path)
    mask = to_pil_image(mask_rel_torch)
    mask_tensor = to_tensor(mask)
    if inverse_mask:
        mask_tensor = 1-mask_tensor
    result_tensor = images[0] * F.interpolate(mask_tensor.unsqueeze(0), size=(images[0].shape[1], images[0].shape[2]), mode='bilinear', align_corners=False).squeeze(0)
    pil_masked_image = to_pil_image(result_tensor)
    original_mask_name = os.path.basename(mask_rel_path)
    new_masked_image_name = f"masked_" + original_mask_name.replace(".pt", ".png")
    masked_image_path = os.path.join(output_dir, new_masked_image_name)
    mask.save(masked_image_path.replace(".png", "_mask.png"))
    pil_masked_image.save(masked_image_path)
    print(f"Saved masked image to {masked_image_path}")
    return masked_image_path



verification_of_keywords_prompts = '''You are given a question, a word/phrase and an image. Please rate the importance degree from 0-10 scale. 
Note that
 - 0 means not important at all and 10 means very important. 
 - Important word/phrase means that this word/phrase is closely related to the image and the question, and it could not be evoked without the use of the image. 
 - If the question does not related to the image, in other words, the answer does not depend on the image content, then any words are not important. 

Question: {question}

A word: {key_expression} 

Only answer important or not important, and the importance degree from 0-10? Then generate the reason to justify your answer.
'''


verification_of_segmentation_prompts_v1 = '''You are given a part of the image and a word/phrase, do you think this is a good segmentation that the given part of the image covers this word/phrase?

Word/phrase: {key_expression}

Answer only "yes" or "no". And then explain your answer.
'''


verification_of_segmentation_prompts_v2 = '''You are given a part of the image and a word/phrase, do you see any part of the image that is related to the word? 

Word/phrase: {key_expression}

Answer only "yes" or "no".
'''