import json
import os
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from tqdm import tqdm
import pandas as pd
from transformers import GenerationConfig
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("/home/estellea/rm_training/estelleafl_private_repo/ai.llava")
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
from llava.conversation import ( conv_templates,
                                   SeparatorStyle)
from llava.mm_utils import get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX
import copy
import inspect
import warnings
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image
import json
import torch.distributed as dist
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
from torchvision.transforms.functional import to_pil_image
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import  logging
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from transformers.generation.beam_search import  BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers import GenerationConfig
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers import (
    StoppingCriteriaList)
from transformers.generation.utils import GenerateOutput

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


def model_factory(model_id, device='cuda'):
    if model_id == 'llava-hf/llava-1.5-7b-hf':
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            output_attentions=True
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_id)
        tokenizer = None
        img_idx = 32000
        num_patches = 576
    else:
        model, processor, tokenizer = build_llava_model(model_id, device)
    return model, processor, tokenizer, {"num_patches": num_patches, "img_idx": img_idx}

def model_generate(model_id, model, processor, tokenizer, question, image_path):
    device=model.device
    if model_id == 'llava-hf/llava-1.5-7b-hf':
        raw_image = Image.open(image_path)
        question_wo_image=question.replace("<image>\n", "")
        conversation = [
            {

            "role": "user",
            "content": [
                {"type": "text", "text": question_wo_image},
                {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(model.device, torch.float16)
        input_ids = inputs['input_ids']
        outputs =model.generate(**inputs, max_new_tokens=200, do_sample=False, output_attentions=True, return_dict_in_generate=True)
        attentions=outputs['attentions']
        generated_token_ids=outputs[0][-len(attentions):]
        generated_tokens= [processor.decode(i) for i in generated_token_ids]
        return inputs, generated_token_ids, generated_tokens, attentions, None
    else:
        model.config.image_aspect_ratio="pad"
        params = {}
        top_p = float(params.get("top_p", 1.0))
        max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
        max_new_tokens = 10
        do_sample = False
        generation_config = {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "max_length": 4096,
        "pad_token_id": 0,
        "transformers_version": "4.31.0",
        "output_attentions": True,
        "return_dict_in_generate": True,
        "output_scores": True,
        }
        generation_config = GenerationConfig(**generation_config)
            
        img = Image.open(image_path)
        text = (question, img, 'Default')
        template_name = 'llava_v1'
        state = conv_templates[template_name].copy()
        state.append_message(state.roles[0], text)
        state.append_message(state.roles[1], None)
        state.skip_next = False
        prompt = state.get_prompt()
        images = [img]
        image_processor = processor
        images = process_images(images, image_processor, model.config)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
        input_ids_len = input_ids.shape[-1]
        stop_str = state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        image_args = {'images': images.to(device, dtype=torch.float16)}
        position_ids = torch.arange(input_ids_len, dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
        inputs_embeds=None
        labels=None
        inputs= {"input_ids": input_ids, "pixel_values":images}
        past_key_values=None
        model.vision_logits_training=False
        outputs = model.generate(
                    inputs=input_ids,
                    do_sample=False,
                    temperature=True,
                    top_p=1.0,
                    generation_config=generation_config,
                    stopping_criteria=[stopping_criteria],
                    use_cache=True,
                    **image_args
                )       
        attentions=outputs['attentions']
        generated_token_ids=outputs[0]
        generated_tokens= [tokenizer.decode(i) for i in generated_token_ids[0]]#processor.decode(generated_token_ids, skip_special_tokens=True)
        return inputs, generated_token_ids, generated_tokens, attentions, images


def model_forward(model_id, model, processor, question, image_path):
    if model_id == 'llava-hf/llava-1.5-7b-hf':
        raw_image = Image.open(image_path)
        question_wo_image=question.replace("<image>\n", "")
        conversation = [
            {

            "role": "user",
            "content": [
                {"type": "text", "text": question_wo_image},
                {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(model.device, torch.float16)
        input_ids = inputs['input_ids']
        outputs =model(**inputs)
        tokens= [processor.decode(i) for i in input_ids]
        return inputs, outputs, tokens

def load_data(path_to_data):
    with open(path_to_data) as f:
        data = json.load(f)
    return data


def plot_heatmap(image, heatmap_mask,alpha, title):
    fig, ax = plt.subplots()
    ax.imshow(np.transpose(image.squeeze().float(), (1, 2, 0)))
    heatmap = ax.imshow(heatmap_mask, cmap='jet', alpha=alpha, interpolation='none') 
    plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.show()
    plt.savefig(f"{title}.png")
    print(f"{title}.png saved")


def plot_heatmap2(mask, token, output_path, image):
    grid_size_temp=(1,1)
    fig, axs = plt.subplots(*grid_size_temp, figsize=(12, 12))
    cmap = plt.get_cmap('jet')
    axs = np.array(axs)
    mask = mask.reshape(336,336).cpu().numpy() 
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = cmap(mask)  #.cpu().numpy())
    mask = Image.fromarray((mask[:, :, :3] * 255).astype(np.uint8)).resize((336,336), Image.BICUBIC)
    mask.putalpha(128)
    img_recovered_grid = to_pil_image(image[0])
    img_recovered_grid.paste(mask, mask=mask)
    ax = axs.flat[0]
    ax.imshow(img_recovered_grid)
    ax.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.suptitle(f"Token: {token}", fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{output_path}.png')
    print(f"Saved the image at {output_path}.png")
    plt.close()
    plt.clf() 

def find_consecutive_indices(A, B):
    max_len = 0
    best_indices = []
    if len(B) == 1:
        try:
            return [torch.where(A == B[0])[0].item()]  # Find the index of the first match
        except IndexError:
            return []  
    for i in range(len(A)):
        current_indices = []
        for j in range(len(B)):
            if i + j < len(A) and A[i + j].item() == B[j].item():
                current_indices.append(i + j)
            else:
                break
                
        if len(current_indices) > max_len:
            max_len = len(current_indices)
            best_indices = current_indices
    
    return best_indices

def find_indices(A,B):
    updated_B = B
    results = []
    while len(results)==0 and len(updated_B)>0:
        results = find_consecutive_indices(A, updated_B)
        updated_B = updated_B[1:]
    return results

def build_llava_model(model_id, device='cuda'):    
    apply_normalization = True #False #
    model_path=model_id
  
    model_base=None
    model_name=model_path
    load_8bit=True
    load_4bit=False
    device='cuda'
    debug = True


    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path), load_8bit=load_8bit, load_4bit=load_4bit, device=device
    )


    return model, image_processor, tokenizer

