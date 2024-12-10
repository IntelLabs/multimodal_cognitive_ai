import json
import os
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils_xai import *
import torch.nn.functional as F
import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='llava-hf/llava-1.5-7b-hf')
    parser.add_argument('--path_to_grounded', type=str, default="fivl-instruct/dataset_grounded")
    parser.add_argument('--path_to_segments', type=str, default=f"fivl-instruct/splits/split_0.json")
    parser.add_argument('--path_to_images', type=str, default="dataset")
    parser.add_argument('--metric', type=str, default="spearman") #can be "mse"
    parser.add_argument('--output_dir', type=str, default="outputs/head_summary")

    args = parser.parse_args()
    device='cuda'
    data = load_data(args.path_to_segments)
    model, processor = model_factory(args.model_id, device=device)
    output_dir = os.path.join(args.output_dir, args.model_id, args.metric)
    os.makedirs(output_dir, exist_ok=True)
    dist_list = []
    num_heads=model.base_model.language_model.config.num_attention_heads
    num_layers=model.base_model.language_model.config.num_hidden_layers
    patches_w, patches_h =  int(model.vision_tower.config.image_size/model.vision_tower.config.patch_size), int(model.vision_tower.config.image_size/model.vision_tower.config.patch_size)
    for sample in tqdm(data):
        sample_id = sample['id']
        conversations = [s for s in sample['conversations'] if s['from']=='human']
        segment = sample['segment']
        image_path = os.path.join(args.path_to_images, sample['image'])
        image = Image.open(image_path)
        orig_w, orig_h = image.size
        for turn, (segment_turn, conv_turn) in enumerate(zip(segment, conversations)):
            question=conv_turn['value']
            inputs, generated_token_ids, generated_tokens, attentions, _ = model_generate(args.model_id, model, processor, question, image_path)
            input_ids = inputs['input_ids']
            only_generated_token_ids = generated_token_ids[0][len(input_ids[0]):]
            out_attentions_token_gen_vision = extract_relevant_attentions_and_mask(attentions, input_ids, generated_token_ids, processor, patches_w, patches_h)
            keytokens_indices, mask_map = extract_keytokens_for_small_segmap(args.path_to_grounded, segment_turn, patches_w, patches_h,  orig_w, orig_h, only_generated_token_ids, input_ids, processor)
            for keytokens_index in keytokens_indices:
                out_attentions_token_gen_vision_for_keytoken = out_attentions_token_gen_vision[:, :, keytokens_index]
                dist_all = compute_metric(args.metric, out_attentions_token_gen_vision_for_keytoken, mask_map, num_layers, num_heads)
                dist_list.append(dist_all)
                if len(dist_list)%10==0 and len(dist_list)>0:
                    plot_head_summary(dist_list, args.metric, output_dir)




                                


