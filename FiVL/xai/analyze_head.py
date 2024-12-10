import json
import os
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from utils_xai import *
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='llava-hf/llava-1.5-7b-hf')
    parser.add_argument('--path_to_grounded', type=str, default="fivl-instruct/dataset_grounded")
    parser.add_argument('--path_to_segments', type=str, default=f"fivl-instruct/splits/split_0.json")
    parser.add_argument('--path_to_images', type=str, default="dataset")
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--head', type=int, default=11)
    parser.add_argument('--output_dir', type=str, default="outputs/head_summary")


    args = parser.parse_args()
    device='cuda'
    data = load_data(args.path_to_segments)
    model, processor = model_factory(args.model_id, device=device)
    output_dir=os.path.join(args.output_dir, args.model_id, args.metric)
    os.makedirs(output_dir, exist_ok=True)
    patches_w, patches_h =  int(model.vision_tower.config.image_size/model.vision_tower.config.patch_size), int(model.vision_tower.config.image_size/model.vision_tower.config.patch_size)
    num_layers=model.base_model.language_model.config.num_hidden_layers
    num_heads=model.base_model.language_model.config.num_attention_heads
    img_token= processor.tokenizer.added_tokens_encoder['<image>']
    for sample in tqdm(data):
        sample_id = sample['id']     
        save_dir = os.path.join(output_dir, sample_id)     
        os.makedirs(save_dir, exist_ok=True)
        conversations = [s for s in sample['conversations'] if s['from']=='human']
        segment = sample['segment']
        image_path = os.path.join(args.path_to_images, sample['image'])
        image = Image.open(image_path)
        orig_w, orig_h = image.size
        for turn, (segment_turn, conv_turn) in enumerate(zip(segment, conversations)):
            question=conv_turn['value']
            inputs, generated_token_ids, generated_tokens, attentions, images = model_generate(args.model_id, model, processor, question, image_path)
            input_ids = inputs['input_ids']
            _,_,w,h = inputs['pixel_values'].shape
            only_generated_token_ids = generated_token_ids[0][len(input_ids[0]):]
            out_attentions_token_gen_vision = extract_relevant_attentions_and_mask(attentions, input_ids, generated_token_ids, processor, patches_w, patches_h)
            only_generated_tokens = [processor.decode(i) for i in only_generated_token_ids]
            layer_seg = args.layer 
            head_seg= args.head
            for i, each_token in enumerate(only_generated_tokens):
                print(f"Layer: {layer_seg}, Head: {head_seg}, Token: {each_token}")
                if "s>" in each_token:
                    continue
                head_interest= out_attentions_token_gen_vision[layer_seg,head_seg,i,:].reshape(24,24).unsqueeze(0).unsqueeze(0)
                head_mask = F.interpolate(head_interest, size=(w,h), mode='bilinear', align_corners=False).squeeze()
                output_path= os.path.join(save_dir, f"turn_{turn}_head_{layer_seg}_{head_seg}_{each_token}.png")
                img_std = torch.tensor(processor.image_processor.image_std).view(3,1,1) #processor or processor.image_processor
                img_mean = torch.tensor(processor.image_processor.image_mean).view(3,1,1)
                images=inputs['pixel_values'].cpu()
                img_recover = images[0] * img_std + img_mean
                plot_heatmap(head_mask, each_token, output_path, img_recover)
