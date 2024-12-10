import json
import os
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

from scipy.stats import spearmanr

def revert_padding(segm_mask_map, w, h):
    aspect_ratio = w / h
    if aspect_ratio > 1: 
        new_width = 24
        new_height = int(24 / aspect_ratio)
    else:  
        new_height = 24
        new_width = int(24 * aspect_ratio)
    left = (24 - new_width) // 2
    top = (24 - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    rectangual_segm_mask_map =  segm_mask_map[:, :, top:bottom, left:right]
    _, _, rec_w, rec_h = rectangual_segm_mask_map.shape
    crop_size = min(rec_w, rec_h) 
    start_x = (rec_w - crop_size) // 2
    start_y = (rec_h - crop_size) // 2
    return rectangual_segm_mask_map[:, :, start_x:start_x + crop_size, start_y:start_y + crop_size]


def find_consecutive_indices(A, B):
    max_len = 0
    best_indices = []
    if len(B) == 1:
        try:
            return [torch.where(A == B[0])[0].item()]  # Find the index of the first match
        except :
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


def model_generate(model_id, model, processor, question, image_path):
    device=model.device
    if model_id == 'llava-hf/llava-1.5-7b-hf':
        raw_image = Image.open(image_path)
        question_wo_image=question.replace("<image>", "").replace("\n", "")
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
        raise NotImplementedError


def model_factory(model_id, device='cuda'):
    if model_id == 'llava-hf/llava-1.5-7b-hf':
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            output_attentions=True
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_id)
    else:
        raise NotImplementedError
    return model, processor


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
    else:
        raise NotImplementedError

def load_data(path_to_data):
    with open(path_to_data) as f:
        data = json.load(f)
    return data


def reshape_output_attentions(attentions):
    all_out_tokens_attention = []
    sen_len=attentions[-1][0].shape[-1]
    out_attentions = torch.zeros(32, 32, sen_len, sen_len)
    for att in attentions:
        all_out_tokens_attention.append(torch.cat(att, dim=0))
    nl, nh, in0_len, out0_len=all_out_tokens_attention[0].shape
    out_attentions[:,:,:in0_len, :out0_len] = all_out_tokens_attention[0]
    curr_in_len=in0_len
    for att in all_out_tokens_attention[1:]:
        _, _ , _, out_len = att.shape
        assert out_attentions[:,:,curr_in_len, :out_len].sum() == 0
        out_attentions[:,:,curr_in_len, :out_len] = att.squeeze(-2)
        curr_in_len+=1
    return out_attentions


def extract_keytokens_for_small_segmap(path_to_grounded, segment_turn, patches_w, patches_h,  orig_w, orig_h, only_generated_token_ids, input_ids, processor):
    segmentation_maps = [os.path.join(path_to_grounded, segment_turn['prompts'][i]['mask_rel']) for i in range(len(segment_turn['prompts']))]
    keytokens = [segment_turn['prompts'][i]['prompt'] for i in range(len(segment_turn['prompts']))]
    min_pixel_seg = patches_w*patches_h
    for segmentation_map, keytoken in zip(segmentation_maps, keytokens):
        segm_mask_map = (torch.load(segmentation_map)/255).reshape(1, 1, patches_w, patches_h)
        sum_pixel_seg = segm_mask_map.sum()
        if sum_pixel_seg < min_pixel_seg:
            segm_mask_map = revert_padding(segm_mask_map, orig_w, orig_h)
            segm_mask_map_resize = F.interpolate(segm_mask_map, size=(patches_w,patches_h), mode='bilinear', align_corners=False).squeeze()
            keytoken_for_min = keytoken
            min_pixel_seg = sum_pixel_seg
    keytoken_ids_for_min = processor(keytoken_for_min)["input_ids"][0][1:]
    keytokens_indices = find_indices(only_generated_token_ids, torch.tensor(keytoken_ids_for_min))
    mask_map = segm_mask_map_resize.reshape(1,1,-1)
    return keytokens_indices, mask_map


def compute_metric(metric, out_attentions_token_gen_vision_for_keytoken, mask_map, num_layers, num_heads):
    if metric=="mse":
        dist_all = ((out_attentions_token_gen_vision_for_keytoken- mask_map)**2).mean(dim=-1).squeeze().cpu().numpy()
    elif metric=="spearman":
        dist_all = np.zeros((num_layers, num_heads))
        for i in range(num_layers):
            for j in range(num_heads):
                att_flat= out_attentions_token_gen_vision_for_keytoken[i, j].cpu().numpy().flatten()
                mask_flat=  mask_map.cpu().numpy().flatten()
                correlation, _ = spearmanr(att_flat, mask_flat)
                dist_all[i, j] = correlation
    return dist_all


def plot_head_summary(dist_list, metric, output_dir): 
    plt.figure(figsize=(8, 6))
    sns.heatmap(sum(dist_list)/len(dist_list), cmap='coolwarm', cbar=True)
    plt.xlabel('heads', fontsize=25) 
    plt.ylabel('layers', fontsize=25) 
    plt.title(f"Head Summary for {metric} averaged on {len(dist_list)} samples", fontsize=25)
    plt.show()
    save_to=f'{output_dir}/vl_alignT_head.png'
    plt.savefig(save_to, bbox_inches='tight')
    print(f"Saved the image at {save_to}")
    plt.close()


def extract_relevant_attentions_and_mask(attentions, input_ids, generated_token_ids, processor, patches_w, patches_h):
    out_attentions = reshape_output_attentions(attentions)
    out_attentions_token_gen = out_attentions[:, :, len(input_ids[0])-1:]
    out_attentions_token_gen.shape[-2] == generated_token_ids.shape[-1] -1 
    img_token= processor.tokenizer.added_tokens_encoder['<image>']
    img_idx = int(torch.where(input_ids[0]==img_token)[0][0])
    out_attentions_token_gen_vision = out_attentions_token_gen[:, :,:, img_idx:img_idx+int(patches_w*patches_h)]
    return out_attentions_token_gen_vision


def plot_heatmap(mask, token, output_path, img_recover):
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('jet') 
    mask = mask.reshape(336,336).cpu().numpy() 
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = cmap(mask)
    mask = Image.fromarray((mask[:, :, :3] * 255).astype(np.uint8)).resize((336,336), Image.BICUBIC)
    mask.putalpha(128)
    img_recovered_grid=to_pil_image(img_recover)
    img_recovered_grid.paste(mask, mask=mask)
    ax.imshow(img_recovered_grid)
    ax.axis('off')
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    ax.set_aspect('equal')
    # plt.suptitle(f"Token: {token}", fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{output_path}')
    print(f"Saved the image at {output_path}")
    plt.close()
    plt.clf() 