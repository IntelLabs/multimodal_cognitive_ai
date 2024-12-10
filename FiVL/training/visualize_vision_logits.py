import json
import os
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse
from torchvision.transforms.functional import to_pil_image

#  python -m debugpy --listen 0.0.0.0:25567 --wait-for-client visualize_vision_logits.py --path_to__segments /export/share/projects/mcai/LLaVA-dataset/dataset_grounded_gpt/mix665k/grounded_maskonly/grounded_split_1_maskonly.json --path_to_images /export/share
# /projects/mcai/LLaVA-dataset/dataset
def plot_mask(mask, token, fig_title, pixel_values, processor, output_dir): 
    img_std = torch.tensor(processor.image_processor.image_std).view(3,1,1)
    img_mean = torch.tensor(processor.image_processor.image_mean).view(3,1,1)
    img_recover = pixel_values[0].cpu() * img_std.cpu() + img_mean.cpu()
    grid_size_temp=(1,1)
    fig, axs = plt.subplots(*grid_size_temp, figsize=(12, 18))
    cmap = plt.get_cmap('jet')
    axs = np.array(axs)
    mask = mask.reshape(24,24).cpu().numpy() 
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = cmap(mask)  #.cpu().numpy())
    mask = Image.fromarray((mask[:, :, :3] * 255).astype(np.uint8)).resize((336,336), Image.BICUBIC)
    mask.putalpha(128)
    img_recovered_grid = to_pil_image(img_recover)
    img_recovered_grid.paste(mask, mask=mask)
    ax = axs.flat[0]
    ax.imshow(img_recovered_grid)
    ax.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.suptitle(f"{fig_title}\nToken: {token}", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{token}.png'))
    print(f"Saved the image at {os.path.join(output_dir, f'{token}.png')}")
    plt.close()
    plt.clf() 

def plot_token(pixel_values, model_id, processor, vision_ids, fig_title, output_dir): 
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    mask_token_dict = {}
    all_token_id=[]
    for i, vision_id in enumerate(vision_ids[0]):
        if int(vision_id) not in mask_token_dict:
            mask_token_dict[int(vision_id)] = torch.zeros(vision_ids.shape[1])
        mask_token_dict[int(vision_id)][i] = 1
        all_token_id.append(int(vision_id))
    for vision_id in set(vision_ids[0].tolist()):
        token = processor.decode(vision_id)
        if "<" not in token and len(token)>1:
            mask_token = mask_token_dict[int(vision_id)]
            plot_mask(mask_token, token, fig_title, pixel_values, processor, output_dir)


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


def load_data(path_to_data):
    with open(path_to_data) as f:
        data = json.load(f)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='llava-hf/llava-1.5-7b-hf')
    parser.add_argument('--path_to_segments', type=str, default="fivl-instruct/splits/split_0.json")
    parser.add_argument('--path_to_images', type=str, default="playground/data")
    parser.add_argument('--output_dir', type=str, default="outputs")

    args = parser.parse_args()
    data = load_data(args.path_to_segments)
    model, processor = model_factory(args.model_id)
    img_token = processor.tokenizer.added_tokens_encoder['<image>']
    num_patches = int((model.vision_tower.config.image_size/model.vision_tower.config.patch_size)**2)
    for sample in tqdm(data):
        conversations = [s for s in sample['conversations'] if s['from']=='human']
        segment = sample['segment']
        image_path = os.path.join(args.path_to_images, sample['image'])
        for turn, (segment_turn, conv_turn) in enumerate(zip(segment, conversations)):
            question=conv_turn['value']
            inputs, outputs, question= model_forward(args.model_id, model, processor, question, image_path)
            pixels=inputs["pixel_values"]
            img_idx = int(torch.where(inputs["input_ids"][0]==img_token)[0][0])
            vision_logits = outputs.logits[:, img_idx:img_idx+num_patches, :]
            vision_ids = vision_logits.argmax(dim=-1)
            img_name = image_path.split("/")[-1].split(".")[0]
            save_dir=f"{args.output_dir}/{img_name}/{turn}"
            fig_title=f"Question: {question}"
            os.makedirs(save_dir, exist_ok=True)
            plot_token(pixels, args.model_id, processor, vision_ids, fig_title, save_dir)