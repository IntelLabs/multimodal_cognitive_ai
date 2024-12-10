from mimetypes import guess_type
import base64
import json 
from tqdm import tqdm
import httpx
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider    
import json 
from PIL import Image
from tqdm import tqdm
import os
import time
import argparse
from prompts_utils import *
from transformers import CLIPProcessor, CLIPModel, StoppingCriteria


 
endpoint = "your_endpoint"
api_key = "your_api_key"
model = "gpt-4o"
num_samples=500

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='eval_outputs')
    parser.add_argument('--segments', type=str, default="fivl-instruct/splits/split_0.json")
    parser.add_argument('--new_segments_dir', type=str, default="eval_outputs/segmented_images_masked")
    parser.add_argument('--dataset_grounded', type=str, default="fivl-instruct/dataset_grounded")
    parser.add_argument('--image_dir', type=str, default="training/LLaVA/playground/data/")

    args = parser.parse_args()
    client = AzureOpenAI(
        api_key=api_key,  
        api_version="2024-02-01",
        azure_endpoint = endpoint,
        http_client = httpx.Client(verify=False)
    )
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.new_segments_dir, exist_ok=True)
    save_file_path = os.path.join(args.output_dir, "gpt4_seg1_eval.json")
    data_to_eval = json.load(open(args.segments, "r"))[:num_samples]

    model_path= "openai/clip-vit-large-patch14-336"   
    device='cuda'
    image_processor = CLIPProcessor.from_pretrained(pretrained_model_name_or_path=model_path, device=device)
    eval_data = []
    for sample in tqdm(data_to_eval, total=len(data_to_eval)):
        conversations = sample['conversations']
        image_path = sample['image']
        image_path = os.path.join(args.image_dir, image_path)
        list_mask_image_path = []
        for seg in sample['segment']:
            turn = seg['turn']
            for prompt in seg['prompts']:
                key_expression = prompt['prompt'].strip()
                basename = os.path.basename(prompt['mask_rel']).replace("pt", "png")
                mask_image_path = os.path.join(args.new_segments_dir, f"masked_{basename}")
                if os.path.exists(mask_image_path):
                    data_url = local_image_to_data_url(mask_image_path)
                else:
                    mask_image_path = save_process_img_for_llava(image_path, args.new_segments_dir, os.path.join(args.dataset_grounded, prompt['mask_rel']), image_processor, image_aspect_ratio="pad", inverse_mask=False)
                    data_url = local_image_to_data_url(mask_image_path)
                input_promt = verification_of_segmentation_prompts_v1.format(key_expression=key_expression)
                response = get_response(client, model, input_promt, data_url, 10)
                prompt['gpt_segmentation_eval'] = response
        eval_data.append(sample)

json.dump(eval_data, open(save_file_path, "w"), indent=4)
