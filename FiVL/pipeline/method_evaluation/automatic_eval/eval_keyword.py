
import json 
from tqdm import tqdm
import os
import copy
import time
import httpx
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider    
from prompts_utils import *
import argparse

endpoint = "your_endpoint"
model = "gpt-4o"
api_key = "your_api_key"
num_samples = 500


 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='eval_outputs')
    parser.add_argument('--segments', type=str, default="fivl-instruct/splits/split_0.json")
    parser.add_argument('--image_dir', type=str, default="training/LLaVA/playground/data")

    args = parser.parse_args()
    client = AzureOpenAI(
        api_key=api_key,  
        api_version="2024-02-01",
        azure_endpoint = endpoint,
        http_client = httpx.Client(verify=False)
    )
    os.makedirs(args.output_dir, exist_ok=True)
    save_file_path = os.path.join(args.output_dir, "gpt4_key_eval.json")
    data_to_eval = json.load(open(args.segments, "r"))[:num_samples]

    eval_data = []
    for sample in tqdm(data_to_eval, total=len(data_to_eval)):
        data_url = convert_img_path_to_url(sample, args.image_dir)        
        conversations = sample['conversations']
        for seg in sample['segment']:
            turn = seg['turn']
            for prompt in seg['prompts']:
                question = conversations[turn-1]['value'].replace("<image>", "").strip()
                key_expression = prompt['prompt'].strip()
                input_promt = verification_of_keywords_prompts.format(question=question, key_expression=key_expression)
                response = get_response(client, model, input_promt, data_url, inference_length=5) 
                prompt['prompts_gpt_verify_output'] = response
        eval_data.append(sample)
    json.dump(eval_data, open(save_file_path, "w"), indent=4)
