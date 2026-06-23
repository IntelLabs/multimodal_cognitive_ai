import os
import os.path as osp
import argparse
from tqdm import tqdm
import pandas as pd
from abc import abstractmethod
import mmap
from transformers import BridgeTowerProcessor, BridgeTowerForImageAndTextRetrieval
from transformers import CLIPProcessor, CLIPModel
from transformers import ViltProcessor, ViltForImageAndTextRetrieval
import requests
from PIL import Image
import torch

n_a = -1000.1
coco_images_path = "/export/share/datasets/COCO2017/val2017"

def assign_exist_image_file_extension(filename, path):
    if os.path.isfile(osp.join(path, filename + '.jpg')):
        filename_with_extension = filename + '.jpg'
    elif os.path.isfile(osp.join(path, filename + '.png')):
        filename_with_extension = filename + '.png'
    else:
        filename_with_extension = filename + '.jpg'
    return filename_with_extension

def get_path_to_coco_image(image_index):
    if '_' in image_index:
        image_index = image_index.split('_')[0]
    image_index = str(image_index)
    filename = '0'*(12-len(image_index)) + image_index 
    filename = assign_exist_image_file_extension(filename, coco_images_path)
    path_to_image_file = osp.join(coco_images_path, filename)
    return path_to_image_file

class Scorer:
    def __init__(self):
        self.n_a = -1000.1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @abstractmethod    
    def compute_prompt_image_matching_score(self, prompt, path_to_image_file):
        pass
        
class BridgeTowerScorer(Scorer):
    def __init__(self):
        super().__init__()
        self.processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm")
        self.model = BridgeTowerForImageAndTextRetrieval.from_pretrained("BridgeTower/bridgetower-large-itm-mlm").to(self.device)
        
    def compute_prompt_image_matching_score(self, prompt, path_to_image_file):
        if osp.exists(path_to_image_file):
            try:
                image = Image.open(path_to_image_file)
                encoding = self.processor(image, prompt, return_tensors="pt").to(self.device)
                outputs = self.model(**encoding)
                score = outputs.logits[0,1].item()
                return score
            except:
                return self.n_a
        else:
            return self.n_a

class ClipScorer(Scorer):
    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
    def compute_prompt_image_matching_score(self, prompt, path_to_image_file):
        if osp.exists(path_to_image_file):
            try:
                image = Image.open(path_to_image_file)
                inputs = self.processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model(**inputs)
                score = outputs.logits_per_text[0,0].item()
                return score
            except:
                return self.n_a
        else:
            return self.n_a
        
class VILTScorer(Scorer):
    def __init__(self):
        super().__init__()
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-coco")
        self.model = ViltForImageAndTextRetrieval.from_pretrained("dandelin/vilt-b32-finetuned-coco").to(self.device)
        
    def compute_prompt_image_matching_score(self, prompt, path_to_image_file):
        if osp.exists(path_to_image_file):
            try:
                image = Image.open(path_to_image_file)
                encoding = self.processor(image, prompt, return_tensors="pt").to(self.device)
                outputs = self.model(**encoding)
                score = outputs.logits[0, 0].item()
                return score
            except:
                return self.n_a
        else:
            return self.n_a
        
def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Image-Text Matching Score Computation for BridgeTower, VITL and Clip',
        usage='python ITM_score_computation.py [<args>] [-h | --help]'
    )

    parser.add_argument('-p', '--prompt_file', default="../output/gen_captions/perplexity_score/sim_.8_.91/cap_val_2017.txt", type=str, help='File including original and counterfactual prompts')
    parser.add_argument('-o', '--original_images', default="../../../coco-counterfactuals/synthetic-images-for-original-captions", type=str, help='Path to folder including original images')
    parser.add_argument('-c', '--counter_images', default="../../../coco-counterfactuals/synthetic-images-for-counterfactual-captions", type=str, help='Path to folder including counter images')
    parser.add_argument('-out', '--output_filename', default="output-ITM-scores/scores.csv", type=str, help='Path to output score filename')
    return parser.parse_args(args)

def create_output_folder(path):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(path)
       print(f'The new directory {path} is created!')


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def compute_summary_score(data):
    df = pd.DataFrame.from_dict(data)
    df['diff_per_origin_prompt_on_ldm'] = df['scores_org_pr_org_img'] - df['scores_org_pr_counter_img']
    df['diff_per_counter_prompt_on_ldm'] = df['scores_counter_pr_counter_img'] - df['scores_counter_pr_org_img']
    df['diff_per_origin_image_on_ldm'] = df['scores_org_pr_org_img'] - df['scores_counter_pr_org_img']
    df['diff_per_counter_image_on_ldm'] = df['scores_counter_pr_counter_img'] - df['scores_org_pr_counter_img']
    
    df['diff_per_origin_prompt_on_coco'] = df['scores_org_pr_coco_img'] - df['scores_org_pr_counter_img']
    df['diff_per_counter_prompt_on_coco'] = df['scores_counter_pr_counter_img'] - df['score_counter_pr_coco_img']
    df['diff_per_coco_image'] = df['scores_org_pr_coco_img'] - df['score_counter_pr_coco_img']
    return df.to_dict('list')

def compute_matching_score_using_model(model, indices, origin_prompts, counter_prompts, origin_image_path, counter_image_path):
    scores_org_pr_org_img = []
    scores_org_pr_counter_img = []
    scores_counter_pr_org_img = []
    scores_counter_pr_counter_img = []
    scores_org_pr_coco_img = []
    score_counter_pr_coco_img = []
    
    for i in tqdm(range(len(indices)), total=len(indices)):
        index = indices[i]
        ori_prompt = origin_prompts[i]
        counter_prompt = counter_prompts[i]
        original_image_filename = assign_exist_image_file_extension(str(index), origin_image_path)
        counter_image_filename = assign_exist_image_file_extension(str(index), counter_image_path)
        path_to_original_image_file = osp.join(origin_image_path, original_image_filename)
        path_to_counter_image_file = osp.join(counter_image_path, counter_image_filename)
        path_to_coco_image_file = get_path_to_coco_image(index)
        
        scores_org_pr_org_img.append(model.compute_prompt_image_matching_score(ori_prompt, path_to_original_image_file))
        scores_org_pr_counter_img.append(model.compute_prompt_image_matching_score(ori_prompt, path_to_counter_image_file))
        scores_counter_pr_org_img.append(model.compute_prompt_image_matching_score(counter_prompt, path_to_original_image_file))
        scores_counter_pr_counter_img.append(model.compute_prompt_image_matching_score(counter_prompt, path_to_counter_image_file))
        
        scores_org_pr_coco_img.append(model.compute_prompt_image_matching_score(ori_prompt, path_to_coco_image_file))
        score_counter_pr_coco_img.append(model.compute_prompt_image_matching_score(counter_prompt, path_to_coco_image_file))
    
    data = {'scores_org_pr_org_img' : scores_org_pr_org_img, 
            'scores_org_pr_counter_img' : scores_org_pr_counter_img, 
            'scores_counter_pr_org_img' : scores_counter_pr_org_img,
            'scores_counter_pr_counter_img' : scores_counter_pr_counter_img,
            'scores_org_pr_coco_img' : scores_org_pr_coco_img,
            'score_counter_pr_coco_img' : score_counter_pr_coco_img,         
           }  
    '''
    df = pd.DataFrame.from_dict(data)
    df['diff_per_origin_prompt_on_ldm'] = df['scores_org_pr_org_img'] - df['scores_org_pr_counter_img']
    df['diff_per_counter_prompt_on_ldm'] = df['scores_counter_pr_counter_img'] - df['scores_counter_pr_org_img']
    df['diff_per_origin_image_on_ldm'] = df['scores_org_pr_org_img'] - df['scores_counter_pr_org_img']
    df['diff_per_counter_image_on_ldm'] = df['scores_counter_pr_counter_img'] - df['scores_org_pr_counter_img']
    
    df['diff_per_origin_prompt_on_coco'] = df['scores_org_pr_coco_img'] - df['scores_org_pr_counter_img']
    df['diff_per_counter_prompt_on_coco'] = df['scores_counter_pr_counter_img'] - df['score_counter_pr_coco_img']
    df['diff_per_coco_image'] = df['scores_org_pr_coco_img'] - df['score_counter_pr_coco_img']
    return df.to_dict('list')
    '''
    out_scores = compute_summary_score(data)
    return out_scores

def update_score_name_including_model(score_dict, model_name):
    new_score_dict = {}
    for k in score_dict.keys():
        new_score_dict[model_name + '_' + k] = score_dict[k]
    return new_score_dict

def compute_matching_scores_for_all_models(indices, origin_prompts, counter_prompts, origin_image_path, counter_image_path):
    all_scores = {}
    #BridgeTower Model
    bridgetower_scorer = BridgeTowerScorer()
    bridgetower_scores = compute_matching_score_using_model(bridgetower_scorer, indices, origin_prompts, counter_prompts, origin_image_path, counter_image_path)
    bridgetower_scores = update_score_name_including_model(bridgetower_scores, 'BT')
    all_scores.update(bridgetower_scores)
    del bridgetower_scorer
    
    #CLIP Model
    clip_scorer = ClipScorer()
    clip_scores = compute_matching_score_using_model(clip_scorer, indices, origin_prompts, counter_prompts, origin_image_path, counter_image_path)
    clip_scores = update_score_name_including_model(clip_scores, 'Clip')
    all_scores.update(clip_scores)
    del clip_scorer

    #VILT model
    vilt_scorer = VILTScorer()
    vilt_scores = compute_matching_score_using_model(vilt_scorer, indices, origin_prompts, counter_prompts, origin_image_path, counter_image_path)
    vilt_scores = update_score_name_including_model(vilt_scores, 'VILT')
    all_scores.update(vilt_scores)
    del vilt_scorer
    
    return all_scores
    
def main(args):
    #create output folder if not exists
    print('Checking input and output...')
    #create_output_folder(args.output_folder)

    if not osp.exists(args.prompt_file):
        raise ValueError('Prompt file does not exists!')
        return    
    
    #reading prompts
    indices = []
    origin_prompts = []
    counter_prompts = []
    #for line in tqdm(infile, total=get_num_lines(args.prompt_file)):
    with open(args.prompt_file) as infile:
        for line in infile:
            if '\n' == line[-1]:
                line = line[0:-1]
            index, origin, counter = line.split("\t")
            indices.append(index)
            origin_prompts.append(origin)
            counter_prompts.append(counter)
        
    #compute scores
    all_scores = compute_matching_scores_for_all_models(indices, origin_prompts, counter_prompts, args.original_images, args.counter_images)    
    #output scores
    data = {'indices' : indices,
            'origin_caption' : origin_prompts,
            'counter_caption' : counter_prompts           
           }
    data.update(all_scores)
    df = pd.DataFrame.from_dict(data)   
    df.to_csv(args.output_filename, sep='\t', encoding='utf-8', index=False)
    print('-------------DONE-----------')
            
if __name__ == '__main__':
    main(parse_args())
