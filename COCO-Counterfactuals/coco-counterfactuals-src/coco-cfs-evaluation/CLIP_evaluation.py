import os
import os.path as osp
from glob import glob
import sys
import numpy as np
import faiss
from collections import defaultdict
import io
import pyarrow as pa
import pandas as pd

from copy import copy
from PIL import Image
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
from pathlib import Path
from transformers import CLIPModel, CLIPProcessor


model_name = "openai/clip-vit-base-patch32"
# model_name = "openai/clip-vit-large-patch14"
processor = CLIPProcessor.from_pretrained(model_name)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# coco_filepath = 'data/ms-coco.csv'
# # coco_filepath = 'data/human-evaluated-coco-cfs.csv'
# # coco_filepath = 'data/coco-cfs.csv'
# print(coco_filepath)

# input_table = pd.read_csv(coco_filepath, header=0)

# class CocoDataset(Dataset):
#     def __init__(self, filepath):
#         self.table = pd.read_csv(filepath, header=0)
    
#     def __len__(self,):
#         return self.table.shape[0]
    
#     def __getitem__(self, index):
#         img_path = self.table['image_path'][index]
#         caption_id = self.table['caption_id'][index]
#         img = Image.open(img_path).convert("RGB")
#         caption = self.table['caption'][index]       
#         return (caption_id, img_path, caption, img)

class ImageCocoDataset(Dataset):
    def __init__(self, filepath):
        self.table = pd.read_csv(filepath, header=0)
        self.image_path = list(set(self.table['image_path'].to_list()))
    
    def __len__(self,):
        return len(self.image_path)
    
    def __getitem__(self, index):
        img_path = self.image_path[index]
        img = Image.open(img_path).convert("RGB")       
        return (img_path, img)
    
def collate_fn_image(batch_list):
    image_paths, images = list(zip(*batch_list))
    # print(image_ids)
    batch = processor(images=images, text=['a picture of something'], padding=True, return_tensors='pt', truncation=True).to(device)
    batch['image_paths'] = image_paths
    return batch

def compute_all_images_embeddings(path_to_model, coco_filepath):
    model = CLIPModel.from_pretrained(path_to_model).to("cuda")
    dataset = ImageCocoDataset(coco_filepath)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False, collate_fn=collate_fn_image)
    extracted_embeddings = {}
    try:
        for _, batch in enumerate(tqdm(dataloader)):    
            image_paths = copy(batch['image_paths'])
            del batch['image_paths']
            with torch.no_grad():
                outputs = model(**batch, output_hidden_states=True)
            batch_size = len(image_paths)
            for bidx in range(batch_size):
                image_path = image_paths[bidx]
                embeds_img   = outputs['image_embeds'][bidx, :].view(1, -1).cpu()

                extracted_embeddings[image_path] = {
                    'image_embed': embeds_img,
                }
    except:
        print('FAILED at computing clip output', captions)
    print('DONE')    
    print('image_extracted_embeddings', len(extracted_embeddings))
    return extracted_embeddings

class TextCocoDataset(Dataset):
    def __init__(self, filepath):
        self.table = pd.read_csv(filepath, header=0)
        self.captions = list(set(self.table['caption'].to_list()))
    
    def __len__(self,):
        return len(self.captions)
    
    def __getitem__(self, index):
        caption = self.captions[index]       
        return (caption, index)
    
def collate_fn_caption(batch_list):
    captions, indices = list(zip(*batch_list))
    img = Image.open(osp.join('/export/share/datasets/COCO2017/val2017', '000000001000.jpg')).convert("RGB")
    batch = processor(images=img, text=captions, padding=True, return_tensors='pt', truncation=True).to(device)
    batch['captions'] = captions
    return batch

def compute_all_caption_embeddings(path_to_model, coco_filepath):
    model = CLIPModel.from_pretrained(path_to_model).to("cuda")
    dataset = TextCocoDataset(coco_filepath)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False, collate_fn=collate_fn_caption)
    extracted_embeddings = {}
    try:
        for _, batch in enumerate(tqdm(dataloader)):    
            captions = copy(batch['captions'])
            del batch['captions']
            with torch.no_grad():
                outputs = model(**batch, output_hidden_states=True)
            batch_size = len(captions)
            for bidx in range(batch_size):
                caption = captions[bidx]
                embeds_txt   = outputs['text_embeds'][bidx, :].view(1, -1).cpu()
    
                extracted_embeddings[caption] = {
                    'text_embed': embeds_txt,
                }
    except:
        print('FAILED at computing clip output', captions)
    print('DONE')    
    print('caption_extracted_embeddings', len(extracted_embeddings))
    return extracted_embeddings
    
def compute_topk_image_text_retrieval(image_extracted_embeddings, text_extracted_embeddings, topk_list, logit_scale, input_table):
    recall_res = defaultdict(list)
    all_text_embeddings = retrieve_all_text_embeddings_as_tensor(text_extracted_embeddings)
    out = {}
    for i, image_path in tqdm(enumerate(image_extracted_embeddings.keys()), total=len(image_extracted_embeddings.keys())):
        image_embedding = image_extracted_embeddings[image_path]['image_embed'].squeeze(0)
        values, indices = compute_similarity_scores(image_embedding, all_text_embeddings, logit_scale)
        for top_k in topk_list:
            candidate_indices = indices[:top_k]
            candidate_captions = [ix(text_extracted_embeddings, i) for i in candidate_indices]
            hit = False
            for i in candidate_captions:
                if input_table[ (input_table['caption'] == i) &
                                (input_table['image_path'] == image_path)].shape[0] >=1 :
                    hit= True
                    break
            if hit == True:
                recall_res[top_k].append(1)
            else:
                recall_res[top_k].append(0)
    for top_k in topk_list:
        out['I2Trecall@' + str(top_k)] = 100*np.mean(recall_res[top_k])
        print(f'Recall@{top_k}: {100*np.mean(recall_res[top_k])}')
    return out    
    
def compute_topk_text_image_retrieval(image_extracted_embeddings, text_extracted_embeddings, topk_list, logit_scale, input_table):
    all_image_embeddings = retrieve_all_image_embeddings_as_tensor(image_extracted_embeddings)
    recall_res = defaultdict(list)
    # recall_res_val2017 = defaultdict(list)
    out = {}
    for i, caption in tqdm(enumerate(text_extracted_embeddings.keys()),total=len(text_extracted_embeddings.keys())):
        text_embedding = text_extracted_embeddings[caption]['text_embed'].squeeze(0)
        values, indices = compute_similarity_scores(text_embedding, all_image_embeddings, logit_scale)
        for top_k in topk_list:
            candidate_indices = indices[:top_k]
            candidate_image_paths = [ix(image_extracted_embeddings, i) for i in candidate_indices]
            
            hit = False
            for i in candidate_image_paths:
                if input_table[ (input_table['caption'] == caption) &
                                (input_table['image_path'] == i)].shape[0] >=1 :
                    hit= True
                    break
            if hit == True:
                recall_res[top_k].append(1)
            else:
                recall_res[top_k].append(0)
    for top_k in topk_list:
        out['T2I.recall@' + str(top_k)] = 100*np.mean(recall_res[top_k])
        print(f'Recall@{top_k}: {100*np.mean(recall_res[top_k])}')
    return out

#################
def retrieve_all_image_embeddings_as_tensor(extracted_embeddings):
    image_embeds = []
    for caption_id in extracted_embeddings.keys():
        image_embed = extracted_embeddings[caption_id]['image_embed'].squeeze(0)
        image_embeds.append(image_embed)
    vectors = np.stack(image_embeds, axis=0)
    vectors = torch.from_numpy(vectors)
    return vectors

def retrieve_all_text_embeddings_as_tensor(extracted_embeddings):
    text_embeds = []
    for caption_id in extracted_embeddings.keys():
        text_embed = extracted_embeddings[caption_id]['text_embed'].squeeze(0)
        text_embeds.append(text_embed)
    vectors = np.stack(text_embeds, axis=0)
    vectors = torch.from_numpy(vectors)
    return vectors

def compute_similarity_scores(source_embedding, all_target_embeddings, logit_scale):
    scores = torch.matmul(source_embedding.to(device), all_target_embeddings.t().to(device))
    scores = scores * logit_scale.exp().to(device)
    scores = scores.detach().cpu()
    values, indices = scores.sort(stable=True, descending=True)
    return values, indices

def ix(dct, n): 
    try:
        return list(dct)[n] 
    except IndexError:
        print('not enough keys')



def evaluate_coco(path_to_model, coco_filepath):
    model = CLIPModel.from_pretrained(path_to_model).to("cuda")
    print('Compute text embeddings')
    
    text_extracted_embeddings = compute_all_caption_embeddings(path_to_model, coco_filepath)
    print('Compute image embeddings')
    image_extracted_embeddings = compute_all_images_embeddings(path_to_model, coco_filepath)
    
    #load ground truth
    input_table = pd.read_csv(coco_filepath, header=0)
    top_ks = [1, 5, 10]
    print('Computing text 2 image retrieval')
    recall_image_retrieval = compute_topk_text_image_retrieval(image_extracted_embeddings, text_extracted_embeddings, top_ks, model.logit_scale, input_table)
    print('Computing image 2 text retrieval')
    recall_text_retrieval = compute_topk_image_text_retrieval(image_extracted_embeddings, text_extracted_embeddings, top_ks, model.logit_scale, input_table)
    return recall_image_retrieval, recall_text_retrieval

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Evaluate CLIP',
        usage='CLIP_evaluation.py [<args>] [-h | --help]'
    )

    parser.add_argument('-c', '--checkpoint_folder', default="baseline", type=str, help='Folder that includes all checkpoints')
    parser.add_argument('-d', '--test_dataset_path', required=True, type=str, help='Path to test dataset')
    parser.add_argument('-o', '--output_dir', required=True, type=str, help='Output folder')
    return parser.parse_args(args)

def main(args):
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    coco_filepath = args.test_dataset_path
    print(f"Test dataset path: {coco_filepath}")
    
    checkpoint_folder = args.checkpoint_folder
    # all_checkpoints = glob(checkpoint_folder + "/**/", recursive = True)
    # all_checkpoints = [i for i in all_checkpoints if 'checkpoint-' in i]
    all_checkpoints = [checkpoint_folder]
    all_results = []
    if checkpoint_folder == 'baseline':
        name = 'clip-baseline'
        res = {}
        res['model'] = name
        recall_image_retrieval, recall_text_retrieval = evaluate_coco(model_name, coco_filepath)
        res.update(recall_image_retrieval)
        res.update(recall_text_retrieval)
        all_results.append(res)
        # return
    else:
        name = 'finetuned_clip' 
        for chkpt in tqdm(all_checkpoints, desc='processing each checkpoint'):
            print('processing checkpoints:', chkpt)
            res = {}
            res['model'] = chkpt
            recall_image_retrieval, recall_text_retrieval = evaluate_coco(chkpt, coco_filepath)
            res.update(recall_image_retrieval)
            res.update(recall_text_retrieval)
            all_results.append(res)
    df_all = pd.DataFrame(all_results)
    output = osp.join(args.output_dir, 'evaluate_' + name + '.csv')
    print('writing results to', output)
    df_all.to_csv(output, index=False, header=True, encoding='utf-8')
        
if __name__ == '__main__':
    main(parse_args())
