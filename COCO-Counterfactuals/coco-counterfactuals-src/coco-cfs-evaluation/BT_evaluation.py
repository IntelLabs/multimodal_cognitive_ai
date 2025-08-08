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
from transformers import BridgeTowerProcessor, BridgeTowerForImageAndTextRetrieval,BridgeTowerForContrastiveLearning
from transformers import AutoConfig


model_name = "BridgeTower/bridgetower-large-itm-mlm-itc" 
processor = BridgeTowerProcessor.from_pretrained(model_name)
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

class EvaluatedDataset(Dataset):
    def __init__(self, filepath):
        self.table = pd.read_csv(filepath, header=0)
        self.image_path = self.table['image_path'].to_list()
        self.caption = self.table['caption'].to_list()
    
    def __len__(self,):
        return self.table.shape[0]
    
    def __getitem__(self, index):
        img_path = self.image_path[index]
        img = Image.open(img_path).convert("RGB") 
        caption = self.caption[index]
        return (img_path, img, caption)

def collate_fn(batch_list):
    image_paths, imgs, captions = list(zip(*batch_list))
    batch = processor(images=imgs, text=captions, padding=True, return_tensors='pt', truncation=True, max_length=514).to(device)
    batch['image_paths'] = image_paths
    batch['captions'] = captions
    return batch

def retrieve_all_image_embeddings_as_tensor(extracted_embeddings):
    image_embeds = []
    image_paths = []
    for image_id in extracted_embeddings.keys():
        image_embed = extracted_embeddings[image_id].squeeze(0)
        image_embeds.append(image_embed)
        image_paths.append(image_id)
    vectors = np.stack(image_embeds, axis=0)
    vectors = torch.from_numpy(vectors)
    return vectors, image_paths

def retrieve_all_text_embeddings_as_tensor(extracted_embeddings):
    text_embeds = []
    captions = []
    for caption in extracted_embeddings.keys():
        text_embed = extracted_embeddings[caption].squeeze(0)
        text_embeds.append(text_embed)
        captions.append(caption)
    vectors = np.stack(text_embeds, axis=0)
    vectors = torch.from_numpy(vectors)
    return vectors, captions

def compute_similarity_scores(source_embedding, all_target_embeddings, logit_scale):
    a = torch.from_numpy(source_embedding).to(device)
    b = all_target_embeddings.t().to(device)
    scores = torch.matmul(a, b )
    scores = scores * logit_scale.exp().to(device)
    scores = scores.detach().cpu()
    values, indices = scores.sort(stable=True, descending=True)
    return values, indices


def compute_topk_text_image_retrieval(text_extracted_embeddings, image_extracted_embeddings, topk_list, logit_scale, dataset):
    all_image_embeddings, image_paths = retrieve_all_image_embeddings_as_tensor(image_extracted_embeddings)
    recall_res = defaultdict(list)
    out = {}
    for i, caption in tqdm(enumerate(text_extracted_embeddings.keys()), total=len(text_extracted_embeddings), desc='Computing T2I'):
        text_embedding = text_extracted_embeddings[caption].squeeze(0)
        values, indices = compute_similarity_scores(text_embedding, all_image_embeddings, logit_scale)
        for top_k in topk_list:
            candidate_indices = indices[:top_k]
            candidate_images = [image_paths[j] for j in candidate_indices]
            hit = False
            for k in candidate_images:
                if dataset.table[ (dataset.table['caption'] == caption) &
                                (dataset.table['image_path'] == k)].shape[0] >=1 :
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

def compute_topk_image_text_retrieval(text_extracted_embeddings, image_extracted_embeddings, topk_list, logit_scale, dataset):
    all_text_embeddings, captions = retrieve_all_text_embeddings_as_tensor(text_extracted_embeddings)
    recall_res = defaultdict(list)
    out = {}
    for i, image_id in tqdm(enumerate(image_extracted_embeddings.keys()), total=len(image_extracted_embeddings), desc='Computing I2T'):
        image_embedding = image_extracted_embeddings[image_id].squeeze(0)
        values, indices = compute_similarity_scores(image_embedding, all_text_embeddings, logit_scale)
        for top_k in topk_list:
            candidate_indices = indices[:top_k]
            candidate_captions = [captions[j] for j in candidate_indices]
            hit = False
            for k in candidate_captions:
                if dataset.table[ (dataset.table['caption'] == k) &
                                (dataset.table['image_path'] == image_id)].shape[0] >=1 :
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

def evaluate_COCO(path_to_model, dataset_path):
    config = AutoConfig.from_pretrained(path_to_model)
    setattr(config, 'contrastive_hidden_size', 512)
    setattr(config, "logit_scale_init_value", 2.6592)
    model = BridgeTowerForContrastiveLearning.from_pretrained(
        path_to_model,
        config=config,
    ).to(device)
    # model = BridgeTowerForContrastiveLearning.from_pretrained(path_to_model).to("cuda")
    dataset = EvaluatedDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False, collate_fn=collate_fn)
    image_extracted_embeddings = {}
    text_extracted_embeddings = {}
    try:
        for _, batch in enumerate(tqdm(dataloader)):    

            captions = copy(batch['captions'])
            image_paths = copy(batch['image_paths'])
            del batch['captions']
            del batch['image_paths']
            with torch.no_grad():
                outputs = model(**batch, return_dict=True)
            batch_size = len(captions)
            # print(len(captions))
            # print(len(image_paths))
            for bidx in range(batch_size):
                image_path = image_paths[bidx]
                text = captions[bidx]
                embeds_txt   = outputs.text_embeds[bidx, :].view(1, -1).detach().cpu().numpy()
                embeds_img   = outputs.image_embeds[bidx, :].view(1, -1).detach().cpu().numpy()
                #logit_per_image = outputs['logits_per_image'].cpu()
                #logit_per_text = outputs['logits_per_text'].cpu()

                image_extracted_embeddings[image_path] = embeds_img
                text_extracted_embeddings[text] = embeds_txt
            del outputs
    except:
        print('FAILED at computing clip output', captions)
    print('DONE')    
    print('image_extracted_embeddings', len(image_extracted_embeddings))
    print('text_extracted_embeddings', len(text_extracted_embeddings))

    top_ks = [1, 5, 10]
    print('Computing text 2 image retrieval')
    recall_image_retrieval = compute_topk_text_image_retrieval(text_extracted_embeddings, image_extracted_embeddings, top_ks, model.logit_scale, dataset)
    print('Computing image 2 text retrieval')
    recall_text_retrieval = compute_topk_image_text_retrieval(text_extracted_embeddings, image_extracted_embeddings, top_ks, model.logit_scale, dataset)
    return recall_image_retrieval, recall_text_retrieval

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Evaluate  BridgeTower',
        usage='BT_evaluation.py [<args>] [-h | --help]'
    )

    parser.add_argument('-d', '--ds', default="data/ms-coco.csv", type=str, help='Path to dataset csv file')
    parser.add_argument('-c', '--checkpoint_folder', default="baseline", type=str, help='Folder that includes all checkpoints')
    parser.add_argument('-o', '--output_dir', required=True, type=str, help='Output folder')
    return parser.parse_args(args)

def main(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    dataset_path = args.ds
    checkpoint_folder = args.checkpoint_folder
    
    all_checkpoints = [checkpoint_folder]
    all_results = []
    if checkpoint_folder == 'baseline':
        name = 'BT-baseline'
        res = {}
        res['model'] = name
        recall_image_retrieval, recall_text_retrieval = evaluate_COCO(model_name, dataset_path)
        res.update(recall_image_retrieval)
        res.update(recall_text_retrieval)
        all_results.append(res)
    else:
        name = 'finetuned_BT'
        for chkpt in tqdm(all_checkpoints, desc='processing each checkpoint'):
            print('processing checkpoints:', chkpt)
            res = {}
            res['model'] = chkpt
            recall_image_retrieval, recall_text_retrieval = evaluate_COCO(chkpt, dataset_path)
            res.update(recall_image_retrieval)
            res.update(recall_text_retrieval)
            all_results.append(res)
    df_all = pd.DataFrame(all_results)
    # output = osp.join('evaluate_'+ dataset_path.split('/')[-1] + '_' + model_name.split('/')[-1] + '.csv')
    output = osp.join(args.output_dir, 'evaluate_' + name + '.csv')
    print('writing results to', output)
    df_all.to_csv(output, index=False, header=True, encoding='utf-8')
        
if __name__ == '__main__':
    main(parse_args())
