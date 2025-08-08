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
from transformers import BridgeTowerForContrastiveLearning, BridgeTowerProcessor


processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm")
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    
flick30k_filepath = osp.join('.', 'f30k_caption_karpathy_test.arrow')

class FlickrDataset(Dataset):
    def __init__(self, filepath):
        
        with pa.memory_map(filepath, 'r') as source:
            self.table = pa.ipc.open_file(source).read_all()
    
    def __len__(self,):
        return len(self.table)
    
    def __getitem__(self, index):
        image_bytes = io.BytesIO(self.table['image'][index].as_py())
        image_bytes.seek(0)
        img = Image.open(image_bytes).convert("RGB")
        caption = self.table['caption'][index][0].as_py()
        image_id = self.table['image_id'][index].as_py()       
        return (image_id, caption, img)

def collate_fn(batch_list):
    image_ids, captions, images = list(zip(*batch_list))
    batch = processor(images=images, text=captions, padding=True, return_tensors='pt', truncation=True).to(device)
    batch['image_ids'] = image_ids
    batch['captions'] = captions
    return batch

def retrieve_all_image_embeddings_as_tensor(extracted_embeddings):
    image_embeds = []
    for image_id in extracted_embeddings.keys():
        image_embed = extracted_embeddings[image_id]['image_embed'].squeeze(0)
        image_embeds.append(image_embed)
    vectors = np.stack(image_embeds, axis=0)
    vectors = torch.from_numpy(vectors)
    return vectors

def retrieve_all_text_embeddings_as_tensor(extracted_embeddings):
    text_embeds = []
    for image_id in extracted_embeddings.keys():
        text_embed = extracted_embeddings[image_id]['text_embed'].squeeze(0)
        text_embeds.append(text_embed)
    vectors = np.stack(text_embeds, axis=0)
    vectors = torch.from_numpy(vectors)
    return vectors

def compute_similarity_scores(source_embedding, all_target_embeddings, logit_scale):
    a = torch.from_numpy(source_embedding).to(device)
    b = all_target_embeddings.t().to(device)
    scores = torch.matmul(a, b )
    scores = scores * logit_scale.exp().to(device)
    scores = scores.detach().cpu()
    values, indices = scores.sort(stable=True, descending=True)
    return values, indices


def compute_topk_text_image_retrieval(extracted_embeddings, topk_list, logit_scale):
    all_image_embeddings = retrieve_all_image_embeddings_as_tensor(extracted_embeddings)
    recall_res = defaultdict(list)
    out = {}
    for i, image_id in enumerate(extracted_embeddings.keys()):
        text_embedding = extracted_embeddings[image_id]['text_embed'].squeeze(0)
        values, indices = compute_similarity_scores(text_embedding, all_image_embeddings, logit_scale)
        for top_k in topk_list:
            if i in indices[:top_k]:
                recall_res[top_k].append(1)
            else:
                recall_res[top_k].append(0)
    for top_k in topk_list:
        out['T2I.recall@' + str(top_k)] = 100*np.mean(recall_res[top_k])
        print(f'Recall@{top_k}: {100*np.mean(recall_res[top_k])}')
    return out

def compute_topk_image_text_retrieval(extracted_embeddings, topk_list, logit_scale):
    all_text_embeddings = retrieve_all_text_embeddings_as_tensor(extracted_embeddings)
    recall_res = defaultdict(list)
    out = {}
    for i, image_id in enumerate(extracted_embeddings.keys()):
        image_embedding = extracted_embeddings[image_id]['image_embed'].squeeze(0)
        values, indices = compute_similarity_scores(image_embedding, all_text_embeddings, logit_scale)
        for top_k in topk_list:
            if i in indices[:top_k]:
                recall_res[top_k].append(1)
            else:
                recall_res[top_k].append(0)
    for top_k in topk_list:
        out['I2Trecall@' + str(top_k)] = 100*np.mean(recall_res[top_k])
        print(f'Recall@{top_k}: {100*np.mean(recall_res[top_k])}')
    return out

def evaluate_Flick30K(path_to_model):
    model = BridgeTowerForContrastiveLearning.from_pretrained(path_to_model).to("cuda")
    dataset = FlickrDataset(flick30k_filepath)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False, collate_fn=collate_fn)
    extracted_embeddings = {}
    try:
        for _, batch in enumerate(tqdm(dataloader)):    

            captions = copy(batch['captions'])
            image_ids = copy(batch['image_ids'])
            del batch['captions']
            del batch['image_ids']
            with torch.no_grad():
                outputs = model(**batch, output_hidden_states=True)
            batch_size = len(captions)
            for bidx in range(batch_size):
                image_id = image_ids[bidx]
                text = captions[bidx]
                embeds_txt   = outputs.text_embeds[bidx, :].view(1, -1).detach().cpu().numpy()
                embeds_img   = outputs.image_embeds[bidx, :].view(1, -1).detach().cpu().numpy()
                
                #logit_per_image = outputs['logits_per_image'].cpu()
                #logit_per_text = outputs['logits_per_text'].cpu()

                extracted_embeddings[image_id] = {
                    'caption': text,
                    'text_embed': embeds_txt,
                    'image_embed': embeds_img,
                }
    except:
        print('FAILED at computing clip output', captions)
    print('DONE')    
    print(len(extracted_embeddings))

    top_ks = [1, 5, 10]
    print('Computing text 2 image retrieval')
    recall_image_retrieval = compute_topk_text_image_retrieval(extracted_embeddings, top_ks, model.logit_scale)
    print('Computing image 2 text retrieval')
    recall_text_retrieval = compute_topk_image_text_retrieval(extracted_embeddings, top_ks, model.logit_scale)
    return recall_image_retrieval, recall_text_retrieval

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Evaluate Flick30K for every checkpoint enclosed in a folder',
        usage='FlickrEvaluationBatch.py [<args>] [-h | --help]'
    )

    parser.add_argument('-c', '--checkpoint_folder', default=".", type=str, help='Folder that includes all checkpoints')
    parser.add_argument('-r', '--recursive', action='store_true', help='Whether evaluating all checkpoints in the checkpoint folder or not')
    parser.add_argument('-o', '--output_dir', type=str, default='.', help='Output dir path')
    return parser.parse_args(args)

def main(args):
    if args.output_dir != '.':
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    checkpoint_folder = args.checkpoint_folder
    if checkpoint_folder != '.':
        if args.recursive:
            all_checkpoints = glob(checkpoint_folder + "/**/", recursive = True)
            all_checkpoints = [i for i in all_checkpoints if 'checkpoint-' in i]
        else:
            all_checkpoints = [checkpoint_folder]
    else:
        all_checkpoints = ['BridgeTower/bridgetower-large-itm-mlm-itc']

    all_results = []
    for chkpt in tqdm(all_checkpoints, desc='processing each checkpoint'):
        print('processing checkpoints:', chkpt)
        res = {}
        res['model'] = chkpt
        recall_image_retrieval, recall_text_retrieval = evaluate_Flick30K(chkpt)
        res.update(recall_image_retrieval)
        res.update(recall_text_retrieval)
        all_results.append(res)
    df_all = pd.DataFrame(all_results)
    output = osp.join(args.output_dir, 'evaluate_Flickr30K.csv')
    print('writing results to', output)
    df_all.to_csv(output, index=False, header=True, encoding='utf-8')
        
if __name__ == '__main__':
    main(parse_args())
