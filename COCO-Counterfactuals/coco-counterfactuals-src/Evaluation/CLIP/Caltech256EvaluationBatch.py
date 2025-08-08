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
from sklearn.metrics import precision_score, accuracy_score

from copy import copy
from PIL import Image
import pickle
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
from pathlib import Path
from transformers import CLIPModel, CLIPProcessor


processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# For Caltech256
caltech256_data = torchvision.datasets.Caltech256('/export/share/projects/mcai/COCO-Counterfactuals/datasets/caltech256/', download=True)
caltech256_labels = copy(caltech256_data.categories)

sents_256 = []
specials = {'billiards':'billiard',
            'binoculars' : 'binocular',
            'bonsai-101' : 'bonsai',
            'brain-101' : 'brain',
            'buddha-101' : 'buddha',
            'chandelier-101' : 'chandelier',
            'chopsticks' : 'chopstick',
            'crab-101' : 'crab',
            'dolphin-101' : 'dolphin',
            'electric-guitar-101' : 'electric-guitar',
            'elephant-101' : 'elephant',
            'ewer-101' : 'ewer',
            'eyeglasses' : 'eyeglass',
            'fireworks' : 'firework',
            'grand-piano-101' : 'grand-piano',
            'hawksbill-101' : 'hawksbill',
            'helicopter-101' : 'helicopter',
            'ibis-101' : 'ibis',
            'kangaroo-101' : 'kangaroo',
            'ketch-101' : 'ketch',
            'laptop-101' : 'laptop',
            'leopards-101' : 'leopard',
            'llama-101' : 'llama',
            'menorah-101' : 'menorah',
            'motorbikes-101' : 'motorbike',
            'mussels' : 'mussel',
            'revolver-101' : 'revolver',
            'scorpion-101' : 'scorpion',
            'socks' : 'sock',
            'starfish-101' : 'starfish',
            'sunflower-101' : 'sunflower',
            'triceratops' : 'triceratop',
            'trilobite-101' : 'trilobite',
            'umbrella-101' : 'umbrella',
            'watch-101' : 'watch',
            'airplanes-101' : 'airplane',
            'car-side-101' : 'car-side',
            'faces-easy-101' : 'face',
           }
for i in range(len(caltech256_labels)):
    tmp = caltech256_labels[i].lower()
    #remove xxx from xxx.label
    tmp = tmp.split('.')[-1]
    #remove plural if needed
    if tmp in specials.keys():
        tmp = specials[tmp]
    #replace '-' with space
    if tmp.lower() != 't-shirt':
        tmp = tmp.replace('-', ' ')
    
    sents_256.append(f'a photo of a {tmp}')
    
def collate_fn(batch_list):
    images, labels = list(zip(*batch_list))
    batch = processor(images=images, text=sents_256, padding=True, return_tensors='pt', truncation=True).to(device)
    batch['labels'] = labels
    return batch


def evaluate_caltech(path_to_model):
    model = CLIPModel.from_pretrained(path_to_model).to("cuda")    
    dataloader = DataLoader(caltech256_data, batch_size=64, shuffle=False, drop_last=False, collate_fn=collate_fn)
    all_labels = []
    all_predictions = []
    try:
        for _, batch in enumerate(tqdm(dataloader)):    
            labels = copy(batch['labels'])
            labels = list(labels)
            del batch['labels']
            with torch.no_grad():
                outputs = model(**batch, output_hidden_states=True)
            batch_size = len(labels)
            logits_per_image = outputs.logits_per_image.detach().cpu()
            predicted_classes = logits_per_image.argmax(dim=1)
            predicted_classes = list(predicted_classes.numpy())
            all_predictions += predicted_classes
            all_labels += labels
            
    except:
        print('FAILED at computing clip output')
    corrects = [ 1 if all_labels[i] == all_predictions[i] else 0 for i in range(len(all_predictions))]
    accuracy = accuracy_score(all_labels, all_predictions)
    print('DONE')
    print('Accuracy', str(accuracy))
    return accuracy, all_labels, all_predictions


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Evaluate Caltech256 for every checkpoint enclosed in a folder',
        usage='Caltech256EvaluationBatch.py [<args>] [-h | --help]'
    )

    parser.add_argument('-c', '--checkpoint_folder', default=".", type=str, help='Folder that includes all checkpoints')
    parser.add_argument('-l', '--writeback_truth_labels', action='store_true', help='Write back truth labels of dataset or not')
    parser.add_argument('-r', '--recursive', action='store_true', help='Whether evaluating all checkpoints in the checkpoint folder or not')
    parser.add_argument('-o', '--output_dir', type=str, default='.', help='Output dir path')
    return parser.parse_args(args)

def write_predictions_to_file_as_npy(predictions, checkpoint, filename):
    with open(osp.join(checkpoint, filename), 'wb') as f:
        np.save(f, np.array(predictions))
        
def write_groundtruth_labels_to_file_as_npy(labels, checkpoint, filename):
    with open(osp.join(checkpoint, filename), 'wb') as f:
        np.save(f, np.array(labels))
        
def main(args):
    checkpoint_folder = args.checkpoint_folder
    if args.output_dir != '.':
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    name = 'caltech256'
    
    if checkpoint_folder != '.':
        if args.recursive:
            all_checkpoints = glob(checkpoint_folder + "/**/", recursive = True)
            all_checkpoints = [i for i in all_checkpoints if 'checkpoint-' in i]
        else:
            all_checkpoints = [checkpoint_folder]
    else:
        all_checkpoints = ['openai/clip-vit-base-patch32']
    all_results = []
    for chkpt in tqdm(all_checkpoints, desc='processing each checkpoint'):
        print()
        print('processing checkpoints:', chkpt)
        res = {}
        res['model'] = chkpt
        accuracy, all_labels, all_predictions = evaluate_caltech(chkpt)
        print('writing predictions to:', args.output_dir)
        write_predictions_to_file_as_npy(all_predictions, args.output_dir, 'predictions_' + name + '.npy')
        res.update({'accuracy' : accuracy})
        all_results.append(res)
    df_all = pd.DataFrame(all_results)
    output = osp.join(args.output_dir, 'evaluate_' + name + '.csv')
    print('writing results to:', output)
    df_all.to_csv(output, index=False, header=True, encoding='utf-8')
    if args.writeback_truth_labels == True:
        write_groundtruth_labels_to_file_as_npy(all_labels, args.output_dir, 'groundtruth_' + name + '.npy')
    print('FINISH')
        
if __name__ == '__main__':
    main(parse_args())
