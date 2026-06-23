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

sents_10 = []
sents_100 = []
# For Cifar10
cifar10_labels = {0 : "airplane", 1: "automobile", 2: "bird", 3 : "cat", 4: "deer", 5: "dog", 6 : "frog", 7 : "horse", 8 : "ship", 9 : "truck"}

for k, v in cifar10_labels.items():
    sents_10.append(f'a photo of a {v}')

# For Cifar100
with open('/export/share/projects/mcai/COCO-Counterfactuals/datasets/new_cifar100/cifar-100-python/meta', 'rb') as handle:
    cifar100_labels = handle.read()
    
cifar100_labels = pickle.loads(cifar100_labels)
cifar100_labels = cifar100_labels['fine_label_names']

for i in range(len(cifar100_labels)):
    tmp = cifar100_labels[i].replace('_', ' ')
    sents_100.append(f'a photo of a {tmp}')
    
def collate_fn_cifar10(batch_list):
    images, labels = list(zip(*batch_list))
    batch = processor(images=images, text=sents_10, padding=True, return_tensors='pt', truncation=True).to(device)
    batch['labels'] = labels
    return batch

def collate_fn_cifar100(batch_list):
    images, labels = list(zip(*batch_list))
    batch = processor(images=images, text=sents_100, padding=True, return_tensors='pt', truncation=True).to(device)
    batch['labels'] = labels
    return batch


def evaluate_Cifar(path_to_model, dataloader):
    model = CLIPModel.from_pretrained(path_to_model).to("cuda")
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

def evaluate_Cifar100(path_to_model):
    cifar100_data = torchvision.datasets.CIFAR100('/export/share/projects/mcai/COCO-Counterfactuals/datasets/new_cifar100/', train=False, download=True)
    dataloader = DataLoader(cifar100_data, batch_size=64, shuffle=False, drop_last=False, collate_fn=collate_fn_cifar100)
    accuracy, all_labels, all_predictions = evaluate_Cifar(path_to_model, dataloader)
    return accuracy, all_labels, all_predictions

def evaluate_Cifar10(path_to_model):
    cifar10_data = torchvision.datasets.CIFAR10('/export/share/projects/mcai/COCO-Counterfactuals/datasets/new_cifar10/', train=False, download=True)
    dataloader = DataLoader(cifar10_data, batch_size=64, shuffle=False, drop_last=False, collate_fn=collate_fn_cifar10)
    accuracy, all_labels, all_predictions = evaluate_Cifar(path_to_model, dataloader)
    return accuracy, all_labels, all_predictions

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Evaluate CIFAR10 and CIFAR100 for every checkpoint enclosed in a folder',
        usage='CifarEvaluationBatch.py [<args>] [-h | --help]'
    )

    parser.add_argument('-c', '--checkpoint_folder', default='.', type=str, help='Folder that includes all checkpoints')
    parser.add_argument('-l', '--writeback_truth_labels', action='store_true', help='Write back truth labels of dataset or not')
    parser.add_argument('-i', '--is_10', action='store_true', help='Cifar10 or not. if not then it is cifar100')
    parser.add_argument('-r', '--recursive', action='store_true', help='Whether evaluating all checkpoints in the checkpoint folder or not')
    parser.add_argument('-o', '--output_dir', type=str, default='.', help='Output dir path')
    return parser.parse_args(args)

# 'predictions_cifar10.npy'
def write_predictions_to_file_as_npy(predictions, checkpoint, filename):
    with open(osp.join(checkpoint, filename), 'wb') as f:
        np.save(f, np.array(predictions))
        
# 'groundtruth_cifar10.npy'
def write_groundtruth_labels_to_file_as_npy(labels, checkpoint, filename):
    with open(osp.join(checkpoint, filename), 'wb') as f:
        np.save(f, np.array(labels))
        
def main(args):
    checkpoint_folder = args.checkpoint_folder
    if args.output_dir != '.':
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    if args.is_10==True:
        name='cifar10'
    else:
        name='cifar100'

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
        print('processing checkpoints:', chkpt)
        res = {}
        res['model'] = chkpt
        if args.is_10==True:
            print('Evaluating on CIFAR10')
            accuracy, all_labels, all_predictions = evaluate_Cifar10(chkpt)
        else:
            print('Evaluating on CIFAR100')
            accuracy, all_labels, all_predictions = evaluate_Cifar100(chkpt)
        print('writing predictions to:', args.output_dir)
        
        write_predictions_to_file_as_npy(all_predictions, args.output_dir, 'predictions_' + name + '.npy')
        res.update({'accuracy' : accuracy})
        all_results.append(res)
    df_all = pd.DataFrame(all_results)
    output = osp.join(args.output_dir, 'evaluate_' + name + '.csv')
    print('writing results to:', output)
    df_all.to_csv(output, index=False, header=True, encoding='utf-8')
    if args.writeback_truth_labels:
        write_groundtruth_labels_to_file_as_npy(all_labels, args.output_dir, 'groundtruth_' + name + '.npy')
    print('FINISH')
        
if __name__ == '__main__':
    main(parse_args())
