## Imports
import numpy as np
import torch
from PIL import Image
import os.path
import argparse
from pathlib import Path
import cv2
import heapq
from torch.nn import functional as F
from torch.utils.data import DataLoader
import tqdm
import einops
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
from utils.factory import create_model_and_transforms, get_tokenizer
from utils.visualization import image_grid, visualization_preprocess
from prs_hook import hook_prs_logger
from matplotlib import pyplot as plt
import pickle as pkl


print(torch.cuda.is_available())
## Hyperparameters

device = 'cuda'
joint_name = 'ViT-L-14_openai'
model_name = joint_name.split('_')[0]
pretrained = "_".join(joint_name.split('_')[1:])
batch_size = 1
imagenet_path = '/mnt/beegfs/mixed-tier/work/amadasu/Experiments/data/'

print(f'Model name: {model_name}')
print(f'Dataset used: {pretrained}')

def get_ViT_B_16_laion2b_s34b_b88k_property_head_layer_labels():
    properties = {'animals': ['L9.H2', 'L10.H4'],
                'locations': ['L9.H8', 'L10.H3', 'L11.H0', 'L11.H6'],
                'art': ['L9.H10', 'L9.H11'],
                'subject': ['L8.H1', 'L8.H4', 'L8.H10', 'L8.H11'],
                'nature': ['L8.H7', 'L9.H3', 'L9.H7', 'L11.H2'],

                }

    return properties

def get_ViT_B_16_openai_property_head_layer_labels():
    properties = {'animals': ['L9.H10', 'L10.H5', 'L11.H1'],
                'locations': ['L8.H1', 'L8.H7', 'L8.H11', 'L9.H3', 'L9.H7', 'L10.H2',
                            'L10.H3', 'L10.H4', 'L10.H7', 'L10.H10', 'L11.H6'],
                }

    return properties

def get_ViT_B_32_datacomp_m_s128m_b4k_property_head_layer_labels():
    properties = {'animals': ['L8.H2', 'L8.H3', 'L11.H1', 'L11.H6'],
                'colors': ['L9.H6', 'L11.H4', 'L11.H9'],
                }

    return properties

def get_ViT_B_32_openai_property_head_layer_labels():
    properties = {'photography': ['L8.H6', 'L8.H7', 'L9.H6', 'L10.H1'],
                'pattern': ['L8.H2', 'L8.H9'],
                'locations': ['L8.H4', 'L8.H8', 'L9.H11', 'L10.H3', 'L10.H7', 'L10.H10',
                            'L11.H6', 'L11.H9', 'L11.H10', 'L11.H11']}

    return properties

def get_ViT_L_14_laion2b_s32b_b82k_property_head_layer_labels():
    properties = {'colors': ['L21.H0', 'L21.H9', 'L22.H10', 'L22.H11', 'L22.H14', 'L23.H8'],
                'locations': ['L20.H0', 'L20.H1', 'L20.H2', 'L20.H3', 'L20.H8', 'L20.H9',
                            'L21.H1', 'L21.H3', 'L21.H11', 'L21.H13', 'L21.H14',
                            'L22.H2', 'L22.H12', 'L22.H13', 'L23.H6'],
                'environment': ['L20.H6', 'L21.H6', 'L22.H7', 'L23.H5'],
                'objects': ['L20.H10', 'L21.H7', 'L22.H3', 'L23.H7'],
                'photography': ['L20.H11', 'L20.H13', 'L23.H13']}

    return properties

def get_ViT_L_14_openai_property_head_layer_labels():
    properties = {'colors': ['L21.H4'],
                'locations': ['L20.H0', 'L20.H2', 'L20.H3', 'L20.H10',
                            'L21.H1', 'L21.H10', 'L21.H11', 'L21.H13', 'L21.H15',
                            'L22.H2', 'L22.H5', 'L22.H14', 'L22.H15', 'L23.H0', 'L23.H6'],
                'environment': ['L20.H5', 'L22.H6', 'L23.H1', 'L23.H14'],
                'texture': ['L20.H13', 'L21.H2', 'L22.H7'],
                'wildlife': ['L20.H1', 'L20.H9', 'L22.H13'],
                'birds': ['L22.H13', 'L23.H7'],
                'clothing': ['L23.H4', 'L23.H13']}

    return properties

model, _, preprocess = create_model_and_transforms(model_name, pretrained=pretrained)
model.to(device)
model.eval()

ds_vis = ImageNet(root=imagenet_path, split="val", transform=visualization_preprocess) # For showing images
ds = ImageNet(root=imagenet_path, split="val", transform=preprocess) # For running the model
dataloader = DataLoader(
    ds, batch_size=batch_size, shuffle=False, num_workers=8
)

prs = hook_prs_logger(model, device)

if(joint_name == 'ViT-B-16_laion2b_s34b_b88k'):
    cur_property_names = get_ViT_B_16_laion2b_s34b_b88k_property_head_layer_labels()
elif(joint_name == 'ViT-B-16_openai'):
    cur_property_names = get_ViT_B_16_openai_property_head_layer_labels()
elif(joint_name == 'ViT-B-32_datacomp_m_s128m_b4k'):
    cur_property_names = get_ViT_B_32_datacomp_m_s128m_b4k_property_head_layer_labels()
elif(joint_name == 'ViT-B-32_openai'):
    cur_property_names = get_ViT_B_32_openai_property_head_layer_labels()
elif(joint_name == 'ViT-L-14_laion2b_s32b_b82k'):
    cur_property_names = get_ViT_L_14_laion2b_s32b_b82k_property_head_layer_labels()
elif(joint_name == 'ViT-L-14_openai'):
    cur_property_names = get_ViT_L_14_openai_property_head_layer_labels()

start = 6
end = min(7, len(cur_property_names))
for property_name in list(cur_property_names.keys())[start:end]:
    print("Modelname: {} ##### property: {}".format(joint_name, property_name))
    attentions_maps = []
    head_layer_labels = cur_property_names[property_name]
    for index, (images, _) in tqdm.tqdm(enumerate(dataloader)):
        images = images.to(device)

        with torch.no_grad():
            prs.reinit()
            current_representation = model.encode_image(images, 
                                                        attn_method='head', 
                                                        normalize=False)
            current_attentions, _ = prs.finalize(current_representation)

            scores = current_attentions.sum(axis=2)

            for ii in range(scores.size()[0]):
                averaged_image_net_attentions = []
                for cur_layer_head in head_layer_labels:
                    layer_no, head_no = cur_layer_head.split()[0].split('.')
                    layer_no = int(layer_no[1:])
                    head_no = int(head_no[1:])
                    averaged_image_net_attentions.append(scores[ii, layer_no, head_no, :])
                
                averaged_image_net_attentions = torch.stack(averaged_image_net_attentions, 0)
                averaged_image_net_attentions = torch.mean(averaged_image_net_attentions, 0)
                attentions_maps.append(averaged_image_net_attentions.detach().cpu().numpy())

    attentions_maps = np.asarray(attentions_maps)
    print(attentions_maps.shape)

    with open('attention_weights/' + model_name + '_' + pretrained + '-' + property_name + '.pkl', 'wb') as f:
        pkl.dump(attentions_maps, f)
