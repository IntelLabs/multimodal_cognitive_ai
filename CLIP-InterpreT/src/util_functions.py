import heapq
from PIL import Image
import numpy as np
import torch
import pickle as pkl
import os.path

from torch.nn import functional as F
from torchvision.datasets import ImageFolder
from torchvision import transforms
import gradio as gradio

from seg import (
    get_model_tokenizer,
    plot_seg_masks, plot_topic_masks
)

from helper import (
    image_grid, _convert_to_rgb
)
from image_property_labels import (
    get_ViT_B_16_laion2b_s34b_b88k_property_head_layer_labels, get_ViT_B_16_openai_property_head_layer_labels,
    get_ViT_B_32_datacomp_m_s128m_b4k_property_head_layer_labels, get_ViT_B_32_openai_property_head_layer_labels,
    get_ViT_L_14_openai_property_head_layer_labels,
    get_ViT_L_14_laion2b_s32b_b82k_property_head_layer_labels,
)

image_path = 'data/' # Path to image folder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

def get_seg_mask(image_pil, input_text1, input_text2, modelname, drop_down_choice="Aggregate"):
    if(drop_down_choice == "Aggregate"):
        layer_no = -1
        head_no = -1
    else:
        layer_no, head_no = drop_down_choice.split()[0].split('.')
        layer_no = int(layer_no[1:])
        head_no = int(head_no[1:])

    model, tokenizer, prs, preprocess = get_model_tokenizer(modelname)
    image = preprocess(image_pil)[np.newaxis, :, :, :]

    prs.reinit()
    with torch.no_grad():
        representation = model.encode_image(image.to(device), 
                                            attn_method='head', 
                                            normalize=False)
        attentions, _ = prs.finalize(representation)

    texts = tokenizer([input_text1, input_text2]).to(device)  # tokenize
    class_embeddings = model.encode_text(texts)
    class_embedding = F.normalize(class_embeddings, dim=-1)

    filenames = plot_seg_masks(model, attentions, class_embedding, image_pil,  modelname, layer_no, head_no)
    return filenames[0], filenames[1]

def get_topic_mask(image_pil, input_text, modelname, drop_down_choice):
    if(drop_down_choice == "Aggregate"):
        layer_no = -1
        head_no = -1
    else:
        layer_no, head_no = drop_down_choice.split()[0].split('.')
        layer_no = int(layer_no[1:])
        head_no = int(head_no[1:])

    model, tokenizer, prs, preprocess = get_model_tokenizer(modelname)
    image = preprocess(image_pil)[np.newaxis, :, :, :]

    prs.reinit()
    with torch.no_grad():
        representation = model.encode_image(image.to(device), 
                                            attn_method='head', 
                                            normalize=False)
        attentions, _ = prs.finalize(representation)

    texts = tokenizer([input_text]).to(device)
    class_embeddings = model.encode_text(texts)
    class_embedding = F.normalize(class_embeddings, dim=-1)

    return plot_topic_masks(model, attentions, class_embedding, image_pil, modelname, layer_no, head_no)

def get_nearest_neighbours_image(image_pil, modelname, drop_down_choice):
    layer_no, head_no = drop_down_choice.split()[0].split('.')
    layer_no = int(layer_no[1:])
    head_no = int(head_no[1:])

    model, tokenizer, prs, preprocess = get_model_tokenizer(modelname)
    image = preprocess(image_pil)[np.newaxis, :, :, :]

    prs.reinit()
    with torch.no_grad():
        representation = model.encode_image(image.to(device), 
                                            attn_method='head', 
                                            normalize=False)
        attentions, _ = prs.finalize(representation)
    
    query = attentions[0, layer_no, :, head_no].sum(axis=0).detach().cpu()

    with open('attention_weights/' + modelname + '-layer' + str(layer_no) + '.pkl', 'rb') as f:
        current_attentions = pkl.load(f)
    
    limit = 8 # Number of nearest neighbours
    db = [(-float("inf"), None) for _ in range(limit)]
    for ii in range(current_attentions.shape[0]):
        scores = torch.from_numpy(current_attentions[ii, head_no, :]) @ query
        heapq.heappushpop(db, (scores, ii))
    
    db = sorted(db, key=lambda x: -x[0])
    
    visualization_preprocess = transforms.Compose(
    [
        transforms.Resize(size=224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(size=(224, 224)),
        _convert_to_rgb,
    ]
    )
    ds_vis = ImageFolder(image_path, transform=visualization_preprocess)

    name = 'grid_image_image.png'
    images = []
    for image_index in db:
        images.append(ds_vis[image_index[1]][0])
    
    return image_grid(images, 1, limit, name)

def get_nearest_neighbours_text(input_text, modelname, drop_down_choice):
    layer_no, head_no = drop_down_choice.split()[0].split('.')
    layer_no = int(layer_no[1:])
    head_no = int(head_no[1:])

    model, tokenizer, _, _ = get_model_tokenizer(modelname)
    texts = tokenizer([input_text]).to(device)  # tokenize
    class_embeddings = model.encode_text(texts)
    query = F.normalize(class_embeddings, dim=-1).squeeze().detach().cpu()

    with open('attention_weights/' + modelname + '-layer' + str(layer_no) + '.pkl', 'rb') as f:
        current_attentions = pkl.load(f)
    
    limit = 3 # Number of nearest neighbours
    db = [(-float("inf"), None) for _ in range(limit)]
    for ii in range(current_attentions.shape[0]):
        scores = torch.from_numpy(current_attentions[ii, head_no, :]) @ query
        heapq.heappushpop(db, (scores, ii))
    
    db = sorted(db, key=lambda x: -x[0])
    
    visualization_preprocess = transforms.Compose(
    [
        transforms.Resize(size=224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(size=(224, 224)),
        _convert_to_rgb,
    ]
    )
    ds_vis = ImageFolder(image_path, transform=visualization_preprocess)

    name = 'grid_image_text.png'
    images = []
    for image_index in db:
        images.append(ds_vis[image_index[1]][0])
    
    return image_grid(images, 1, limit, name)

def get_nearest_neighbours_image_with_properties(image_pil, modelname, propertyname):
    
    if(modelname == 'ViT-B-16_laion2b_s34b_b88k'):
        head_layer_labels = get_ViT_B_16_laion2b_s34b_b88k_property_head_layer_labels(propertyname)
    elif(modelname == 'ViT-B-16_openai'):
        head_layer_labels = get_ViT_B_16_openai_property_head_layer_labels(propertyname)
    elif(modelname == 'ViT-B-32_datacomp_m_s128m_b4k'):
        head_layer_labels = get_ViT_B_32_datacomp_m_s128m_b4k_property_head_layer_labels(propertyname)
    elif(modelname == 'ViT-B-32_openai'):
        head_layer_labels = get_ViT_B_32_openai_property_head_layer_labels(propertyname)
    elif(modelname == 'ViT-L-14_laion2b_s32b_b82k'):
        head_layer_labels = get_ViT_L_14_laion2b_s32b_b82k_property_head_layer_labels(propertyname)
    elif(modelname == 'ViT-L-14_openai'):
        head_layer_labels = get_ViT_L_14_openai_property_head_layer_labels(propertyname)

    all_layer_head_query_rep = []
    for cur_layer_head in head_layer_labels:
        layer_no, head_no = cur_layer_head.split()[0].split('.')
        layer_no = int(layer_no[1:])
        head_no = int(head_no[1:])

        model, tokenizer, prs, preprocess = get_model_tokenizer(modelname)
        image = preprocess(image_pil)[np.newaxis, :, :, :]

        prs.reinit()
        with torch.no_grad():
            representation = model.encode_image(image.to(device), 
                                                attn_method='head', 
                                                normalize=False)
            attentions, _ = prs.finalize(representation)
        
        all_layer_head_query_rep.append(attentions[0, layer_no, :, head_no].sum(axis=0).detach().cpu())
    
    query = torch.mean(torch.stack(all_layer_head_query_rep, 0), 0)

    with open('attention_weights/' + modelname + '-' + propertyname + '.pkl', 'rb') as f:
        current_attentions = pkl.load(f)

    limit = 4 # Number of nearest neighbours
    db = [(-float("inf"), None) for _ in range(limit)]
    for ii in range(current_attentions.shape[0]):
        scores = torch.from_numpy(current_attentions[ii]) @ query
        heapq.heappushpop(db, (scores, ii))
    db = sorted(db, key=lambda x: -x[0])
    
    visualization_preprocess = transforms.Compose(
    [
        transforms.Resize(size=224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(size=(224, 224)),
        _convert_to_rgb,
    ]
    )
    ds_vis = ImageFolder(image_path, transform=visualization_preprocess)

    name = 'grid_image_image_property.png'
    images = []
    for image_index in db:
        images.append(ds_vis[image_index[1]][0])
    
    return image_grid(images, 1, 4, name)
