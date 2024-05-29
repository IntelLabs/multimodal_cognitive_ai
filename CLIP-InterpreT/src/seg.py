import numpy as np
import cv2
import torch
from torch.nn import functional as F
import einops
from utils.factory import create_model_and_transforms, get_tokenizer
from prs_hook import hook_prs_logger
from matplotlib import pyplot as plt1
from matplotlib import pyplot as plt2
from HeatMap import HeatMap

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model_tokenizer(modelname):
    if(modelname == 'ViT-B-16_laion2b_s34b_b88k'):
        model_name = 'ViT-B-16'
        pretrained = 'laion2b_s34b_b88k'
    elif(modelname == 'ViT-B-16_openai'):
        model_name = 'ViT-B-16'
        pretrained = 'openai'
    elif(modelname == 'ViT-B-32_datacomp_m_s128m_b4k'):
        model_name = 'ViT-B-32'
        pretrained = 'datacomp_m_s128m_b4k'
    elif(modelname == 'ViT-B-32_openai'):
        model_name = 'ViT-B-32'
        pretrained = 'openai'
    elif(modelname == 'ViT-L-14_laion2b_s32b_b82k'):
        model_name = 'ViT-L-14'
        pretrained = 'laion2b_s32b_b82k'
    elif(modelname == 'ViT-L-14_openai'):
        model_name = 'ViT-L-14'
        pretrained = 'openai'

    model, _, preprocess = create_model_and_transforms(model_name, pretrained=pretrained)
    model.to(device)
    model.eval()

    tokenizer = get_tokenizer(model_name)

    prs = hook_prs_logger(model, device)
    return model, tokenizer, prs, preprocess

def plot_seg_masks(model, attentions, class_embedding, image_pil, modelname, layer_no, head_no):
    filenames = ['graph1', 'graph2']

    if((layer_no == "-1 (Aggregate)") and (head_no == "-1 (Aggregate)")):
        attention_map = attentions[0, :, 1:, :].sum(axis=(0,2)) @ class_embedding.T
    else:
        layer_no, head_no = int(layer_no), int(head_no)
        attention_map = attentions[0, layer_no, 1:, head_no, :] @ class_embedding.T

    if(modelname.startswith('ViT-B-16')):
        N_ = 14
        M_ = 14
    elif(modelname.startswith('ViT-B-32')):
        N_ = 7
        M_ = 7
    elif(modelname.startswith('ViT-L-14')):
        N_ = 16
        M_ = 16
    attention_map = F.interpolate(einops.rearrange(attention_map, '(B N M) C -> B C N M', N=N_, M=M_, B=1), 
                                  scale_factor=model.visual.patch_size[0],
                                  mode='bilinear').to(device)
    attention_map = attention_map[0].detach().cpu().numpy()

    plt1.figure()
    plt1.imshow(attention_map[0] - np.mean(attention_map,axis=0))

    v = attention_map[0] - attention_map[1]
    min_ = min((attention_map[0] - attention_map[1]).min(), (attention_map[1] - attention_map[0]).min())
    max_ = max((attention_map[0] - attention_map[1]).max(), (attention_map[1] - attention_map[1]).max())
    v = v - min_
    v = np.uint8((v / (max_ - min_)) * 255)
    high = cv2.cvtColor(cv2.applyColorMap(v, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    plt1.axis('off')
    hm = HeatMap(np.asarray(image_pil), v)
    hm.save(filenames[0], 'png', transparency=0.6,
            color_map='seismic',
            show_axis=False,
            show_original=False,
            show_colorbar=False,
            width_pad=-10)

    plt2.figure()
    plt2.imshow(attention_map[1] - np.mean(attention_map,axis=0),)

    v = attention_map[1] - attention_map[0]
    v = v - min_
    v = np.uint8((v / (max_ - min_)) * 255)
    high = cv2.cvtColor(cv2.applyColorMap(v, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    plt2.axis('off')
    hm = HeatMap(np.asarray(image_pil), v)
    hm.save(filenames[1], 'png', transparency=0.6,
            color_map='seismic',
            show_axis=False,
            show_original=False,
            show_colorbar=False,
            width_pad=-10)

    return [filenames[0] + '.png', filenames[1] + '.png']

def plot_topic_masks(model, attentions, class_embedding, image_pil, modelname, layer_no, head_no):
    
    fname = 'heatmap_result'
    if((layer_no == -1) and (head_no == -1)):
        attention_map = attentions[0, :, 1:, :].sum(axis=(0,2)) @ class_embedding.T
    else:
        attention_map = attentions[0, layer_no, 1:, head_no, :] @ class_embedding.T

    if(modelname.startswith('ViT-B-16')):
        N_ = 14
        M_ = 14
    elif(modelname.startswith('ViT-B-32')):
        N_ = 7
        M_ = 7
    elif(modelname.startswith('ViT-L-14')):
        N_ = 16
        M_ = 16

    attention_map = F.interpolate(einops.rearrange(attention_map, '(B N M) C -> B C N M', N=N_, M=M_, B=1), 
                                  scale_factor=model.visual.patch_size[0],
                                  mode='bilinear').to(device)
    attention_map = attention_map[0].detach().cpu().numpy()

    plt1.figure()
    plt1.imshow(attention_map[0])

    v = attention_map[0]
    min_ = attention_map[0].min()
    max_ = attention_map[0].max()
    v = v - min_
    v = np.uint8((v / (max_ - min_)) * 255)
    high = cv2.cvtColor(cv2.applyColorMap(v, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    plt1.axis('off')
    hm = HeatMap(np.asarray(image_pil), v)
    hm.save(fname, 'png', transparency=0.6,
            color_map='seismic',
            show_axis=False,
            show_original=False,
            show_colorbar=False,
            width_pad=-10)

    return fname + '.png'
