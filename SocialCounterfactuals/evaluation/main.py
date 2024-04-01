import os
import json
import argparse
import logging
from tkinter import N
logging.disable(logging.INFO) 
logging.disable(logging.WARNING)
import pandas as pd
import torch
from load_data import (
    load_probe_data,
    load_downstream
)
import vision_models
from train import (
    evaluate_skew, evaluate_ndkl, evaluate_biask,
    save_cosine_scores,
    evaluate_ndkl_modified
)

slumr_data_dir = '/export/share/projects/mcai/counterfactuals/mm_bias/final_dataset/'
data_dir = 'data/'
logs_dir = 'logs/'
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

datapaths = {}
datapaths['occupation-race-gender'] = [slumr_data_dir + 'race_gender/']
datapaths['occupation-physicaltrait-gender'] = [slumr_data_dir + 'physical_gender/']
datapaths['occupation-physicaltrait-race'] = [slumr_data_dir + 'physical_race/']
datapaths['occupation-physicaltrait-race-gender'] = [slumr_data_dir + 'output_v2/physicaltrait_race_gender_8_occupations/20240129_154940/filtered_all/']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', type=str, default='openai/clip-vit-base-patch32', help='Type of tasks')
    parser.add_argument('--probename', type=str, default='trait-race-gender', help='Type of tasks')
    parser.add_argument('--datasetname', type=str, default=None, help='Type of dataset')
    parser.add_argument('--biasvar', type=str, default='gender', help='Bias variable')
    parser.add_argument('--parbiasvar', type=str, default=None, help='Bias variable')
    parser.add_argument('--keywordtype', type=str, default=None, help='Train/Test')
    parser.add_argument('--probesplit', type=str, default=None, help='Test/None')
    parser.add_argument('--isinterbias', type=bool, default=False, help='Bias variable')
    parser.add_argument('--topk', type=int, default=0, help='Number of ranks')
    parser.add_argument('--minimages', type=int, default=0, help='Minimum number of images')
    parser.add_argument('--metric', type=str, default='skew', help='Bias metric')
    parser.add_argument('--savecosine', type=bool, default=False, help='Compute individual cosine scores')

    args = parser.parse_args()
    modelname = args.modelname
    probename = args.probename
    keyword = probename.split('-')[0]
    biasvar = args.biasvar
    datasetname = args.datasetname
    parbiasvar = args.parbiasvar
    keywordtype = args.keywordtype
    probesplit = args.probesplit
    isinterbias = args.isinterbias
    topk = args.topk
    minimages = args.minimages
    metric = args.metric
    savecosine = args.savecosine

    if(datasetname is None):
        if(probesplit == None):
            all_images = []
            all_images_joined = []
            for cur_dir in datapaths[probename]:
                for x in os.listdir(cur_dir):
                    all_images.append(x)
                    all_images_joined.append(os.path.join(cur_dir, x))
        else:
            df_split = pd.read_csv(data_dir + probename + '-split-test.csv')
            all_images = df_split['images'].tolist()
            all_images_joined = df_split['images_joined'].to_list()

    if(datasetname is None):
        images, images_joined, labels, labels_cnt, label_names, keywords_present = load_probe_data(all_images,
                                                                    all_images_joined,
                                                                    data_dir, probename, keyword, 
                                                                    biasvar, parbiasvar, keywordtype, isinterbias,
                                                                    topk, minimages)
    else:
        images, images_joined, labels, labels_cnt, label_names, keywords_present = load_downstream(data_dir,
                                                                                                datasetname,
                                                                                                biasvar)

    print(labels_cnt)
    max_topk = min(sum(cur_label_cnt) for cur_label_cnt in labels_cnt)

    if(topk < len(label_names)):
        topk = len(label_names)
    
    #if(topk > max_topk):
    #    topk = max_topk

    if(minimages == 0):
        minimages == topk

    print("############################################################################################################################")
    #print(f'Examples of image sets: {images[0]}')
    #print(f'Examples of image sets: {images_joined[0]}')
    #print(f'Examples of label sets: {labels[0:2]}')
    print(f'Label counts: {labels_cnt[0:2]}')
    print(f'Label names: {label_names}')
    print(f'Key words present: {keywords_present[0:2]}')
    print("****************************************************************************************************************************")
    print(f'Number of image sets: {len(images)}')
    print(f'Number of label sets: {len(labels)}')
    print(f'Number of Label counts: {len(labels_cnt)}')
    print(f'Number of Labels: {len(label_names)}')
    print(f'Number of keywords: {len(keywords_present)}')
    print("****************************************************************************************************************************")
    print(f'Model name: {modelname}')
    print(f'Probe Name: {probename}')
    if(not isinterbias):
        print(f'Bias variable: {biasvar}')
    else:
        interbias = probename.split('-')[1] + '-' + probename.split('-')[2]
        print(f'Inter Bias: {interbias}')
    print(f'Partial Bias: {parbiasvar}')
    print(f'Number of ranks: {topk}')
    print(f'Minimum images: {minimages}')
    print(f'Maximum topk: {max_topk}')
    print(f'Bias metric: {metric}')

    if(('clip-' in modelname.lower()) or ('-clip' in modelname.lower())):
        save_model_name = 'clip'
        model = vision_models.CLIP_zeroshot(model_name=args.modelname).to(device)
    elif('slip' in modelname.lower()):
        save_model_name = 'slip'
        model = vision_models.SLIP_zeroshot(model_name=args.modelname).to(device)
    elif('alip' in modelname.lower()):
        save_model_name = 'alip'
        model = vision_models.ALIP_zeroshot(model_name=args.modelname).to(device)
    elif('flava' in modelname.lower()):
        save_model_name = 'flava'
        model = vision_models.FLAVA_zeroshot(model_name=args.modelname).to(device)
    elif('openclip' in modelname.lower()):
        save_model_name = 'openclip'
        model = vision_models.OpenCLIP_zeroshot().to(device)
    elif('laclip' in modelname.lower()):
        save_model_name = 'laclip'
        model = vision_models.LaCLIP_zeroshot(model_name=args.modelname).to(device)
    elif('blip2' in modelname.lower()):
        save_model_name = 'blip2'
        model = vision_models.BLIP2_zeroshot(model_name=args.modelname).to(device)

    print("****************************************************************************************************************************")
    save_file = 'scores/'
    save_model_name = 'alip'
    if(not isinterbias):
        save_file += save_model_name + '_' + probename + '_' + biasvar  + '_skew_topk_' + str(topk) + '_minimages_' + str(minimages)
    elif((args.parbiasvar is not None) and (isinterbias)):
        save_file += save_model_name + '_' + probename + '_' + interbias + '_partialbias_' + parbiasvar + '_skew_topk_' + str(topk) + '_minimages_' + str(minimages)
    else:
        save_file += save_model_name + '_' + probename + '_' + interbias 
        if(save_cosine_scores):
            save_file = save_file + '_' + 'savecosine'
    
    if(metric == 'skew'):
        if(not savecosine):
            final_scores = evaluate_skew(model, images, images_joined, labels, labels_cnt, label_names, 
                                                    keywords_present,
                                                    keyword, biasvar, isinterbias, topk, save_model_name, probename)
            print(f'Final Score of the task: {final_scores}')
        else:
            df = save_cosine_scores(model, all_images, all_images_joined, probename, save_model_name, keyword, keywordtype,
                                    parbiasvar, isinterbias)
            df.to_excel(save_file + '.xlsx', index=False, header=True)
    
    elif(metric == 'ndkl'):
        final_scores = evaluate_ndkl(model, images, images_joined, labels, labels_cnt, label_names, 
                                    keywords_present,
                                    keyword, biasvar, isinterbias, topk, save_model_name, probename)
        
        print(f'Average NDKL Score of the task: {final_scores}')
    elif(metric == 'ndkl-modified'):
        final_scores = evaluate_ndkl_modified(model, images, images_joined, labels, labels_cnt, label_names, 
                                    keywords_present,
                                    keyword, biasvar, isinterbias, topk, save_model_name, probename)
        
        print(f'Average NDKL modified Score of the task: {final_scores}')
    elif(metric == 'biask'):
        final_scores = evaluate_biask(model, images, images_joined, labels, labels_cnt, label_names, 
                                                keywords_present,
                                                keyword, biasvar, isinterbias, topk, save_model_name, probename)
        print(f'Average Bias@K Score of the task: {final_scores}')

if __name__ == '__main__':
    main()
