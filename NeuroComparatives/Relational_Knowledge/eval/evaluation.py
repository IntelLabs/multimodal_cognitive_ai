import json
import pandas as pd
import post_filter
import visualize
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch
import os
os.environ["CURL_CA_BUNDLE"]=""

def _read_txt(path):
    logs = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            logs.append(d)
    return logs

def get_models_dict(models=['facebook/bart-large']):
    model_dict = {}
    tokenizer_dict = {}
    for model_name in models:# BART
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelWithLMHead.from_pretrained(model_name)
        torch.cuda.empty_cache()    
        model.eval()
        model = model.to('cuda')
        model_dict[model_name] = model
        tokenizer_dict[model_name] = tokenizer
    return model_dict, tokenizer_dict


def rank_result(file_name,logs_path,model_dict, tokenizer_dict,save=True, models=['facebook/bart-large']):
    logs = _read_txt(logs_path)
    indexes = list(range(len(logs)))
    visualize.show_as_table([logs],indexes,save=save,file=file_name,models=models,model_dict = model_dict,tokenizer_dict = tokenizer_dict,)

def dedup_result(file_name, out_name, sent_model, sort_by = 'facebook/bart-large',method = "cluster",thresh=1.0):
    '''
    file_name: .csv
    '''
    data = pd.read_csv(file_name)
    data_dedup = filter.tables_dedup(sent_model,data,sort_by=sort_by,method = method, thresh=thresh)
    data_dedup.to_csv(out_name)

