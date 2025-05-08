import evaluation
import visualize
import post_filter
import sys
sys.path.insert(0,'../data/')
from sentence_transformers import SentenceTransformer
import os
import pickle
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelWithLMHead
from filter import run_filtering


model_dict, tokenizer_dict = evaluation.get_models_dict()
sent_model = SentenceTransformer('sentence-transformers/sentence-t5-xl').to("cuda")

files = ["out_ent_paris100k_sampled50_top10k"]
all_hyps = True
thresh = 0.5
filter_on_first_noun = False
entity_constraint_set = False
n_cpu = 15
method = "cluster"
method = "self-bleu"

for input_file in files:
    gen_txt_filename = f"../Generated/wikidata/{input_file}.txt"
    dir = f"./eval_results/{input_file}/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    gen_csv_filename = dir + f"{input_file}.csv"
    gen_csv_dedup_filename = dir + f"{input_file}_dedup{str(thresh)}.csv"
    gen_csv_dedup_filter_filename = dir + f"{input_file}_dedup{str(thresh)}_filter.pkl"
    final_visualize_filename = dir + f"{input_file}_dedup{str(thresh)}_filter.csv"

    # 1.2 load generated knoweldges
    if not os.path.isfile(gen_csv_filename):
        data = evaluation._read_txt(gen_txt_filename)
        indexes = list(range(len(data)))
        visualize.show_as_table([data],indexes,file=gen_csv_filename, save=True, models=[],model_dict = model_dict,tokenizer_dict = tokenizer_dict,second_entity=True,all_hyps=all_hyps)#'facebook/bart-large'
    gen_csv = pd.read_csv(gen_csv_filename)
    gen_csv = gen_csv.drop(columns="Unnamed: 0")

    #2 Dedup Data
    if method == "memthod":
        gen_csv_dedup_filename_show_all = dir+ f"{input_file}_dedup{str(thresh)}_selfbleu.csv"
        gen_csv_dedup_filename = gen_csv_dedup_filename_show_all
    if not os.path.isfile(gen_csv_dedup_filename): 
        gen_csv_dedup = post_filter.tables_dedup(sent_model,gen_csv,sort_by='orignal_scores',method = method, 
        thresh=thresh,return_all_in_cluster=True)
        gen_csv_dedup.to_csv(gen_csv_dedup_filename)
    else:
        gen_csv_dedup = pd.read_csv(gen_csv_dedup_filename)
    
    # run_filtering(input_file, filter_on_first_noun, entity_constraint_set, n_cpu)