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
import argparse
from filter import run_filtering

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="result file to be filtered")
    parser.add_argument("--output_file", type=str, help="output")
    parser.add_argument("--thresh", type=float, help="cluster argument")
    parser.add_argument('--filter_on_first_noun', action='store_true',
                            help="")
    parser.add_argument('--entity_constraint_set', action='store_true',
                            help="")
    parser.add_argument("--n_cpu", type=int, help="# of cpus")
    parser.add_argument('--all_hyps', action='store_true',
                            help="whether to return all the hyps")
    parser.add_argument('--ordered', action='store_true',
                            help="whether ordered constraint satisfaction was used during generation")
    args = parser.parse_args()
    model_dict, tokenizer_dict = evaluation.get_models_dict()
    sent_model = SentenceTransformer('sentence-transformers/sentence-t5-xl').to("cuda")

    files = [args.input_file]
    all_hyps = args.all_hyps
    thresh = args.thresh
    print(f"start filtering")
    for input_file in files:
        gen_txt_filename = input_file
        base = os.path.basename(args.output_file).replace('.csv','')
        dir = os.path.dirname(args.output_file)
        gen_csv_filename = os.path.join(dir, f"{base}.csv")
        gen_csv_dedup_filename = os.path.join(dir, f"{base}_dedup.csv")
        print(gen_csv_filename,gen_csv_dedup_filename)

        # 1.2 load generated knoweldges
        if not os.path.isfile(gen_csv_filename):
            data = evaluation._read_txt(gen_txt_filename)
            indexes = list(range(len(data)))
            visualize.show_as_table([data],indexes,file=gen_csv_filename, save=True, models=[],model_dict = model_dict,tokenizer_dict = tokenizer_dict,second_entity=True,all_hyps=all_hyps)
        gen_csv = pd.read_csv(gen_csv_filename)
        gen_csv = gen_csv.drop(columns="Unnamed: 0")

        #2 Dedup Data
        if not os.path.isfile(gen_csv_dedup_filename): 
            gen_csv_dedup = post_filter.tables_dedup(sent_model,gen_csv,sort_by='orignal_scores',method = "cluster", 
            thresh=thresh,return_all_in_cluster=True)
            gen_csv_dedup.to_csv(gen_csv_dedup_filename)
        else:
            gen_csv_dedup = pd.read_csv(gen_csv_dedup_filename)

        #3 
        run_filtering(gen_csv_dedup_filename, args.filter_on_first_noun, args.entity_constraint_set, args.n_cpu, args.ordered)
    print("finished")
if __name__ == "__main__":
    main()
