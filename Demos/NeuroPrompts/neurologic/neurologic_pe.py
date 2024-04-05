import os
neurologic_path = os.environ['NEUROLOGIC_PATH']
os.environ['COMMON_POS_FILENAME'] = os.path.join(neurologic_path, 'token_common_pos_100K.tsv')

import sys
sys.path.insert(0, neurologic_path)

import json
import csv
import math
import argparse
import torch
import numpy as np
from tqdm import tqdm
import itertools
from pathlib import Path
import time
from os import path
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForCausalLM
import utils
from lexical_constraints import init_batch
from generate import generate
from prompt_constraints import load_inputs
import pkg_resources
sys.path.insert(0,'./data/')

def is_hpu_available():
    return 'habana-torch-plugin' in {pkg.key for pkg in pkg_resources.working_set}


if is_hpu_available():
    import habana_frameworks.torch.core as htcore
    device = torch.device("hpu")
else:
    device = torch.device("cuda")

print(f"device: {device}")


def generate_neurologic(input_prompt, model, tokenizer, model_type, constraint_method, clusters_file, 
                        user_constraints=None,
                        negative_constraints=None,
                        n_per_prompt=1, 
                        n_clusters = 5, 
                        n_per_cluster = 5, 
                        constraint_limit=None, 
                        batch_size=1, 
                        beam_size=15, 
                        max_tgt_length=32,
                        min_tgt_length=5, 
                        ngram_size=None,
                        length_penalty=None,
                        sat_tolerance=2,
                        beta=1.25,
                        early_stop=None,
                        num_return_sequences=10,
                        diversity=True,
                        do_beam_sample=False,
                        beam_temperature=10000000,
                        return_all_hyps=False,
                        return_all=False,
                        lp_return_size=3,
                        stochastic=False,
                        length_penalties=[0.1, 0.4, 0.7, 1.0],
                        ordered=True,
                        look_ahead=0,
                        seed=0):
    
    if model_type == 'ppo':
        tokenizer.pad_token = tokenizer.eos_token
        PAD_ID = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        eos_token_id=50256
    else:
        PAD_ID = tokenizer.convert_tokens_to_ids('<pad>')
        eos_token_id = tokenizer.eos_token_id


    torch.cuda.empty_cache()
    model.eval()
    model = model.to(device)

    # 2. Token Processing & General Constraints
    print(f'vocab_size: {tokenizer.vocab_size}, POS_size: {len(utils.POS)}')
    digit_num = 10 ** (len(str(tokenizer.vocab_size)) - 1)
    POS_start = (tokenizer.vocab_size // digit_num + 1) * digit_num
    POS = {k: v + POS_start for k, v in utils.POS.items()}
    period_id = [tokenizer.convert_tokens_to_ids('.')]
    period_id.append(tokenizer.convert_tokens_to_ids('Ġ.'))
    eos_ids = [tokenizer.eos_token_id] + period_id
    bad_token = ['?', ':', "'", '"', ',"', '."', '\u2014', '-', '_', '@', 'Ċ', 'Ġ:', 'Ġ(', 'Ġ)', '(', ')', 'Ġ[', 'Ġ]', '[', ']', 'Ġ{', 'Ġ}', '{', '}']
    bad_words_ids = [tokenizer.convert_tokens_to_ids([t]) for t in bad_token]
    if model_type == 'ppo':
        bad_words_ids = [i for i in bad_words_ids if i != [eos_token_id]]

    print(PAD_ID)
    print(eos_ids)

    max_n = None
    data = load_inputs(input_prompt, constraint_method, clusters_file, max_n, model_type, 
                       n_per_prompt=n_per_prompt, n_clusters=n_clusters, n_per_cluster=n_per_cluster, 
                       user_constraints=user_constraints, negative_constraints=negative_constraints,
                       seed=seed)
    print(f"data length: {len(data)}")
    flat_data = []
    print("processing data")
    for d in tqdm(data):
        for ins in d["prompt_and_constraint"]:
            constraint_list_str = ins["cur_constraints_str"]
            constraints_list = utils.tokenize_constraints(tokenizer, POS, [constraint_list_str])
            constraints = init_batch(raw_constraints=constraints_list, beam_size=beam_size, eos_id=eos_ids,ordered=ordered)
            batch = tokenizer(ins["input_id"], return_tensors="pt")
            flat_data.append({"input_id":batch['input_ids'], 'attention_mask' : batch['attention_mask'],"constraint":constraints,'cur_constraints_str':constraint_list_str,'original_prompt':d["original_prompt"],'index':d['index']})
    print(flat_data[0])
    
    constraints_lists = [[[]] for x in range(len(data))]
    num_concept = len(data)
    
    print(f"num_concept: {num_concept}")
    print(f"constraint_path[0]: {constraints_lists[0]}")

    logs = []
    second_entity = None
    prompts = []
    times = []
    output_sequences = []
    print(f"\nStart Generation:")
    buf = defaultdict(list)
    with tqdm(total=len(flat_data)/batch_size) as pbar:
        for i in range(0,len(flat_data),batch_size):
            batch = flat_data[i:i+batch_size]
            input_ids = [x["input_id"] for x in batch]
            attention_mask = [x["attention_mask"] for x in batch]
            batch_input_ids = torch.cat(input_ids,dim=0).to(device)
            batch_attention_mask = torch.cat(attention_mask,dim=0).to(device)
            constraints = [x["constraint"] for x in batch]
            batch_constraints = list(itertools.chain(*constraints))
            start = time.time()
            outputs = generate(self=model,
                                input_ids=batch_input_ids,
                                attention_mask=batch_attention_mask,
                                pad_token_id=PAD_ID,
                                bad_words_ids=bad_words_ids,
                                min_length=min_tgt_length,
                                max_length=max_tgt_length + batch_input_ids.shape[1],
                                num_beams=beam_size,
                                no_repeat_ngram_size=ngram_size,
                                constraints=batch_constraints,
                                length_penalty=length_penalty, # this penalty only affects when the beam is finished.
                                sat_tolerance=sat_tolerance, 
                                num_sequences_per_return=num_return_sequences, 
                                do_beam_sample=do_beam_sample, 
                                beam_temperature=beam_temperature,
                                diversity=diversity,
                                return_all=return_all,
                                length_penalties=length_penalties, 
                                lp_return_size=lp_return_size,
                                stochastic = stochastic,
                                constraint_limit=constraint_limit,
                                look_ahead=look_ahead, 
                                eos_token_id=eos_token_id,
                                use_cache=True)
            end = time.time()
            elapsed = (end -start)/60
            times.append(elapsed)
            print(elapsed)
            for j in range(len(input_ids)):
                original_prompt = batch[j]["original_prompt"]
                index = batch[j]["index"]
                input_id = input_ids[j]
                cur_constraints_str_clause = batch[j]["cur_constraints_str"]
                print(f"prompt: {tokenizer.decode(input_id[0])} cur_constraints_str: {(cur_constraints_str_clause[:3], cur_constraints_str_clause[-1])} time: {elapsed} minutes")
                out_seqs = outputs["decoded"][j]
                lps = outputs["lps"][j]
                scores = [x[0] for x in outputs["all_hyps"][j]]
                decoded_output_sequences = []
                for z in range(out_seqs.shape[0]):
                    seq = out_seqs[z,:]
                    decoded = tokenizer.decode(seq)
                    generation = decoded.split('<|endoftext|>')[0].rstrip()
                    generation = generation.strip().replace('<|endoftext|>', '')
                    decoded_output_sequences.append(generation)
                all_hyps = [tokenizer.decode(x[1]).split('<|endoftext|>')[0].rstrip() for x in outputs["all_hyps"][j]]
                out = {'prompt': tokenizer.decode(input_id[0]),'index': index,'constraints':cur_constraints_str_clause, 'generation': decoded_output_sequences,"lps":lps}
                if return_all_hyps:
                    out["all_hyps"] = all_hyps
                    out["scores"] = scores
                buf[original_prompt].append(out)
            output_sequences.append(decoded_output_sequences)
            pbar.update(1)
    print(f"average time per iter: {sum(times)/len(times)}")

    return output_sequences
