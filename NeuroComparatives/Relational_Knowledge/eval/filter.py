import pandas as pd
import torch
import pickle
import numpy as np
import spacy
import time
import os
import hashlib
from tqdm import tqdm
from joblib import Parallel, delayed
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def preprocess_generations(file_path):
    gen = pd.read_csv(file_path, skip_blank_lines=False)
    gen = gen.iloc[:,1:]
    gen = gen.rename(columns = {'knowledge 0' : 'knowledge'})
    
    cols = ['1st entity', '2nd entity', 'class']
    col_index = [list(gen.columns).index(i) for i in cols]
    
    gen['cluster'] = gen.isna().apply(lambda x: all(x), axis=1).astype(int).cumsum()
    non_na = list(gen[cols].dropna().index) + [gen.shape[0]]
    for i in tqdm(range(len(non_na)-1)):
        for col in col_index:
            gen.iloc[non_na[i]+1:non_na[i+1], col] = gen.iloc[non_na[i], col]
    
    gen = gen.dropna(subset=['prompt'])
    gen['2nd entity satisfied'] = gen.apply(lambda x: x['2nd entity'] in x['knowledge'], axis=1)
    
    return gen


def pos_filter(gen, filter_on_first_noun, entity_constraint_set, ordered):
    nlp = spacy.load("en_core_web_sm")
    adverbs = ['more','less']
    non_comparisons = ['other']
    
    adj_all = []; adj_comp_all = []; adv_all = []; adv_comp_all = []; pos_match_all = []; first_noun_all = []; than_all = [];  first_noun_chunk_all = []; first_tk_all = []; last_tk_all = [];
    for i in range(gen.shape[0]):
        continuation = gen.iloc[i]['knowledge'][len(gen.iloc[i]['prompt']):].strip()
        doc = nlp(continuation)

        adj = [tk.text for tk in doc if tk.pos_ == 'ADJ']
        adj_comp = [tk for tk in adj if (tk.endswith('er') and tk not in non_comparisons) or tk in adverbs]
        adv = [tk.text for tk in doc if tk.pos_ == 'ADV']
        adv_comp = [tk for tk in adv if tk in adverbs]

        tokens = [tk.text for tk in doc]
        if len(tokens) == 0:
            adj_all.append('')
            adj_comp_all.append('')
            adv_all.append('')
            adv_comp_all.append('')
            pos_match_all.append('')
            first_noun_all.append('')
            than_all.append(False)
            first_noun_chunk_all.append('')
            first_tk_all.append('')
            last_tk_all.append('')
            continue
        pos = [tk.pos_ for tk in doc]
        pos_match = [' '.join(tokens[i:i+4]) for i in range(len(tokens)-3) if pos[i:i+4] in [['NOUN', 'AUX', 'DET', 'ADJ'], ['NOUN', 'AUX', 'ADV', 'ADJ']]]
        pos_match += [' '.join(tokens[i:i+5]) for i in range(len(tokens)-4) if pos[i:i+5] in [['NOUN', 'AUX', 'ADV', 'VERB', 'ADJ'], ['NOUN', 'ADV', 'VERB', 'DET', 'ADJ'], ['NOUN', 'AUX', 'ADV', 'AUX', 'ADJ']]]
        
        first_noun_start = None; first_noun_end = None
        for j in range(len(pos)):
            if pos[j] == 'NOUN' and first_noun_start is None:
                first_noun_start = j
            elif first_noun_start is not None and pos[j] != 'NOUN':
                first_noun_end = j
                break
        first_noun = ' '.join(tokens[first_noun_start:first_noun_end])
        first_noun_chunk = [chunk.text for chunk in doc.noun_chunks]
        first_noun_chunk = '' if len(first_noun_chunk) == 0 else first_noun_chunk[0]

        adj_all.append(adj)
        adj_comp_all.append(adj_comp)
        adv_all.append(adv)
        adv_comp_all.append(adv_comp)
        pos_match_all.append(pos_match)
        first_noun_all.append(first_noun)
        than_all.append('than' in tokens)
        first_noun_chunk_all.append(first_noun_chunk)
        first_tk_all.append(tokens[0])
        if len(tokens) == 1 or tokens[-1] != '.':
            last_tk_all.append(tokens[-1])
        else:
            last_tk_all.append(tokens[-2])
        
    gen['adj'] = adj_all
    gen['adj_comp'] = adj_comp_all
    gen['adv'] = adv_all
    gen['adv_comp'] = adv_comp_all
    gen['pos_match'] = pos_match_all
    gen['first_noun'] = first_noun_all
    gen['than'] = than_all
    gen['first_noun_chunk'] = first_noun_chunk_all
    gen['first_tk'] = first_tk_all
    gen['last_tk'] = last_tk_all

    if ordered and entity_constraint_set:
        gen['adverb_constraint'] = gen['positive constriants'].apply(lambda x: x.split(';')[2])
        gen['aux_verb_constraint'] = gen['positive constriants'].apply(lambda x: x.split(';')[1])
        gen['entity_constraint'] = gen['positive constriants'].apply(lambda x: x.split(';')[0])
    elif ordered:
        gen['adverb_constraint'] = gen['positive constriants'].apply(lambda x: x.split(';')[1])
        gen['aux_verb_constraint'] = gen['positive constriants'].apply(lambda x: x.split(';')[0])
        gen['entity_constraint'] = None
    else:
        gen['adverb_constraint'] = gen['positive constriants'].apply(lambda x: x.split(';')[0])
        gen['aux_verb_constraint'] = gen['positive constriants'].apply(lambda x: x.split(';')[1])
        gen['entity_constraint'] = gen['positive constriants'].apply(lambda x: x.split(';')[2]) if entity_constraint_set else None

    gen['first_tk_is_adverb_or_noun'] = gen.apply(lambda x: x['first_tk'] not in ['which', 'always'] and any([x['first_tk'] in j for j in [x['first_noun_chunk'], x['adverb_constraint']]]), axis=1)
    gen['last_tk_is_aux_verb'] = gen.apply(lambda x: x['last_tk'] == x['aux_verb_constraint'], axis=1)
    gen['last_tk_is_adverb'] = gen.apply(lambda x: x['last_tk'] == x['adverb_constraint'], axis=1)

    if entity_constraint_set:
        gen['first_noun_is_entity2'] = gen.apply(lambda x: any([j in x['first_noun'] for j in x['entity_constraint'].split(',')]), axis=1)
        gen['ent2_satisfied'] = gen.apply(lambda x: [j for j in x['entity_constraint'].split(',') if j in x['first_noun']], axis=1).astype(str)
    else:
        gen['first_noun_is_entity2'] = gen.apply(lambda x: x['2nd entity'] in x['first_noun'], axis=1)

    keep_index = ((gen['adj_comp'].str.len() > 0) | (gen['adv_comp'].str.len() > 0)| (gen['pos_match'].str.len() > 0)) & (~gen['than']) & (~gen['last_tk_is_aux_verb']) & (~gen['last_tk_is_adverb'])
    if filter_on_first_noun:
        keep_index = keep_index & gen['first_noun_is_entity2'] & gen['first_tk_is_adverb_or_noun']

    gen['keep'] = False
    gen.loc[keep_index, 'keep'] = True

    return gen


def group_by_constraint_pos(gen, entity_constraint_set):
    gen['adj_comp'] = gen['adj_comp'].astype(str)
    gen['adv_comp'] = gen['adv_comp'].astype(str)
    gen['pos_match'] = gen['pos_match'].astype(str)

    group_cols = ['1st entity', '2nd entity', 'positive constriants']
    if entity_constraint_set:
        group_cols += ['ent2_satisfied']
    
    gen1 = gen[gen['adj_comp'].str.len() > 0]
    gen1 = gen1.sort_values(group_cols + ['adj_comp', 'orignal_scores'], ascending=False).drop_duplicates(group_cols + ['adj_comp'])
    
    gen2 = gen[(gen['adj_comp'].str.len() == 0) & (gen['adv_comp'].str.len() > 0)]
    gen2 = gen2.sort_values(group_cols + ['adv_comp', 'adj', 'orignal_scores'], ascending=False).drop_duplicates(group_cols + ['adv_comp', 'adj'])
    
    gen3 = gen[(gen['adj_comp'].str.len() == 0) & (gen['adv_comp'].str.len() == 0) & (gen['pos_match'].str.len() > 0)]
    gen3 = gen3.sort_values(group_cols + ['pos_match', 'adj', 'orignal_scores'], ascending=False).drop_duplicates(group_cols + ['pos_match', 'adj'])
    
    gen_grouped = pd.concat([gen1, gen2, gen3])
    if entity_constraint_set:
        gen_grouped = gen_grouped.sort_values(['1st entity', '2nd entity', 'ent2_satisfied', 'orignal_scores'], ascending=False)
    else:
        gen_grouped = gen_grouped.sort_values(['1st entity', '2nd entity', 'orignal_scores'], ascending=False)
    
    return gen_grouped


def classify_entailment(model, tokenizer, premise, hypothesis, max_length = 256, batch_size=8):
    max_val_list = []; max_index_list = []
    for b in range(int(np.ceil(len(hypothesis)/batch_size))):
        pairs = [(premise, h) for h in hypothesis[(b*batch_size):((b+1)*batch_size)]]

        tokenized_input_seq_pair = tokenizer.batch_encode_plus(pairs, max_length=max_length, return_token_type_ids=True, truncation=True, padding=True)
        input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().to("cuda")
        token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().to("cuda")
        attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().to("cuda")

        outputs = model(input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=None)

        predicted_probability = torch.softmax(outputs[0], dim=1).detach().cpu()
        max_val, max_index = predicted_probability.max(dim=1)
        max_val_list.append(max_val)
        max_index_list.append(max_index)
    
    max_val = torch.concat(max_val_list)
    max_index = torch.concat(max_index_list)
    
    return [max_val, max_index]


def contradiction_filter(gen, gen_all, top_n):
    tokenizer = AutoTokenizer.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli", use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli").to("cuda")
    
    ent_pairs = gen[['1st entity', '2nd entity', 'prompt']].value_counts().index
    gen_out = []
    for ent_tuple in tqdm(ent_pairs):
        gen_subset = gen[(gen['1st entity'] == ent_tuple[0]) & (gen['2nd entity'] == ent_tuple[1]) & (gen['prompt'] == ent_tuple[2])]
        n = gen_subset.shape[0]
        contradict = [None]*n; gen_contradict_index = [None]*n; gen_entail_index = [None]*n; n_contradict = [None]*n; n_entail = [None]*n
        i = 0; not_contradicted = 0
        while not_contradicted < top_n and i < n:
            other_gen = gen_all[(gen_all['prompt'] == gen_subset.iloc[i]['prompt']) & (gen_all['knowledge'] != gen_subset.iloc[i]['knowledge'])]
            premise = gen_subset.iloc[i]['knowledge'].split(',')[-1].strip()
            hypothesis = other_gen['knowledge'].apply(lambda x: x.split(',')[-1].strip()).tolist()
            if len(hypothesis) == 0:
                i+=1
                continue
            
            max_val, max_index = classify_entailment(model, tokenizer, premise, hypothesis)
            entail_index = ((max_val >= 0.85) & (max_index == 0))
            df_entail = other_gen.iloc[entail_index.tolist()].assign(prob = max_val[entail_index])
            df_entail = pd.concat([df_entail, gen_subset.iloc[i:i+1]])
            contradict_index = ((max_val >= 0.99) & (max_index == 2))
            df_contradict = other_gen.iloc[contradict_index.tolist()].assign(prob = max_val[contradict_index])
            
            contradict[i] = df_entail['cluster_size'].sum() <= df_contradict['cluster_size'].sum()
            n_contradict[i] = df_contradict['cluster_size'].sum()
            n_entail[i] = df_entail['cluster_size'].sum()
            gen_contradict_index[i] = df_contradict['id'].tolist()
            gen_entail_index[i] = df_entail['id'].tolist()
            
            if not contradict[i]:
                not_contradicted+=1
            i+=1
        
        gen_subset = gen_subset.assign(n_entail = n_entail, n_contradict = n_contradict, contradict_index = gen_contradict_index, entail_index = gen_entail_index, contradicted = contradict)
        gen_out.append(gen_subset)
        
    return pd.concat(gen_out)

def run_filtering(file_path, filter_on_first_noun, entity_constraint_set, n_cpu, ordered):
    gen = preprocess_generations(file_path)
    
    gen = pd.concat(Parallel(n_jobs=n_cpu)(delayed(pos_filter)(gen_subset, filter_on_first_noun, entity_constraint_set, ordered) for gen_subset in np.array_split(gen, n_cpu)))
    gen = gen.assign(id = gen['knowledge'].apply(lambda x: hashlib.sha1(x.encode('utf-8')).hexdigest()))
    gen.to_csv(file_path[:-4] + '_unfiltered.csv', index=False)

    gen_filtered = gen[gen['keep']]
    cluster_ct = gen_filtered[['1st entity', '2nd entity','cluster']].value_counts().reset_index().rename(columns={0:'cluster_size'})
    gen_filtered = pd.merge(gen_filtered, cluster_ct, on = ['1st entity', '2nd entity','cluster'], how = 'left')
    
    gen_filtered = gen_filtered.sort_values(['1st entity', '2nd entity', 'cluster','orignal_scores'], ascending=False)
    gen_filtered = gen_filtered.groupby(['1st entity', '2nd entity', 'cluster']).head(1)
    
    gen_filtered = gen_filtered.sort_values(['1st entity', '2nd entity', 'orignal_scores'], ascending=False)
    gen_filtered.to_csv(file_path[:-4] + '_filtered.csv', index=False)

    gen_top1_by_constraint_pos = group_by_constraint_pos(gen_filtered, entity_constraint_set)
    gen_top1_by_constraint_pos = contradiction_filter(gen_top1_by_constraint_pos, gen_filtered, top_n = 5)
    gen_top1_by_constraint_pos.to_csv(file_path[:-4] + '_top1_by_constraint_pos_filtered.csv', index=False)
    
    gen_top1_by_constraint_pos_contradiction_filtered = gen_top1_by_constraint_pos[gen_top1_by_constraint_pos['contradicted'] == False]
    gen_top1_by_constraint_pos_contradiction_filtered.to_csv(file_path[:-4] + '_top1_by_constraint_pos_contradiction_filtered.csv', index=False)
