import metrics
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering 
from metrics import bleu_list_n
from nltk.translate.bleu_score import SmoothingFunction
smoothing_function = SmoothingFunction().method1
def show_tables_by_length(data, sort_by = "gpt2-large", cutoff=8.5):
    length = len(data.index)
    if "Segments" not in data:
        data.insert(4, "Segments", [np.nan]+ ["<10"]*(length-1))
    prompt_indexes = np.argwhere(data.loc[:,"1st entity"].notna().values).tolist()
    ranges = [(0,9), (9,11), (11,14),(14,100)]
    new_prompt_dfs = []
    for idx,prompt_index in enumerate(prompt_indexes):
        prompt_index = prompt_index[0]
        if idx == len(prompt_indexes)-1:
            end_index = length
        else:
            end_index = prompt_indexes[idx+1][0]

        new_seg = []
        cur_seg = data.loc[prompt_index:end_index-1,:]
        knowledges = cur_seg["knowledge 0"][1:].values
        # print([(x,len(x.split())) for x in knowledges])
        for r in ranges:
            start_range = r[0]
            end_range = r[1]
            index_list = [x[0]+1 for x in np.argwhere([1 if (len(x.split()) <=end_range and len(x.split()) > start_range)else 0 for x in knowledges])]
            cur_seg.index = list(range(len(cur_seg)))
            cur_seg_sub = cur_seg.loc[index_list,:]
            cur_seg_sub = cur_seg_sub.sort_values([sort_by],ascending=True,ignore_index=False)
            cur_seg_sub["Segments"] = [f"{start_range} < x <= {end_range}"] * len(cur_seg_sub)

            new_seg.append(cur_seg_sub)
        new_seg = pd.concat([cur_seg.loc[0:0,:]]+new_seg, ignore_index=True)
        new_prompt_dfs.append(new_seg)
    
    new_prompt_dfs = pd.concat(new_prompt_dfs, ignore_index=True)
    display(new_prompt_dfs.style.set_properties(**{'white-space': 'pre-wrap','text-align': 'left',}))
    return new_prompt_dfs
def tables_dedup(model,data, sort_by = None,method="greedy", dist_method = "mover", thresh=0.8,return_all_in_cluster=False):
    length = len(data.index)
    prompt_indexes = np.argwhere(data.loc[:,"1st entity"].notna().values).tolist()
    new_prompt_dfs = []
    for idx,prompt_index in tqdm(enumerate(prompt_indexes)):
        prompt_index = prompt_index[0]
        if idx == len(prompt_indexes)-1:
            end_index = length
        else:
            end_index = prompt_indexes[idx+1][0]

        new_seg = []
        cur_seg = data.loc[prompt_index+1:end_index-1,:]
        if sort_by:
            cur_seg = cur_seg.sort_values([sort_by],ascending=True,ignore_index=False)
        cur_seg.index = list(range(len(cur_seg)))
        knowledges = cur_seg["knowledge 0"].values
        if method=="greedy":
            remove_index = set()
            for i in range(len(knowledges)):
                if i not in remove_index:
                    for j in range(i+1,len(knowledges)):
                        score = sentence_score(knowledges[i], [knowledges[j]])
                        # print(knowledges[i], knowledges[j],score)
                        if score > thresh:
                            remove_index.add(j)
            remove_index = [x for x in list(remove_index)]
            # print(remove_index)
            retain_index = [x for x in range(1,len(cur_seg)) if x not in remove_index]
            cur_seg = cur_seg.loc[retain_index,:]
        elif method=="cluster":
            embeddings = torch.tensor(model.encode(knowledges))
            # score_matrix = torch.einsum("ad,bd->ab",embeddings,embeddings)
            # print(score_matrix[0,:])
            # print(torch.matmul(embeddings,embeddings[0].unsqueeze(1)))
            # for i in range(len(knowledges)):
            #     print(knowledges[i], score_matrix[0,i])
            clustering = AgglomerativeClustering(n_clusters = None,distance_threshold=thresh)
            clustered = clustering.fit(embeddings)
            labels = clustered.labels_
            retain_index = []
            for i in range(clustered.n_clusters_):
                indexes = np.where(labels==i)
                # print(f"group {i}: {knowledges[indexes[0]]}")
                if return_all_in_cluster:
                    retain_index.append(indexes[0])
                else:
                    retain_index.append(indexes[0][0])
            if return_all_in_cluster:
                segs = []
                for cluster_ind in retain_index:
                    seg = cur_seg.loc[cluster_ind,:]
                    col_names = list(cur_seg.columns.values)
                    header = [{x:"" for x in col_names}]
                    seg = pd.concat([pd.DataFrame(header), seg],ignore_index=True)
                    segs.append(seg)
                cur_seg = pd.concat(segs, ignore_index=True)
            else:  
                cur_seg = cur_seg.loc[retain_index,:]
            
        elif method == "self-bleu":
            new_knowledge = knowledges
            retain_index = []
            original_index = np.array(list(range(len(new_knowledge))))
            while True:
                cur_scores = np.array(bleu_list_n(new_knowledge,smoothing_function,corpus=True,n_gram=4))
                max_score = max(cur_scores)
                if max_score < 0.5:
                    break
                not_max_idx = cur_scores != max_score
                original_index = original_index[not_max_idx]
                new_knowledge = new_knowledge[not_max_idx]
            retain_index = original_index
            cur_seg = cur_seg.loc[retain_index,:]
        elif method == "cluster_self-bleu":
            embeddings = torch.tensor(model.encode(knowledges))
            clustering = AgglomerativeClustering(n_clusters = None,distance_threshold=thresh)
            clustered = clustering.fit(embeddings)
            labels = clustered.labels_
            retain_index = []
            for i in range(clustered.n_clusters_):
                indexes = np.where(labels==i)
                retain_index.append(indexes[0])
            new_knowledge = knowledges
            retain_index_bleu = []
            original_index = np.array(list(range(len(new_knowledge))))
            while True:
                cur_scores = np.array(bleu_list_n(new_knowledge,smoothing_function,corpus=True,n_gram=4))
                max_score = max(cur_scores)
                if max_score < 0.5:
                    break
                not_max_idx = cur_scores != max_score
                original_index = original_index[not_max_idx]
                new_knowledge = new_knowledge[not_max_idx]
            retain_index_bleu = original_index

            segs = []
            for cluster_ind in retain_index:
                seg = cur_seg.loc[cluster_ind,:]
                deduped = [True if x in retain_index_bleu else False for x in cluster_ind]
                seg["left by bleu"] = deduped
                col_names = list(cur_seg.columns.values)
                header = [{x:"" for x in col_names}]
                seg = pd.concat([pd.DataFrame(header), seg],ignore_index=True)
                segs.append(seg)

            cur_seg = pd.concat(segs, ignore_index=True)
        new_prompt_dfs.append(data.loc[prompt_index:prompt_index,:])
        new_prompt_dfs.append(cur_seg)
    
    new_prompt_dfs = pd.concat(new_prompt_dfs, ignore_index=True)
    # new_prompt_dfs = new_prompt_dfs.drop(columns="Unnamed: 0")
    # display(new_prompt_dfs.style.set_properties(**{'white-space': 'pre-wrap','text-align': 'left',}))
    return new_prompt_dfs

def knowledge_filter_semantic_similarity(model, data):
    for dp in tqdm(data):
        question = dp["question"]
        knowledges1 = dp["generated_knowledges1"]
        knowledges2 = dp["generated_knowledges2"]
        knowledges = knowledges1 + knowledges2
        question_embedding = torch.tensor(model.encode([question]))
        embeddings1 = torch.tensor(model.encode(knowledges))       
        # print(question_embedding.shape, embeddings1.shape)
        score_matrix1 = torch.einsum("ad,bd->ab",question_embedding,embeddings1).detach().cpu().numpy()
        chosen_idx1 = np.argmax(score_matrix1)
        chosen_know1 = knowledges[chosen_idx1]
        dp["picked"] = [chosen_know1]
        dp["ss_scores"] = score_matrix1

def knowledge_filter_ppl(model, tokenizer, data):
    for dp in tqdm(data):
        question = dp["question"]
        knowledges1 = dp["generated_knowledges1"]
        knowledges2 = dp["generated_knowledges2"]

        appended1 = [question + x for x in knowledges1]
        appended2 = [question + x for x in knowledges2]
        input_ids1 = [torch.LongTensor(tokenizer(x,return_tensors="pt")['input_ids']).to('cuda') for x in appended1]
        input_ids2 = [torch.LongTensor(tokenizer(x,return_tensors="pt")['input_ids']).to('cuda') for x in appended2]
        ppls1 = metrics._compute_ppl(input_ids1, model)
        ppls2 = metrics._compute_ppl(input_ids2, model)
        chosen_idx1 = np.argmin(ppls1)
        chosen_idx2 = np.argmin(ppls2)
        chosen_know1 = knowledges1[chosen_idx1]
        chosen_know2 = knowledges2[chosen_idx2]
        dp["picked"] = [chosen_know1,chosen_know2]
        dp["ppl_scores"] = ppls1 + ppls2