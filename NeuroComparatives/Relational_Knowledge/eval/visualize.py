import torch
import json
import pandas as pd
from tqdm import tqdm
from IPython.display import display
from metrics import _compute_ppl,bleu_list_n,file_compute_mover,mover_sentence_score
import numpy as np
stride = 512
# model_dict = {"gpt2-large":model, "bart-large":bart_model,"t5-large":t5_model}
# tokenizer_dict = {"gpt2-large":tokenizer, "bart-large":bart_tokenizer,"t5-large":t5_tokenizer}
def _compute_score(batch_input_ids, model,cross_entropy=True):
    # takes in a list of input ids, output perplexity for each of them
    try:
        max_length = model.config.n_positions
    except:
        max_length = 1024    
    ppls = []
    for b in range(len(batch_input_ids)):
        input_ids = batch_input_ids[b]
        nlls = []
        for i in range(0, input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids = input_ids[:, begin_loc:end_loc].to("cuda")
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            with torch.no_grad():
                if cross_entropy:
                    outputs = model(input_ids, labels=target_ids)
                    # print(outputs)
                    print(outputs[0], trg_len, end_loc)
                    neg_log_likelihood = -1 * outputs[0] 
                else:
                    outputs = model(input_ids)
                    neg_log_likelihood = 0
                    print(outputs[0].shape, outputs[0][0,0,0:10]) # [B, L, Vocab Size]
                    print(input_ids) # [B, L]
                    scores = F.log_softmax(outputs[0], dim=-1)
                    for j in range(input_ids.shape[1]):
                        neg_log_likelihood += scores[0, j, input_ids[0,j]]
                    
                    
            # print(neg_log_likelihood)
            nlls.append(neg_log_likelihood)

        ppl = torch.stack(nlls).sum()
        ppls.append(ppl.item())
    return ppls
def fit_to_pd(data,indexes,models,model_dict=None,tokenizer_dict=None,batch=32,second_entity=None,all_hyps=False):
    formatted = []
    n = len(data[0])
    for i in tqdm(indexes):
        constraint_number = len(data[0][i]["constraints+knowledge"])
        last_prompt = ""
        prompt_concept_pair_knowledges = []
        for j in range(constraint_number):
            # add prompt
            if "prompt" in data[0][i]["constraints+knowledge"][j]:
                prompt = data[0][i]["constraints+knowledge"][j]["prompt"]
                if last_prompt!=prompt:
                    last_prompt = prompt
            
            # add constraints
            # row.append(" \n\n ".join(["Terms: [" + ", ".join(x["terms"]) + "]; max_count:" + str(x["max_count"]) + "; min_count:" + str(x["min_count"]) for x in data[0][i]["constraints+knowledge"][j]["constraints"] if x["polarity"] == 1]))
            constraints = data[0][i]["constraints+knowledge"][j]["constraints"]
            for file in data:
                part_knowledge = file[i]["constraints+knowledge"][j]["knowledge"]
                if "lps" in file[i]["constraints+knowledge"][j]:
                    part_lp = file[i]["constraints+knowledge"][j]["lps"]
                else:
                    part_lp = ["n/a"] * len(part_knowledge)
                if all_hyps:
                    part_lp = file[i]["constraints+knowledge"][j]["scores"]
                    part_knowledge = file[i]["constraints+knowledge"][j]["all_hyps"]

                knowledges = [sent(prompt,constraints,[0] * len(models),x,y) for x, y in zip(part_knowledge,part_lp)]
                prompt_concept_pair_knowledges.extend(knowledges)
        row = []
        row.append(data[0][i]["sentence"])
        if second_entity:
            row.extend([data[0][i]["2nd_ent"],data[0][i]["class"]]+[""]*len(models) + ["","","",""])
        else:
            row.extend(["","","",""])
        formatted.append(row)
        
        for j,model in enumerate(models):
            temp_model = model_dict[model]
            temp_tokenizer = tokenizer_dict[model]
            input_ids = [torch.LongTensor(temp_tokenizer(x.text,return_tensors="pt")['input_ids']).to('cuda') for x in prompt_concept_pair_knowledges]
            cell_ppls = _compute_ppl(input_ids, temp_model,batch)
            for idx, knowledge in enumerate(prompt_concept_pair_knowledges):
                knowledge.ppl[j] = cell_ppls[idx]
        if len(models) == 0:
            for idx, knowledge in enumerate(prompt_concept_pair_knowledges):
                knowledge.ppl.append(idx)
        rank = np.argsort([x.ppl[0] for x in prompt_concept_pair_knowledges])
        sorted_knowledges = list(np.array(prompt_concept_pair_knowledges)[rank])
        for knowledge in sorted_knowledges:
            row = []
            row.append("")
            if second_entity:
                row.append("")
                row.append("")
            row.append(knowledge.prompt)
            row.append(";".join([",".join(x["terms"]) for x in knowledge.constraints if x["polarity"]]))
            for j in range(len(models)):
                row.append(knowledge.ppl[j])
            row.append(knowledge.text)
            row.append(knowledge.lp)
            formatted.append(row)
    return formatted
def show_as_table(data, indexes, display=False,save=False,file="",models=["gpt2-large"], model_dict=None,tokenizer_dict=None,batch=32,second_entity= False,all_hyps=False):
    formatted_data = fit_to_pd(data,indexes,models,model_dict,tokenizer_dict,batch,second_entity,all_hyps = all_hyps)
    pd.set_option('display.max_colwidth', None)
    knowledge_col = [f"knowledge {i}" for i in range(len(data))]
    second_entity_list = None
    lp = None
    classes = None
    hyps = None
    orignal_scores = None
    if second_entity:
        second_entity_list = ["2nd entity"]
        classes = ["class"]
        lp =["length penalty"]
    if all_hyps:
        lp = ["orignal_scores"]
    pdataframe = pd.DataFrame(formatted_data,columns=["1st entity"] + second_entity_list + classes + ["prompt","positive constriants"]+ models +[x for x in knowledge_col] + lp)
    # print(pdataframe)
    # pdataframe = pdataframe.drop(columns="Unnamed: 0")
    if save:
        pdataframe.to_csv(file)
    if display:
        display(pdataframe.style.set_properties(**{'white-space': 'pre-wrap','text-align': 'left',}))

    # return pdataframe

class sent:
    def __init__(self, prompt, constraints, ppl, text,lp):
        self.prompt = prompt
        self.constraints = constraints
        self.ppl = ppl
        self.text = text
        self.lp = lp
        
def rank_grouped(data, indexes):
    ppls = [0] * len(data)
    n = len(data[0])
    base = 0
    data_store = []
    for i in indexes:
        constraint_number = len(data[0][i]["constraints+knowledge"])
        og_sent = data[0][i]["sentence"]
        # print(i, og_sent)
        prompt_concept_pair_knowledges = []
        for j in range(constraint_number):
            # Avg Dis
            prompt = data[0][i]["constraints+knowledge"][j]["prompt"]
            len_prompt = len(prompt.split())
            constraints = data[0][i]["constraints+knowledge"][j]["constraints"]
            constraint = constraints[0]["terms"][0]
            for k, file in enumerate(data):
                knowledges = [sent(prompt,constraints,0,x) for x in file[i]["constraints+knowledge"][j]["knowledge"]]
                prompt_concept_pair_knowledges.extend(knowledges)
            # ppl
        input_ids = [torch.LongTensor(tokenizer(x.text,return_tensors="pt")['input_ids']).to('cuda') for x in prompt_concept_pair_knowledges]
        cell_ppls = _compute_ppl(input_ids, model)
        rank = np.argsort(cell_ppls)
        sorted_knowledges = list(np.array(prompt_concept_pair_knowledges)[rank])
        sorted_ppls = list(np.array(cell_ppls)[rank])
        data_store.append([sent(x.prompt,x.constraints,sorted_ppls[idx],x.text) for idx,x in enumerate(sorted_knowledges)])
        cell_ppl = np.mean(cell_ppls)
        ppls[k] += cell_ppl
        base += 1
            
    ppls = [x/base for x in ppls]
    return data_store
