from tqdm import tqdm
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import nltk
import sys
from collections import defaultdict
import torch

sys.path.insert(0,'../..')
sys.path.insert(0,'../../emnlp19-moverscore')
# from moverscore_v2 import get_idf_dict, word_mover_score 


#### Average Distance to 1st Constraint
def file_measure_distance_1stCosntraint(files, indexes):
    data = load_files(files)
    total_distance = [0] * len(data)
    n = len(data[0])
    base = 0
    for i in indexes:
        constraint_number = len(data[0][i]["constraints+knowledge"])
        last_prompt = ""
        og_sent = data[0][i]["sentence"]
        for j in range(constraint_number):
            row = []
                
            # add prompt
            prompt = data[0][i]["constraints+knowledge"][j]["prompt"]
            len_prompt = len(prompt.split())
            constraint = data[0][i]["constraints+knowledge"][j]["constraints"][0]["terms"][0]
            # print(prompt, constraint)
            for k, file in enumerate(data):
                dis = 0
                for g in file[i]["constraints+knowledge"][j]["knowledge"]:
                    for z,w in enumerate(g.split()):
                        if constraint in w:
                            dis += z-len_prompt+1
                            break
                    # print(g,dis)
                total_distance[k] += dis/len(file[i]["constraints+knowledge"][j]["knowledge"])
                base += 1
    total_distance = [x/base for x in total_distance]
    return total_distance
                    

#### Perplexity
stride = 512
def _compute_ppl(batch_input_ids, model,batch=None):
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
                outputs = model(input_ids, labels=target_ids)
                # print(outputs[0])
                neg_log_likelihood = outputs[0] * trg_len
            # print(neg_log_likelihood)
            nlls.append(neg_log_likelihood)

        ppl = torch.exp2(torch.stack(nlls).sum() / end_loc)
        ppls.append(ppl.item())
    return ppls
def file_compute_ppl(files, indexes, model, tokenizer):
    data = load_files(files)
    total_distance = [0] * len(data)
    n = len(data[0])
    base = 0
    ppls = [0] * len(files)
    for i in tqdm(indexes):
        constraint_number = len(data[0][i]["constraints+knowledge"])
        for j in range(constraint_number):
            for k, file in enumerate(data):
                knowledges = file[i]["constraints+knowledge"][j]["knowledge"]
                input_ids = [torch.LongTensor(tokenizer(x,return_tensors="pt")['input_ids']).to('cuda') for x in knowledges]
                cell_ppls = _compute_ppl(input_ids, model)
                cell_ppl = np.mean(cell_ppls)
                ppls[k] += cell_ppl
            base += 1
    ppls = [x/base for x in ppls]
    return ppls



#### Self-BLEU
# ref https://github.com/ari-holtzman/degen/blob/master/metrics/self_bleu.py
# https://github.com/geek-ai/Texygen/blob/master/utils/metrics/SelfBleu.py
def bleu_i(weights, all_sentences, smoothing_function, i):
    # noinspection PyTypeChecker
    return sentence_bleu(
        references=all_sentences[:i] + all_sentences[i + 1:],
        hypothesis=all_sentences[i],
        weights=weights,
        smoothing_function=smoothing_function)
def bleu_list(weights, all_sentences, smoothing_function):
    n = len(all_sentences)
    bleu_scores = []
    for i in tqdm(range(n)):
        bleu = bleu_i(weights, all_sentences, smoothing_function, i)
        bleu_scores.append(bleu)
    return bleu_scores

# we may use this because if we compare each sentence with the rest of the sentences as reference, it introduces problems for our use case (see examples2 vs exapmles1)
# hence we compare each sentence with only one reference
def bleu_matrix(weights, all_sentences, smoothing_function):
    n = len(all_sentences)
    bleu_scores = []
    for i in range(n):
        for j in range(n):
            bleu = sentence_bleu(references=[all_sentences[j]],hypothesis=all_sentences[i],weights=weights,smoothing_function=smoothing_function)
            bleu_scores.append(bleu)
    return bleu_scores
def bleu_list_n(all_sentences, smoothing_function,corpus = False, n_gram=4):
    all_sentences = [nltk.word_tokenize(x) for x in all_sentences]
    if n_gram == 1:
        weights = (1.0, 0, 0, 0)
    elif n_gram == 2:
        weights = (0.5, 0.5, 0, 0)
    elif n_gram == 3:
        weights = (1.0 / 3, 1.0 / 3, 1.0 / 3, 0)
    elif n_gram == 4:
        weights = (0.25, 0.25, 0.25, 0.25)
    elif n_gram == 5:
        weights = (0.2, 0.2, 0.2, 0.2, 0.2)
    else:
        raise ValueError
    if corpus:
        bleus = bleu_list(weights, all_sentences, smoothing_function)
    else:
        bleus = bleu_matrix(weights, all_sentences, smoothing_function)
    return bleus
def file_compute_bleu(files, indexes, smoothing_function,corpus=False,n_gram=4):
    data = load_files(files)
    n = len(data[0])
    base = 0
    bleus = [0] * len(files)
    for i in tqdm(indexes):
        constraint_number = len(data[0][i]["constraints+knowledge"])
        for j in range(constraint_number):
            for k, file in enumerate(data):
                knowledges = file[i]["constraints+knowledge"][j]["knowledge"]
                cell_bleu = bleu_list_n(knowledges, smoothing_function,corpus=corpus,n_gram=n_gram)
                bleus[k] += cell_bleu
            base += 1
    bleus = [x/base for x in bleus]
    return bleus


#### Mover-Score
def mover_sentence_score(hypothesis: str, references, trace=0):
    
    idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)
    
    hypothesis = [hypothesis] * len(references)
    
    sentence_score = 0 

    scores = word_mover_score(references, hypothesis, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=False)
    
    sentence_score = np.mean(scores)
    
    if trace > 0:
        print(hypothesis, references, sentence_score)
            
    return sentence_score
def file_compute_mover(files, indexes):
    data = load_files(files)
    n = len(data[0])
    base = 0
    movers = [0] * len(files)
    for i in tqdm(indexes):
        constraint_number = len(data[0][i]["constraints+knowledge"])
        for j in range(constraint_number):
            for k, file in enumerate(data):
                knowledges = file[i]["constraints+knowledge"][j]["knowledge"]
                scores = [sentence_score(x, knowledges[:idx] + knowledges[idx+1:]) for idx,x in enumerate(knowledges)]
                movers[k] += np.mean(scores)
            base += 1
    movers = [x/base for x in movers]
    return movers

# Sentence T5 Score
def compute_sentence_score(sent_model,knowledges):
    embeddings = torch.tensor(sent_model.encode(knowledges))
    return embeddings