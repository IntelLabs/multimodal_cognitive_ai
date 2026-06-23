# This is to generate counter captions by:
# 1. From one original caption, generate a set of candidate counter captions by masking out each noun of original one and use MLM to fill mask.
# 2. Among candidate counter captions, those have similarity score with original caption between lb_score and ub_score will be filtered out (to avoid counter caption and original caption are identical)
# 3. Among the remaining candidate counter captions, the one with highest perplexity score will be returned.

import os
import os.path as osp
import argparse
from tqdm import tqdm
import mmap

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util

from transformers import pipeline
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

def get_noun_position(sent):
    text = word_tokenize(sent)
    tags = nltk.pos_tag(text)
    indices = []
    for i in range(len(tags)):
        if tags[i][1] == 'NN' or tags[i][1] == 'NNS':
            indices.append(i)
    return text, indices

def is_word_at_index_noun(sent, index):
    text = word_tokenize(sent)
    tags = nltk.pos_tag(text)
    try:
        if tags[index][1] in ['NN', 'NNS', 'NNP']:
            return True
        else:
            return False
    except: #to ignore cases where mask is replaced by non-word like symbols '.', ',', ':' and as a result, index input here is not valid anymore. 
        print('Exception:', sent)
        return False

# remove candidate sentences whose mask is filled by a non-complete word (i.e., symbols like , . : !) 
# by comparing len of tokens of candidate sentences vs len of tokens of origin sentences
def remove_from_list_sent_has_mask_filled_by_non_complete_words(sent_list, tokenized_origin_text):
    refined_list = [sent for sent in sent_list if len(word_tokenize(sent)) == len(tokenized_origin_text)]
    return refined_list

def remove_from_list_sentences_has_not_noun_at_index(sent_list, index):
    refined_list = [sent for sent in sent_list if is_word_at_index_noun(sent, index)]
    return refined_list

def counter_sent_gen(text, indices, unmasker, top_k = 10):
    if len(indices) == 0:
        return None
    counter_sents = []
    for i in indices:
        tmp_text = text[:]
        tmp_text[i] = unmasker.tokenizer.mask_token
        unmask_sents = unmasker(" ".join(tmp_text), top_k = top_k)
        counter_sents_of_index_i = [ unmask_sent['sequence'] for unmask_sent in unmask_sents]
        counter_sents_of_index_i = remove_from_list_sent_has_mask_filled_by_non_complete_words(counter_sents_of_index_i, text)
        counter_sents_of_index_i = remove_from_list_sentences_has_not_noun_at_index(counter_sents_of_index_i, i)
        counter_sents = counter_sents + counter_sents_of_index_i        
    return counter_sents

def compute_perplexity(sent, tokenizer, auto_regressive_model):
    encodings = tokenizer(sent, return_tensors="pt")
    max_length = auto_regressive_model.config.n_positions
    stride = 1
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(auto_regressive_model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = auto_regressive_model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over input tokens.
            # Multiply it with trg_len to get the summation instead of average.
            # We will take average over all the tokens to get the true average
            # in the last step of this example.
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl
    
def get_most_appropriate_counter_sent_based_on_perplexity(sent, counter_sents, similarity_model, tokenizer_for_perplexity, model_for_perplexity, up_sim_score=0.91, lb_sim_score=0.2):
    #Compute embedding for both lists
    most_appropriate_sent = None
    if counter_sents == None or len(counter_sents) == 0:
        return None
    best_perplexity_score = float('inf')

    original_sent_embedding = similarity_model.encode(sent, convert_to_tensor=True)
    for counter_sent in counter_sents:
        counter_sent_embedding = similarity_model.encode(counter_sent, convert_to_tensor=True)
        similar_score = util.pytorch_cos_sim(original_sent_embedding, counter_sent_embedding).detach().cpu().numpy()[0][0]
        #print(counter_sent, similar_score)
        if lb_sim_score < similar_score <= up_sim_score:
            perplexity_score = compute_perplexity(counter_sent, tokenizer_for_perplexity, model_for_perplexity)
            if perplexity_score < best_perplexity_score:
                best_perplexity_score = perplexity_score
                most_appropriate_sent = counter_sent
                
    return most_appropriate_sent

def get_most_appropriate_counter_sent(sent, unmasker, similarity_model, up_sim_score, lb_sim_score, tokenizer_for_perplexity, model_for_perplexity):
    text, indices = get_noun_position(sent)
    counter_sents = counter_sent_gen(text, indices, unmasker)
    counter_sent = get_most_appropriate_counter_sent_based_on_perplexity(sent, counter_sents, similarity_model, tokenizer_for_perplexity, model_for_perplexity, up_sim_score=up_sim_score, lb_sim_score=lb_sim_score)
    return counter_sent
    

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Counter Prompt Generation Using MLM and Perplexity',
        usage='counter_caption_generation_perplexity.py [<args>] [-h | --help]'
    )

    parser.add_argument('-p', '--prompt_file', default="input/prompts_sample.txt", type=str, help='File including prompts')
    parser.add_argument('-o', '--output_folder', default="output/", type=str, help='Path to output folder')
    # RE = Replacement Edit, RF = Refinement Edit
    parser.add_argument('-l', '--lb_score', default=0.8, type=float, help = 'float number as lower bound score')
    parser.add_argument('-u', '--ub_score', default=0.91, type=float, help = 'float number as upper bound score')
    return parser.parse_args(args)

def create_output_folder(path):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(path)
       print(f'The new directory {path} is created!')
    
def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def printConfig(args):
    print(f'Prompt file: {args.prompt_file}')
    print(f'Output path: {args.output_folder}')
    print(f'Lower bound of similarity score: {args.lb_score}')
    print(f'Upper bound of similarity score: {args.ub_score}')
    
def main(args):
    #create output folder if not exists
    print('Checking input and output...')
    create_output_folder(args.output_folder)

    if not osp.exists(args.prompt_file):
        raise ValueError('Prompt file does not exists!')
        return
    output_filename = args.prompt_file.split('/')[-1]
    
    printConfig(args)
    
    #initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # fill-mask pipelines
    unmasker_roberta = pipeline('fill-mask', model='roberta-base', device=0)
    #unmasker_albert = pipeline('fill-mask', model='albert-base-v1')
    similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

    # model to compute perplexity
    model_id_for_perplexity = "gpt2-large"
    model_for_perplexity = GPT2LMHeadModel.from_pretrained(model_id_for_perplexity).to(device)
    tokenizer_for_perplexity = GPT2TokenizerFast.from_pretrained(model_id_for_perplexity)
    
    #reading prompts
    with open(args.prompt_file) as infile:
        with open(osp.join(args.output_folder, output_filename), 'w') as outfile:
            for line in tqdm(infile, total=get_num_lines(args.prompt_file)):
                if '\n' == line[-1]:
                    line = line[0:-1]
                index, origin = line.split("\t")
                counter_sent = get_most_appropriate_counter_sent(origin, unmasker_roberta, similarity_model, args.ub_score, args.lb_score, tokenizer_for_perplexity, model_for_perplexity)
                if counter_sent is not None:
                    outfile.write(index + '\t' + origin  + '\t' + counter_sent)
                    outfile.write('\n')
                else:
                    print('Cannot generate for: index:', index, '\t sent: ', origin)
            
if __name__ == '__main__':
    main(parse_args())
