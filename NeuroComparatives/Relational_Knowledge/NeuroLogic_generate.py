import os
import json
import csv
import math
import argparse
import torch
import logging
import numpy as np
from tqdm import tqdm
import itertools
from pathlib import Path
import time
from os import path
import pickle
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
from constraint_methods import constraints_map
import utils
from lexical_constraints import init_batch
from generate import generate
from data.DataStore import QuaRelDataStore

sys.path.insert(0,'./data/')
logger = logging.getLogger(__name__)
def assign_order(constraint_list_str,order_type="default"):
    if order_type == "default":
        i = 0
        for x in constraint_list_str:
            if x["polarity"]:
                x["order"] = i
                i += 1
            else:
                x["order"] = None
    elif order_type == "first_order":
        i = 0
        for x in constraint_list_str:
            if x["polarity"]:
                x["order"] = i
            else:
                x["order"] = None
    elif order_type == "entity_first_order":
        i = 0
        for x in constraint_list_str:
            if x["polarity"]:
                x["order"] = i
                if i == 1:
                    pass
                else:
                    i += 1
            else:
                x["order"] = None

    return constraint_list_str
    

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, help="pretrained language model to use")
    parser.add_argument("--output_file", type=str, help="output file")
    parser.add_argument("--input_path", type=str, help="initialization of decoding")
    parser.add_argument("--constraint_path", type=str, help="constraints")
    parser.add_argument("--general_constraint_path", type=str, help="to help with general knowledge")
    parser.add_argument("--constraint_method", type=str, help="to help with general knowledge")
    parser.add_argument("--constraint_limit", type=str, help="to help fixing early satisfaction")


    parser.add_argument('--batch_size', type=int, default=256,
                        help="Batch size for decoding.")
    parser.add_argument('--beam_size', type=int, default=10,
                        help="Beam size for searching")
    parser.add_argument('--max_tgt_length', type=int, default=100,
                        help="maximum length of decoded sentences")
    parser.add_argument('--min_tgt_length', type=int, default=0,
                        help="minimum length of decoded sentences")
    parser.add_argument('--ngram_size', type=int, default=3,
                        help='all ngrams can only occur once')
    parser.add_argument('--length_penalty', type=float, default=0.6,
                        help="length penalty for beam search")
    parser.add_argument('--sat_tolerance', type=int, default=2,
                        help="minimum satisfied clause of valid candidates")
    parser.add_argument('--beta', type=float, default=0.,
                        help="reward factor for in progress constraint")
    parser.add_argument('--early_stop', type=float, default=None,
                        help="optional early stop if all constraints are satisfied")
    parser.add_argument('--num_return_sequences', type=int, default=None,
                        help="knowledge prompts generated for each question")
    parser.add_argument('--diversity', action='store_true',
                        help="whether to encourage POS diversity")
    parser.add_argument('--use_general_constraints', action='store_true',
                        help="whether to use the general constraints")
    parser.add_argument('--do_beam_sample', action='store_true',
                        help="whether to use sampling in neurologic")
    parser.add_argument('--beam_temperature', type=float, default=0.6,
                        help="temperature for neurologic sampling")
    
    parser.add_argument('--return_all', action='store_true',
                        help="whether to use different LPs")
    parser.add_argument('--lp_return_size', type=int, default=3,
                        help='each LP returns, overwite num_return_sequences.')
    parser.add_argument('--stochastic', action='store_true',
                        help="whether to use stochastic beam search/gumbel")
    length_penalties = [0.1, 0.4, 0.7, 1.0]
    
    parser.add_argument('--aux_num', type=int, default=2,
                        help='each pos constraint max limit.')
    parser.add_argument('--prefix_as_prompt', action='store_true',
                        help="whether to use prefix as prompt")
    parser.add_argument('--entity_as_prompt', action='store_true',
                        help="whether to use entity as prompt")
    parser.add_argument('--aux_as_prompt', action='store_true',
                        help="whether to use aux verb as prompt")
    parser.add_argument('--adv_as_prompt', action='store_true',
                        help="whether to use adverb as prompt")
    parser.add_argument('--long_aux', action='store_true',
                        help="whether to use the long list of aux verbs")
    parser.add_argument('--long_det', action='store_true',
                        help="whether to use the long list of determiners")
    parser.add_argument('--long_adv', action='store_true',
                        help="whether to use the long list of adverbs")
    # parser.add_argument('--add_no_prompt', action='store_true',
    #                     help="whether to add generation with no prefix")
    parser.add_argument('--divide_aux', action='store_true',
                        help="whether to divide aux into seperate constraints")
    parser.add_argument('--divide_adv', action='store_true',
                        help="whether to divide aux into seperate constraints")

    parser.add_argument("--conditional", type=str, default="",help="conditioanl generation")

    parser.add_argument('--contrast', action='store_true',
                    help="generate contrast knowledges")
    parser.add_argument('--ordered', action='store_true',
                    help="order constraint satisfaction")
    parser.add_argument('--ordered_type', type=str, default="default",
                        help="ordered_type is the order type of constraints")
    parser.add_argument('--special_comparative_constraint', type=str, default=None,
                    help="more or less or 'er' after")
    parser.add_argument('--entity_only', action='store_true',
                    help="generate contrast knowledges")
    parser.add_argument('--look_ahead', type=int, default=None,
                        help='each pos constraint max limit.')
    parser.add_argument("--device", type=str, help="device type to use for compute", default="cuda")
    args = parser.parse_args()
    print(args)
    print(f"output_file: {args.output_file}")

    device = torch.device(args.device)

    # 1. Initialize Tokenizers & Models
    print(f"Decoding with: {args.model_name}")
    if args.model_name == 'gpt2-xl':
        model = utils.load_model(args.model_name,"../models/gpt2-xl/")
        tokenizer = utils.load_tokenizer(args.model_name,"../models/gpt2-xl/")
    elif args.model_name == 'EleutherAI/gpt-j-6B':
        model = AutoModelForCausalLM.from_pretrained(args.model_name, revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.device == 'cuda':
        torch.cuda.empty_cache()
    model.eval()
    model = model.to(device)

    # 2. Token Processing & General Constraints
    print(f'vocab_size: {tokenizer.vocab_size}, POS_size: {len(utils.POS)}')
    digit_num = 10 ** (len(str(tokenizer.vocab_size)) - 1)
    POS_start = (tokenizer.vocab_size // digit_num + 1) * digit_num
    POS = {k: v + POS_start for k, v in utils.POS.items()}
    
    PAD_ID = tokenizer.eos_token_id
    bad_token = [':', "'", '-', '_', '@', 'Ċ', 'Ġ:', 'Ġ(', 'Ġ)', '(', ')', 'Ġ[', 'Ġ]', '[', ']', 'Ġ{', 'Ġ}', '{', '}']
    # bad_punc_ids = [[26], [737], [784], [1399], [1377], [3693], [960], [3228], [851], [2014], [492], [8412], [9], [1911], [8836], [7198], [1776], [30866], [986], [4], [1906], [7441], [5974], [720], [3907], [4211], [10], [20995], [1222], [2474], [47113], [13402], [438], [9816], [7131], [1106], [2637], [11907], [45297], [12237], [4210], [18265], [63], [3926], [15729], [22857], [36165], [4458], [16317], [9313], [32203], [7398], [4083], [11580], [9962], [8183], [43735], [1539], [20543], [23068], [23141], [4407], [33963], [18823], [7061], [19056], [1303], [8348], [15920], [40670], [20262], [12248], [25792], [7874], [22799], [16151], [29865], [30827], [28200], [43054], [4357], [7200], [15089], [9333], [25208], [4032], [93], [17241], [1701], [30864], [3256], [30823], [12906], [59], [4907], [38155], [21169], [29343], [11208], [2602], [1427], [5523], [40345], [22369], [4841], [13700], [31854], [37102], [10185], [16224], [21727], [5855], [17569], [28255], [5641], [27370], [22174], [16764], [30159], [22180], [25224], [26945], [29557], [18566], [33180], [30640], [33623], [6353], [13896], [16142], [22177], [13090], [15166], [16843], [44807], [91], [4707], [43108], [35072], [43666], [26793], [25947], [5746], [12962], [48869], [10235], [41185], [5299], [18962], [9805], [44435], [11593], [8864], [3581], [9129], [26525], [20375], [48585], [23193], [2109], [10221], [37405], [42468], [36310], [38834], [24247], [6557], [12359], [35098], [45367], [18849], [31583], [40493], [40792], [796], [35540], [38508], [17478], [45160], [16725], [834], [46444], [19570], [4808], [62], [2162], [43634], [40549], [40283], [1174], [50184], [32501], [25125], [9101], [28358], [17020], [32941], [38857], [14373], [4557], [24328], [4181], [49129], [34617], [27754], [25780], [22940], [5196], [31660], [23513], [11496], [31817], [32014], [27896], [42720], [32790], [21689], [21410], [21096], [33283], [18161], [44926], [41468], [17730], [30143], [1343], [27193], [15166, 4210], [45035], [40623], [9525], [38430], [41832], [44359], [49011], [42783], [13697], [30644], [20317], [6], [41052], [0], [28], [16791], [14988], [930], [21747], [24172], [42943], [20740], [46268], [5595], [49946], [15327], [1378], [14004], [25608], [36853], [8728], [16078], [13531], [6739], [35944], [22074], [12195], [34507], [34400], [26171], [4008], [21356], [29001], [14808], [25151], [17202], [3548], [43095], [15351], [45990], [33778], [28134], [25748], [40948], [30201], [26825], [22020], [29], [19571], [2634], [17414], [41200], [14064], [15437], [48529], [39658], [21912], [15168], [22161], [2430], [1], [8351], [14], [13018], [553], [828], [1600], [532], [526], [30], [0], [8], [7], [314], [892], [345], [921], [679], [1375], [1119], [339], [339, 13], [484], [484, 13], [673], [673, 13], [616], [616, 13], [1231], [1231, 13], [1022], [1022, 13], [881], [881, 13], [716], [716, 13], [645], [645, 13], [4249], [4249, 13], [407], [407, 13], [355], [355, 13], [428], [428, 13], [612], [612, 13], [994], [994, 13], [976], [976, 13], [1178], [1178, 13], [2092], [2092, 13], [262, 1708], [262, 1708, 13], [416, 783], [416, 783, 13], [656], [656, 13]]
    bad_punc_smallnoprepodiscourse_ids = [[26], [737], [784], [1399], [1377], [3693], [960], [3228], [851], [2014], [492], [8412], [9], [1911], [8836], [7198], [1776], [30866], [986], [4], [1906], [7441], [5974], [720], [3907], [4211], [10], [20995], [1222], [2474], [47113], [13402], [438], [9816], [7131], [1106], [2637], [11907], [45297], [12237], [4210], [18265], [63], [3926], [15729], [22857], [36165], [4458], [16317], [9313], [32203], [7398], [4083], [11580], [9962], [8183], [43735], [1539], [20543], [23068], [23141], [4407], [33963], [18823], [7061], [19056], [1303], [8348], [15920], [40670], [20262], [12248], [25792], [7874], [22799], [16151], [29865], [30827], [28200], [43054], [4357], [7200], [15089], [9333], [25208], [4032], [93], [17241], [1701], [30864], [3256], [30823], [12906], [59], [4907], [38155], [21169], [29343], [11208], [2602], [1427], [5523], [40345], [22369], [4841], [13700], [31854], [37102], [10185], [16224], [21727], [5855], [17569], [28255], [5641], [27370], [22174], [16764], [30159], [22180], [25224], [26945], [29557], [18566], [33180], [30640], [33623], [6353], [13896], [16142], [22177], [13090], [15166], [16843], [44807], [91], [4707], [43108], [35072], [43666], [26793], [25947], [5746], [12962], [48869], [10235], [41185], [5299], [18962], [9805], [44435], [11593], [8864], [3581], [9129], [26525], [20375], [48585], [23193], [2109], [10221], [37405], [42468], [36310], [38834], [24247], [6557], [12359], [35098], [45367], [18849], [31583], [40493], [40792], [796], [35540], [38508], [17478], [45160], [16725], [834], [46444], [19570], [4808], [62], [2162], [43634], [40549], [40283], [1174], [50184], [32501], [25125], [9101], [28358], [17020], [32941], [38857], [14373], [4557], [24328], [4181], [49129], [34617], [27754], [25780], [22940], [5196], [31660], [23513], [11496], [31817], [32014], [27896], [42720], [32790], [21689], [21410], [21096], [33283], [18161], [44926], [41468], [17730], [30143], [1343], [27193], [15166, 4210], [45035], [40623], [9525], [38430], [41832], [44359], [49011], [42783], [13697], [30644], [20317], [6], [41052], [0], [28], [16791], [14988], [930], [21747], [24172], [42943], [20740], [46268], [5595], [49946], [15327], [1378], [14004], [25608], [36853], [8728], [16078], [13531], [6739], [35944], [22074], [12195], [34507], [34400], [26171], [4008], [21356], [29001], [14808], [25151], [17202], [3548], [43095], [15351], [45990], [33778], [28134], [25748], [40948], [30201], [26825], [22020], [29], [19571], [2634], [17414], [41200], [14064], [15437], [48529], [39658], [21912], [15168], [22161], [2430], [1], [8351], [14], [13018], [553], [828], [1600], [532], [526], [30], [0], [8], [7], [314], [892], [345], [921], [679], [339], [339, 13], [1119], [484], [484, 13], [673], [673, 13], [1375], [616], [616, 13], [775], [356], [1231], [1231, 13], [1022], [1022, 13], [881], [881, 13], [2035], [2035, 13], [6159], [6159, 13], [290], [290, 13], [618], [618, 13], [981], [981, 13], [3584], [3584, 13], [716], [716, 13], [645], [645, 13], [4249], [4249, 13], [407], [407, 13], [355], [355, 13], [780], [780, 13], [1201], [1201, 13], [3584], [3584, 13], [3443], [3443, 13], [2158], [2158, 13], [4361], [4361, 13], [780], [780, 13], [25578], [25578, 13], [50002], [50002, 13], [19018], [19018, 13], [33695], [33695, 13], [46596], [46596, 13], [12891, 11813], [12891, 11813, 13], [19032], [19032, 13], [9472], [9472, 13], [15066], [15066, 13], [428], [428, 13], [612], [612, 13], [994], [994, 13], [976], [976, 13], [1178], [1178, 13], [2092], [2092, 13], [262, 1708], [262, 1708, 13], [416, 783], [416, 783, 13], [656], [656, 13], [621], [621, 13]]
    period_id = [tokenizer.convert_tokens_to_ids('.')]
    
    if 'gpt2' in args.model_name:
        period_id.append(tokenizer.convert_tokens_to_ids('Ġ.'))

        bad_words_ids = [tokenizer.convert_tokens_to_ids([t]) for t in bad_token]
        bad_words_ids = bad_words_ids + bad_punc_smallnoprepodiscourse_ids + [[11]] + [[290]] #+ [[13]] # 13 is period id, 11 is comma, 290 is " and"
    else:
        gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
        bad_words_ids = [gpt2_tokenizer.convert_tokens_to_ids([t]) for t in bad_token]
        bad_words_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(gpt2_tokenizer.decode(t))) for t in bad_words_ids]
        
        bad_ids = bad_punc_smallnoprepodiscourse_ids + [[11]] + [[290]]
        bad_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(gpt2_tokenizer.decode(i))) for i in bad_ids]
        bad_words_ids = bad_words_ids + bad_ids

        if 'llama' in args.model_name:
            period_id += [tokenizer.convert_tokens_to_ids('▁.'), tokenizer.bos_token_id]
        elif 'bart' in args.model_name:
            period_id += [tokenizer.convert_tokens_to_ids('Ġ.')]
        else:
            raise NotImplementedError
    
    eos_ids = [tokenizer.eos_token_id] + period_id

    # 0. Load Data
    # data = LionsDataStore(args.input_path,header=0,header_names=True).prepare_for_slurm()
    data = [] 
    with open(args.input_path) as f:
        for line in f:
            pair = json.loads(line)
            data.append(pair)
    print(f"data length: {len(data)}")
    flat_data = []
    print("processing data")
    for d in tqdm(data):
        for ins in d["prompt_and_constraint"]:
            constraint_list_str = ins["cur_constraints_str"]
            if args.ordered:
                constraint_list_str = constraint_list_str[::-1]
                constraint_list_str = assign_order(constraint_list_str, args.ordered_type)
            else:
                constraint_list_str = assign_order(constraint_list_str, args.ordered_type)
            if args.special_comparative_constraint:
                max_order = max([x["order"] for x in constraint_list_str if x["order"] != None])
                constraint_list_str.append( {'terms':["er","ier"], 'polarity': 1, 'max_count': 1, 'min_count': 1, 'type': args.special_comparative_constraint,'order' : max_order+1})
            constraints_list = utils.tokenize_constraints(tokenizer, POS, [constraint_list_str])
            constraints = init_batch(raw_constraints=constraints_list,
                                         beam_size=args.beam_size,
                                         eos_id=eos_ids,ordered=args.ordered)
            # if 'gpt2' in args.model_name:
            input_id = tokenizer.encode(ins["input_id"], return_tensors="pt")
            # else:
            #     input_id = tokenizer.encode(ins["input_id"], return_tensors="pt", add_special_tokens=False)
            #     input_id = torch.concat([torch.tensor([[tokenizer.bos_token_id]]),input_id],axis=1)
            flat_data.append({"input_id":input_id,
                              "constraint":constraints,
                              'cur_constraints_str':constraint_list_str,
                              'ent_class':d["ent_class"],
                              'entity': d["entity"],
                              'second_entity': d["second_entity"]})
    print(flat_data[0])
    general_constraints = []
    original_jsonl_objs = []
    with open(args.general_constraint_path) as f:
        lines = f.read()
        obj = json.loads(lines)        
        original_jsonl_objs.append(obj)
        current_constraints = []
        for cons in obj['constraints']['clauses']:
            current_constraints.append(cons)
        general_constraints.append(current_constraints)
    
    constraints_lists = [[[]] for x in range(len(data))]
    num_concept = len(data)
    
    print(f"num_concept: {num_concept}")
    print(f"constraint_path[0]: {constraints_lists[0]}")

    logs = []
    second_entity = None
    prompts = []
    times = []
    print(f"\nStart Generation:")
    buf = defaultdict(list)
    with tqdm(total=len(flat_data)/args.batch_size) as pbar:
        for i in range(0,len(flat_data),args.batch_size):
            batch = flat_data[i:i+args.batch_size]
            input_ids = [x["input_id"] for x in batch]
            batch_input_ids = torch.cat(input_ids,dim=0).to(device)
            constraints = [x["constraint"] for x in batch]
            batch_constraints = list(itertools.chain(*constraints))
            start = time.time()
            outputs = generate(self=model,
                            input_ids=batch_input_ids,
                            attention_mask=None,
                            pad_token_id=PAD_ID,
                            bad_words_ids=bad_words_ids,
                            min_length=args.min_tgt_length,
                            max_length=args.max_tgt_length + batch_input_ids.shape[1],
                            num_beams=args.beam_size,
                            no_repeat_ngram_size=args.ngram_size,
                            constraints=batch_constraints,
                            length_penalty=args.length_penalty, # this penalty only affects when the beam is finished.
                            sat_tolerance=args.sat_tolerance,
                            num_sequences_per_return=args.num_return_sequences,
                            do_beam_sample=args.do_beam_sample,
                            beam_temperature=args.beam_temperature,
                            diversity=args.diversity,
                            return_all=args.return_all,
                            length_penalties=length_penalties, 
                            lp_return_size=args.lp_return_size,
                            stochastic = args.stochastic,
                            constraint_limit=args.constraint_limit,
                            look_ahead=args.look_ahead)
            end = time.time()
            elapsed = (end -start)/60
            times.append(elapsed)
            for j in range(len(input_ids)):
                ent_class = batch[j]["ent_class"]
                entity = batch[j]["entity"]
                second_entity = batch[j]["second_entity"]
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
                    generated_knowledge = decoded.split(tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id))[0].rstrip()
                    generated_knowledge = generated_knowledge.strip().replace(tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id), '')
                    if 'llama' in args.model_name:
                        generated_knowledge = generated_knowledge.replace(tokenizer.convert_ids_to_tokens(tokenizer.bos_token_id), '').strip()
                    decoded_output_sequences.append(generated_knowledge)
                if 'llama' in args.model_name:
                    eos_token = tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id)
                    bos_token = tokenizer.convert_ids_to_tokens(tokenizer.bos_token_id)
                    all_hyps = [tokenizer.decode(x[1]).split(eos_token)[0].rstrip().strip().replace(eos_token, '').replace(bos_token, '').strip() for x in outputs["all_hyps"][j]]
                else:
                    all_hyps = [tokenizer.decode(x[1]).split(tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id))[0].rstrip() for x in outputs["all_hyps"][j]]
                #print(decoded_output_sequences)
                #print([(x[2], tokenizer.decode(x[1])) for x in outputs["all_hyps"][j]])
                # for idx, a in enumerate(all_hyps):
                #     if a == "Compared to rice, wheat flour is typically used to make pizza.":
                #         print(outputs["all_hyps"][j][idx])
                out = {'prompt': tokenizer.decode(input_id[0]),'constraints':cur_constraints_str_clause, 'knowledge': decoded_output_sequences,"lps":lps,"all_hyps":all_hyps,"scores":scores}
                buf[(ent_class,entity,second_entity)].append(out)
            pbar.update(1)
    fout = Path(args.output_file).open("w", encoding="utf-8")
    for title in buf:
        ent_class = title[0]
        entity = title[1]
        second_entity = title[2]
        output_sequences = buf[title]
        log = json.dumps({'class':ent_class,'sentence': entity,'2nd_ent': second_entity,'constraints+knowledge':output_sequences})
        # print(log)
        logs.append(log)
        fout.write(f'{log}\n')
        fout.flush()
    print(f"average time per iter: {sum(times)/len(times)}")
    # with open(f'./Generated/wikidata/prompts.csv', "w") as f:
    #     csvwriter = csv.writer(f,delimiter="\t")
    #     for d in prompts:
    #         for p in d:
    #             csvwriter.writerow([p])
    #         csvwriter.writerow([])
if __name__ == "__main__":
    main()
