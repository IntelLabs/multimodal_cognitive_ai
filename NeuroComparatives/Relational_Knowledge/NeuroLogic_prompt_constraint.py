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
from transformers import AutoTokenizer, AutoModelWithLMHead
import sys
from constraint_methods import constraints_map
sys.path.insert(0,'..')
from zero_shot import utils
from lexical_constraints import init_batch
from generate import generate
from data.DataStore import QuaRelDataStore
sys.path.insert(0,'./data/')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, help="pretrained language model to use")
    parser.add_argument("--output_file", type=str, help="output file")
    parser.add_argument("--input_path", type=str, help="initialization of decoding")
    parser.add_argument("--general_constraint_path", type=str, help="to help with general knowledge")
    parser.add_argument("--constraint_method", type=str, help="to help with general knowledge")
    parser.add_argument("--constraint_limit", type=str, help="to help fixing early satisfaction")

    parser.add_argument('--beam_size', type=int, default=10,
                        help="Beam size for searching")
    parser.add_argument('--use_general_constraints', action='store_true',
                        help="whether to use the general constraints")
    
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
    parser.add_argument('--entity_only', action='store_true',
                    help="generate contrast knowledges")
    parser.add_argument('--look_ahead', type=int, default=None,
                        help='each pos constraint max limit.')
    args = parser.parse_args()
    print(args)
    print(f"output_file: {args.output_file}")
    
    # 0. Load Data
    data = [] 
    with open(args.input_path) as f:
        for line in f:
            l=line.split('\t')
            l[-1] = l[-1].strip()
            data.append(l)
    # print(data)

    # 1. Initialize Tokenizers & Models
    print(f"Decoding with: {args.model_name}")
    model = None
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 2. Token Processing & General Constraints
    print(f'vocab_size: {tokenizer.vocab_size}, POS_size: {len(utils.POS)}')
    digit_num = 10 ** (len(str(tokenizer.vocab_size)) - 1)
    POS_start = (tokenizer.vocab_size // digit_num + 1) * digit_num
    POS = {k: v + POS_start for k, v in utils.POS.items()}
    period_id = [tokenizer.convert_tokens_to_ids('.')]
    period_id.append(tokenizer.convert_tokens_to_ids('Ġ.'))
    eos_ids = [tokenizer.eos_token_id] + period_id
    PAD_ID = tokenizer.convert_tokens_to_ids('<pad>')
    bad_token = [':', "'", '-', '_', '@', 'Ċ', 'Ġ:', 'Ġ(', 'Ġ)', '(', ')', 'Ġ[', 'Ġ]', '[', ']', 'Ġ{', 'Ġ}', '{', '}']
    bad_words_ids = [tokenizer.convert_tokens_to_ids([t]) for t in bad_token]
    bad_punc_ids = [[26], [737], [784], [1399], [1377], [3693], [960], [3228], [851], [2014], [492], [8412], [9], [1911], [8836], [7198], [1776], [30866], [986], [4], [1906], [7441], [5974], [720], [3907], [4211], [10], [20995], [1222], [2474], [47113], [13402], [438], [9816], [7131], [1106], [2637], [11907], [45297], [12237], [4210], [18265], [63], [3926], [15729], [22857], [36165], [4458], [16317], [9313], [32203], [7398], [4083], [11580], [9962], [8183], [43735], [1539], [20543], [23068], [23141], [4407], [33963], [18823], [7061], [19056], [1303], [8348], [15920], [40670], [20262], [12248], [25792], [7874], [22799], [16151], [29865], [30827], [28200], [43054], [4357], [7200], [15089], [9333], [25208], [4032], [93], [17241], [1701], [30864], [3256], [30823], [12906], [59], [4907], [38155], [21169], [29343], [11208], [2602], [1427], [5523], [40345], [22369], [4841], [13700], [31854], [37102], [10185], [16224], [21727], [5855], [17569], [28255], [5641], [27370], [22174], [16764], [30159], [22180], [25224], [26945], [29557], [18566], [33180], [30640], [33623], [6353], [13896], [16142], [22177], [13090], [15166], [16843], [44807], [91], [4707], [43108], [35072], [43666], [26793], [25947], [5746], [12962], [48869], [10235], [41185], [5299], [18962], [9805], [44435], [11593], [8864], [3581], [9129], [26525], [20375], [48585], [23193], [2109], [10221], [37405], [42468], [36310], [38834], [24247], [6557], [12359], [35098], [45367], [18849], [31583], [40493], [40792], [796], [35540], [38508], [17478], [45160], [16725], [834], [46444], [19570], [4808], [62], [2162], [43634], [40549], [40283], [1174], [50184], [32501], [25125], [9101], [28358], [17020], [32941], [38857], [14373], [4557], [24328], [4181], [49129], [34617], [27754], [25780], [22940], [5196], [31660], [23513], [11496], [31817], [32014], [27896], [42720], [32790], [21689], [21410], [21096], [33283], [18161], [44926], [41468], [17730], [30143], [1343], [27193], [15166, 4210], [45035], [40623], [9525], [38430], [41832], [44359], [49011], [42783], [13697], [30644], [20317], [6], [41052], [0], [28], [16791], [14988], [930], [21747], [24172], [42943], [20740], [46268], [5595], [49946], [15327], [1378], [14004], [25608], [36853], [8728], [16078], [13531], [6739], [35944], [22074], [12195], [34507], [34400], [26171], [4008], [21356], [29001], [14808], [25151], [17202], [3548], [43095], [15351], [45990], [33778], [28134], [25748], [40948], [30201], [26825], [22020], [29], [19571], [2634], [17414], [41200], [14064], [15437], [48529], [39658], [21912], [15168], [22161], [2430], [1], [8351], [14], [13018], [553], [828], [1600], [532], [526], [30], [0], [8], [7], [314], [892], [345], [921], [679], [1375], [1119], [339], [339, 13], [484], [484, 13], [673], [673, 13], [616], [616, 13], [1231], [1231, 13], [1022], [1022, 13], [881], [881, 13], [716], [716, 13], [645], [645, 13], [4249], [4249, 13], [407], [407, 13], [355], [355, 13], [428], [428, 13], [612], [612, 13], [994], [994, 13], [976], [976, 13], [1178], [1178, 13], [2092], [2092, 13], [262, 1708], [262, 1708, 13], [416, 783], [416, 783, 13], [656], [656, 13]]
    bad_words_ids = bad_words_ids + bad_punc_ids + [[11]] #+ [[13]] # 13 is period id, 11 is comma
    
    general_constraints = []
    original_jsonl_objs = []
    with open(args.general_constraint_path) as f:
        lines = f.read()
        # print(lines)
        # escape_lines = json.dumps(lines)
        # print(escape_lines)
        obj = json.loads(lines)        
        original_jsonl_objs.append(obj)
        current_constraints = []
        for cons in obj['constraints']['clauses']:
            current_constraints.append(cons)
        general_constraints.append(current_constraints)
    
    constraints_lists = [[[]] for x in range(len(data))]
    print(f"constraint_path[0]: {constraints_lists[0]}")

    n = len(data) 
    constraint_method = constraints_map[args.constraint_method]
    second_entity = None
    potential_constraint_types = 1

    print(f"Prepare for Generation")
    prepared_data = []
    for i in tqdm(range(n)):
        ent_class = data[i][0]
        entity = data[i][1]
        ent_set = []
        if args.contrast:
            second_entity = data[i][2]
            if len(data[i]) >3:
                ent_set = data[i][2:]
        constraint_sets = []
        print(f"\nentity:{entity},second_entity:{second_entity}, ent_set: {ent_set}")
        for constraint_type_idx in range(potential_constraint_types):
            constraint = constraints_lists[constraint_type_idx]
            if len(data[i]) >3:
                constraint = [ent_set]
            # print(constraint)
            input_ids, constraints,constraint_list_str = constraint_method(args.beam_size,ent_class,entity,tokenizer,eos_ids,POS,args.use_general_constraints,general_constraints,constraint,aux_num=args.aux_num,
            prefix_as_prompt = args.prefix_as_prompt, entity_as_prompt = args.entity_as_prompt,aux_as_prompt = args.aux_as_prompt,adv_as_prompt = args.adv_as_prompt,look_ahead=args.look_ahead,
            long_aux=args.long_aux,long_det=args.long_det,long_adv=args.long_adv, divide_aux=args.divide_aux,divide_adv=args.divide_adv,conditional=args.conditional,second_entity = second_entity,entity_only=args.entity_only)
            print([tokenizer.decode(x[0]) for x in input_ids])
            print(len(input_ids), len(constraints),len(constraint_list_str), len(constraints[0]))
            print(constraint_list_str[0][0][:5], constraint_list_str[0][0][-1])
            for idx,input_id in enumerate(input_ids):
                cur_constraints = constraints[idx]
                cur_constraints_str = constraint_list_str[idx]
                for j,constraint_set in enumerate(cur_constraints):
                    # print(constraint_set[0],constraint_set[0].positive_state len(constraint_set))
                    # .detach().cpu().numpy().tolist()
                    constraint_sets.append({"input_id":tokenizer.decode(input_id[0]),'cur_constraints_str':cur_constraints_str[j]})
        prepared_data.append({'ent_class':ent_class,'entity': entity,'second_entity': second_entity,"prompt_and_constraint":constraint_sets})

    with open(args.output_file, "w") as f:
        for d in prepared_data:
            json.dump(d,f)
            f.write('\n')

    with open(f'./Generated/wikidata/prompts.csv', "w") as f:
        csvwriter = csv.writer(f,delimiter="\t")
        for d in prepared_data:
            csvwriter.writerow([d["prompt_and_constraint"][0]["input_id"],d["prompt_and_constraint"][0]["cur_constraints_str"]])
            # csvwriter.writerow([])
if __name__ == "__main__":
    main()
