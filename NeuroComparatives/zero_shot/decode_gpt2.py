import json
import math
import argparse
import torch
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from os import path
from transformers import AutoTokenizer, AutoModelWithLMHead

from zero_shot.generate import generate
from zero_shot import utils
from lexical_constraints import init_batch

logger = logging.getLogger(__name__)

import random
random.seed(10)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, help="pretrained language model to use")
    parser.add_argument("--output_file", type=str, help="output file")
    parser.add_argument("--input_file", type=str, help="jsonl file containing constraints and prefix")

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

    # TODO: Prune factor is deprecated. Remove from code.
    parser.add_argument('--diversity', type=bool, default=True,
                        help="whether to encourage POS diversity")
    parser.add_argument('--sat_tolerance', type=int, default=2,
                        help="minimum satisfied clause of valid candidates")
    parser.add_argument('--look_ahead', type=int, default=None,
                        help="number of step to look ahead")
    parser.add_argument('--num_sequences_per_return', type=int, default=5,
                        help="number of sequences per return")
    parser.add_argument('--do_beam_sample', type=bool, default=True,
                        help="whether to use sampling in neurologic")
    parser.add_argument('--beam_temperature', type=float, default=0.6,
                        help="temperature for neurologic sampling")

    parser.add_argument('--num_gpus', default=1, help="No. of GPUs to use. If more than 1, split the model.", type=int)

    args = parser.parse_args()

    print(f"Decoding with: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelWithLMHead.from_pretrained(args.model_name)

    if args.num_gpus == 4:

        device_map = {0: [0, 1, 2, 3, 4, 5, 6, 7, 8],
                      1: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
                      2: [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                      3: [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]}
        print("Splitting model like so: {}".format(json.dumps(device_map, indent=2)))
        torch.cuda.empty_cache()
        model.parallelize(device_map) # Splits the model across several devices
        DEVICE = "cuda:0"
        model.eval()
    else:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        torch.cuda.empty_cache()
        model = model.to(DEVICE)
        model.eval()

    print(f'vocab_size: {tokenizer.vocab_size}, POS_size: {len(utils.POS)}')
    digit_num = 10 ** (len(str(tokenizer.vocab_size)) - 1)
    POS_start = (tokenizer.vocab_size // digit_num + 1) * digit_num
    POS = {k: v + POS_start for k, v in utils.POS.items()}





    period_id = [tokenizer.convert_tokens_to_ids('.')]
    period_id.append(tokenizer.convert_tokens_to_ids('Ġ.'))
    eos_ids = [tokenizer.eos_token_id] + period_id
    PAD_ID = tokenizer.convert_tokens_to_ids('<pad>')
    bad_token = [':', "'", '-', '_', '@', 'Ċ', 'Ġ:', 'Ġ(', 'Ġ)', '(', ')', 'Ġ[', 'Ġ]', '[', ']'
                 , 'Ġ{', 'Ġ}', '{', '}']
    bad_words_ids = [tokenizer.convert_tokens_to_ids([t]) for t in bad_token]

    input_lines = []
    constraints_list = []
    original_jsonl_objs = []
    with open(args.input_file) as f:
        for line in f:
            obj = json.loads(line)
            original_jsonl_objs.append(obj)
            input_lines.append(obj['prompt_text'])
            current_constraints = []
            for cons in obj['constraints']['clauses']:
                current_constraints.append(cons)
            constraints_list.append(current_constraints)

    constraints_list = utils.tokenize_constraints(tokenizer, POS, constraints_list)

    input_lines = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)) for x in input_lines]

    if path.exists(args.output_file):
        count = len(open(args.output_file, 'r').readlines())
        fout = Path(args.output_file).open("a", encoding="utf-8")
        input_lines = input_lines[count:]
        constraints_list = constraints_list[count:]
    else:
        fout = Path(args.output_file).open("w", encoding="utf-8")
    total_batch = math.ceil(len(input_lines) / args.batch_size)
    next_i = 0

    resultant_jsonl_objs = []

    with tqdm(total=total_batch) as pbar:
        while next_i < len(input_lines):
            _chunk = input_lines[next_i:next_i + args.batch_size]
            constraints = init_batch(raw_constraints=constraints_list[next_i:next_i + args.batch_size],
                                     beam_size=args.beam_size,
                                     eos_id=eos_ids)
            jsonl_objs = original_jsonl_objs[next_i:next_i + args.batch_size]
            buf = _chunk
            next_i += args.batch_size

            max_len = max([len(x) for x in buf])
            buf = [x + [PAD_ID] * (max_len - len(x)) for x in buf]

            input_ids = torch.stack([torch.from_numpy(np.array(x)) for x in buf])
            input_ids = input_ids.to(DEVICE)
            attention_mask = (~torch.eq(input_ids, PAD_ID)).int()
            attention_mask = attention_mask.to(DEVICE)

            outputs = generate(self=model,
                               input_ids=input_ids,
                               attention_mask=attention_mask,
                               pad_token_id=PAD_ID,
                               bad_words_ids=bad_words_ids,
                               min_length=args.min_tgt_length,
                               max_length=args.max_tgt_length,
                               num_beams=args.beam_size,
                               no_repeat_ngram_size=args.ngram_size,
                               length_penalty=args.length_penalty,
                               constraints=constraints,
                               diversity=args.diversity,
                               sat_tolerance=args.sat_tolerance,
                               look_ahead=args.look_ahead,
                               num_sequences_per_return=args.num_sequences_per_return,
                               do_beam_sample=args.do_beam_sample,
                               beam_temperature=args.beam_temperature)

            prompt = [tokenizer.decode(x) for x in buf]
            output_sequences = [tokenizer.decode(o).replace('<|endoftext|>', '').strip()
                                for i, o in enumerate(outputs)]

            assert len(jsonl_objs) * args.num_sequences_per_return == len(output_sequences)

            chunks_of_generations = [output_sequences[idx: idx + args.num_sequences_per_return] for idx in range(0, len(output_sequences), args.num_sequences_per_return)]
            for obj, chunk in zip(jsonl_objs, chunks_of_generations):
                obj['output_sequences'] = chunk
                fout.write(json.dumps(obj))
                fout.write("\n")
                fout.flush()

            pbar.update(1)


if __name__ == "__main__":
    main()
