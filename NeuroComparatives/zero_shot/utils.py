# coding=utf-8
# The MIT License (MIT)

# Copyright (c) Microsoft Corporation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
from typing import List

POS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ',
       'SPACE', 'SYM', 'VERB', 'X']
POS = {k: v for v, k in enumerate(POS)}


def read_lines(input_file: str) -> List[str]:
    lines = []
    with open(input_file, "rb") as f:
        for l in f:
            lines.append(l.decode().strip())
    return lines


def read_constraints(constraints_filename):
    """

    Args:
        constraints_filename: Path of the file containing constraints. One instance per line. Each line formatted as a jsonl of the following format:

        {
          "clauses": [
            {
              "terms": ["is", "are", "for"],
              "polarity": 1,
              "max_count": 1,
              "min_count": 1,
              "type": "Term"
            },
            {
              "terms": ["apple"],
              "polarity": 0,
              "max_count": 0,
              "min_count": 0,
              "type": "Term"
            }
          ]
        }

    Returns: thing

    """
    constraints_for_all_instances = []
    lines = read_lines(constraints_filename)

    for line in lines:
        constraint_obj = json.loads(line)
        current_constraints = []
        for cons in constraint_obj['clauses']:
            current_constraints.append(cons)
        constraints_for_all_instances.append(current_constraints)
    return constraints_for_all_instances


def tokenize_constraints(tokenizer, POS_tokenizer, raw_cts):

    def tokenize(phrase, cons_type):
        if cons_type == "Term":
            token_ids = [tokenizer.encode(f' {phrase}')]
        elif cons_type == "POS":
            token_ids = [[POS_tokenizer[x] for x in phrase.split('-')]]
        elif cons_type == "Punc":
            token_ids = [tokenizer.encode(f'{phrase}'), tokenizer.encode(f' {phrase}')]
        else:
            raise NotImplementedError
        return token_ids

    constraints_for_all_instances = []
    for cons in raw_cts:
        constraints_for_all_instances.append([(
            [x for t in list(map(tokenize, clause['terms'], [clause['type']] * len(clause['terms']))) for x in t],
            clause['polarity'] == 1,
            clause['min_count'],
            clause['max_count'],
            clause['type'],
            clause['look_ahead'] if 'look_ahead' in clause else 0,
        ) for clause in cons])

    return constraints_for_all_instances
