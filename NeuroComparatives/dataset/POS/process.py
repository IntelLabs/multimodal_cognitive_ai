import json
from transformers import AutoTokenizer

POS_LIST = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SPACE', 'SYM', 'VERB', 'X']
POS_MAP = {k: v for v, k in enumerate(POS_LIST)}


def read_common_pos(common_pos_file, start_index):
    dictionary = open(common_pos_file, 'r').readlines()

    mapper = {}
    for i, row in enumerate(dictionary):
        try:
            word = row.split()[0]
            pos = json.loads(' '.join(row.strip().split()[1:]))
        except:
            word = ' '
            pos = json.loads(row.strip())

        mapper[word] = [POS_MAP[x] + start_index for x in pos.keys()]
    return mapper
