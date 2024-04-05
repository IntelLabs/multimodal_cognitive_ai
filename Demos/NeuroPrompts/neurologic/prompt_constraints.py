import json
import pandas as pd
import os
import itertools

QA_SPECIAL_TOKENS = {"Question": "<human>", "Answer": "<bot>", "StartPrefix": "<prefix>", "EndPrefix": "</prefix>"}

def format_pair(pairs): 
    return ["{}{}{}".format(QA_SPECIAL_TOKENS["Question"], pairs[0], QA_SPECIAL_TOKENS["Answer"])]

def get_noun_verb_chunks(text, tagger):
    sentence = Sentence(text)
    tagger.predict(sentence)

    chunks = [i.data_point.tokens for i in sentence.get_labels() if i.value in ['VP', 'NP']]
    chunks = [' '.join([j.text for j in i]) for i in chunks]

    return chunks

def load_inputs(input, constraint_method, clusters_file, max_n=None, model_type=None, 
                n_per_prompt=1, n_clusters = 5, n_per_cluster = 5, user_constraints=None,
                negative_constraints=None, seed=0):

    if constraint_method == 'clusters':
        with open(clusters_file, 'rb') as f:
            clusters = json.load(f)
    else:
        raise NotImplementedError

    if os.path.isfile(input):
        prompts = []; index = []
        if input.endswith('.txt'):
            with open(input) as f:
                for line in f.readlines():
                    i, text = line.strip().split('\t')
                    index.append(i)
                    prompts.append(text)
        elif input.endswith('.json'):
            with open(input, 'rb') as f:
                input_dict = json.load(f)
            prompts = [i['generated_caption'] for i in input_dict if len(i['generated_caption']) > 5]
            index = [i['curr_img_name'] for i in input_dict if len(i['generated_caption']) > 5]

        prompts = prompts[:max_n]
        index = index[:max_n]
    else:
        prompts = [input]
        index = [0]

    inputs = []
    for j in range(len(prompts)):
        for k in range(n_per_prompt):
            text = prompts[j].capitalize()

            skip = False
            if user_constraints is not None:
                input_dict = {"original_prompt" : text, "index" : index[j]}
                constraints_dict = [{"terms": c, "polarity": 1, "max_count": 1, "min_count": 1, "type": "Term", "order" : 0} for c in user_constraints]
                if negative_constraints is not None:
                    constraints_dict += [{"terms": negative_constraints, "polarity": 0, "max_count": 0, "min_count": 0, "type": "Term", "order" : 0}]
                gen_prefix = text.rstrip('.')
                input_dict['prompt_and_constraint'] = [{"input_id": gen_prefix, "cur_constraints_str": constraints_dict}]
            elif constraint_method == 'clusters':
                input_dict = {"original_prompt" : text, "index" : index[j]}
                constraints_dict = []
                order = 0
                for i in range(n_clusters):
                    constraints_dict.append({"terms": pd.Series(clusters['cluster_' + str(i)]).sample(n=n_per_cluster, random_state=seed).to_list(), 
                                            "polarity": 1, "max_count": 1, "min_count": 1, "type": "Term", "order" : order})
                if negative_constraints is not None:
                    constraints_dict += [{"terms": negative_constraints, "polarity": 0, "max_count": 0, "min_count": 0, "type": "Term", "order" : 0}]
                gen_prefix = text.rstrip('.')
                input_dict['prompt_and_constraint'] = [{"input_id": gen_prefix, "cur_constraints_str": constraints_dict}]
            else:
                raise NotImplementedError

            if not skip:
                inputs.append(input_dict)
    
    return inputs