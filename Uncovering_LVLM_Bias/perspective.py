import pandas as pd
from googleapiclient import discovery
import json
import time
import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NpEncoder, self).default(obj)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Path to output directory.",
    )
    parser.add_argument(
        "--occupations",
        type=str,
        required=True,
        help="List of occupations to use. Set to 'all' to disable filtering on occupation",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        required=True,
        help="List of prompts to use.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        required=True,
        help="Number of counterfactual sets to sample per occupation. Set to -1 to disable subsampling",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="API key for the Perspective API.",
    )
    parser.add_argument(
        "--n-partitions",
        type=int,
        default=1,
        help="Number of total partitions.",
    )
    parser.add_argument(
        "--partition",
        type=int,
        default=0,
        help="Partition index.",
    )
    args = parser.parse_args()

    client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=args.api_key,
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    static_discovery=False,
    )
    
    occupations = args.occupations.split(',')
    text_cols = args.prompts.split(',')

    generations = pd.read_csv(os.path.join(args.out_dir, 'generations.csv'))
    if occupations[0] != 'all':
        print('Filtering and subsampling for occupations: ' + str(occupations))
        gens_subset = []
        for i in occupations:
            gens = generations[generations['subject'] == i]
            if args.n_samples != -1:
                top_100 = gens[['filename','rank','im_seed']].drop_duplicates().sample(n=args.n_samples, random_state=0)
                gens = gens[gens.apply(lambda x: x['filename'] + str(x['rank']) + str(x['im_seed']), axis=1).isin(top_100.apply(lambda x: x['filename'] + str(x['rank']) + str(x['im_seed']), axis=1))]
            gens_subset.append(gens)
        generations = pd.concat(gens_subset)

    print(f"Partition index {args.partition} ({args.partition + 1} / {args.n_partitions})")
    generations_index = np.array_split(list(range(generations.shape[0])), args.n_partitions)[args.partition]
    generations = generations.iloc[generations_index,:]

    filename = 'perspective_api_' 
    filename+= 'all_occupations_' if occupations[0] == 'all' else str(len(occupations)) + '_occupations_' 
    filename+= 'all_gens_' if args.n_samples == -1 else str(args.n_samples) + '_samples_' 
    filename+= '_all_prompts_' if len(text_cols) == 18 else str(len(text_cols)) + '_prompts_'
    filename+= str(args.partition)
    out_file = os.path.join(args.out_dir, filename + '.jsonl')
    scores = []
    if os.path.isfile(out_file):
        with open(out_file,"r") as f:
            for line in f:
                scores.append(json.loads(line))

    # t2 = time.time()
    start_index = len(scores)
    for i in tqdm(range(start_index, generations.shape[0])):
        df = generations.iloc[i:i+1][[col for col in generations.columns if col not in text_cols]]
        out_dict = {k: df[k].values[0] for k in df.columns}
        out_dict['scores'] = {}
        
        for j in text_cols:
            analyze_request = {
            'comment': { 'text': str(generations.iloc[i][j]) },
            'requestedAttributes': {'TOXICITY': {}, 'SEVERE_TOXICITY': {}, 'IDENTITY_ATTACK': {}, 'INSULT': {}, 'PROFANITY': {}, 'THREAT': {}, 'SEXUALLY_EXPLICIT': {}, 'FLIRTATION': {}},
            'languages' : ['en']
            }
            # t1 = time.time()
            # time.sleep(max(0.9-(t1-t2),0))
            response = client.comments().analyze(body=analyze_request).execute()
            # t2 = time.time()
            out_dict['scores'][j] = response
        
        scores.append(out_dict)
        with open(out_file, 'a') as f:
            json.dump(out_dict, f, cls=NpEncoder)
            f.write(os.linesep)

    out_df = generations[[col for col in generations.columns if col not in text_cols]]
    for j in text_cols:
        out_df[j] = [i['scores'][j] for i in scores]
    out_df = pd.melt(out_df, id_vars= [col for col in generations.columns if col not in text_cols], var_name='prompt', value_name='scores')
    score_cols = list(out_df.iloc[0]['scores']['attributeScores'].keys())
    for k in score_cols:
        out_df[k] = out_df['scores'].apply(lambda x: x['attributeScores'][k]['summaryScore']['value'])

    out_df.to_csv(os.path.join(args.out_dir, filename + '.csv'), index=False)