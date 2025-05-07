import json
import os
import pandas as pd
import argparse
import warnings
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Path to output directory.",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        required=True,
        help="Path to the metadata file.",
    )
    args = parser.parse_args()

    files = [i for i in os.listdir(args.out_dir) if i.endswith('.jsonl')]
    out_list = []
    for file in files:
        with open(os.path.join(args.out_dir, file), 'r') as f:
            for line in f:
                out_list.append(json.loads(line))

    out_df = []
    k = 0
    for out_dict in tqdm(out_list):
        df = {i: out_dict[i] for i in out_dict.keys() if i not in ['args', 'text']}
        # df = {**df, **out_dict['args']}
        if 'args' in out_dict.keys():
            df['do_sample'] = out_dict['args']['do_sample']
        df['text'] = out_dict['text']
        if type(out_dict['text']) == list:
            df = pd.DataFrame(df, index=list(range(k,k+len(out_dict['text']))))
            k+=len(out_dict['text'])
        else:
            df = pd.DataFrame(df,index=[k])
            k+=1
        out_df.append(df)

    out_df = pd.concat(out_df)
    if out_df['text_seed'].value_counts().std() != 0:
        warnings.warn('The number of generations per text seed differs:')
        print(out_df['text_seed'].value_counts())

    out_df['text_seed'] = out_df['text_seed'].astype(str)
    out_df = out_df.pivot(columns=['prompt', 'text_seed'], values='text', index=[i for i in out_df.columns if i not in ['prompt', 'text_seed','text','response']])
    out_df = out_df.reset_index()
    out_df.columns = ['_'.join(col) for col in out_df.columns.values]
    out_df = out_df.rename(columns = {i : i[:-1] for i in out_df.columns if i.endswith('_')})

    if args.metadata_path != "None":
        metadata = pd.read_csv(args.metadata_path)

        attributes = metadata[metadata['filename'].apply(lambda x: x.startswith('a_picture_of_'))].iloc[0:1,:]
        dataset_type = '_'.join(attributes['filename'].values[0].split('_')[3:5])
        output_cols = [i for i in attributes.columns if i.startswith('caption_') and len(i) < len('caption_11_')]

        attributes['prefix'] = 'A picture of a '
        attributes['prefix_2'] = 'A picture of an '
        attributes['subject'] = attributes['filename'].apply(lambda x: ' ' + ' '.join(x.split('_')[5:]))
        for i in output_cols:
            attributes[i] = attributes[i].str.replace(attributes['prefix'].iloc[0],'').str.replace(attributes['prefix_2'].iloc[0],'').str.replace(attributes['subject'].iloc[0],'').str.replace(attributes['subject'].iloc[0].title(),'')

        attributes = attributes[output_cols]
        attributes = attributes.rename(columns = {i: i.split('_')[1] for i in attributes.columns})
        attributes = pd.melt(attributes, var_name = 'im_index', value_name = 'a1a2')

        if dataset_type == 'race_gender':
            a1 = attributes['a1a2'].apply(lambda x: ' '.join(x.split()[:-1]))
            a2 = attributes['a1a2'].apply(lambda x: x.split()[-1])
            attributes = attributes.assign(a1 = a1, a2 = a2)
            attributes = attributes.assign(a1_type = 'race', a2_type = 'gender', dataset_type = dataset_type)
        elif dataset_type == 'physical_race':
            a1 = attributes['a1a2'].apply(lambda x: x.split()[0])
            a2 = attributes['a1a2'].apply(lambda x: ' '.join(x.split()[1:]))
            attributes = attributes.assign(a1 = a1, a2 = a2)
            attributes = attributes.assign(a1_type = 'physical', a2_type = 'race', dataset_type = dataset_type)
        elif dataset_type == 'physical_gender':
            a1 = attributes['a1a2'].apply(lambda x: x.split()[0])
            a2 = attributes['a1a2'].apply(lambda x: x.split()[1])
            attributes = attributes.assign(a1 = a1, a2 = a2)
            attributes = attributes.assign(a1_type = 'physical', a2_type = 'gender', dataset_type = dataset_type)
        else:
            raise NotImplementedError

        out_df['im_index'] = out_df['im_index'].astype(str)
        attributes['im_index'] = attributes['im_index'].astype(str)
        out_df = pd.merge(out_df, attributes, how = 'left', on = 'im_index')

        out_df['subject'] = out_df.apply(lambda x: x['filename'][x['filename'].index(x['dataset_type']):].replace(x['dataset_type'] + '_',''), axis=1)

    out_df.to_csv(os.path.join(args.out_dir, 'generations.csv'), index=False)