from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import os
import pandas as pd

im_path = 'SocialCounterfactuals/images'
Path(im_path).mkdir(parents=True, exist_ok=True)

metadata = pd.read_csv('metadata/metadata.csv')

ds = load_dataset('SocialCounterfactuals/')
for i in tqdm(ds['train']):
    filename = metadata[(metadata['counterfactual_set'] == i['counterfactual_set']) & (metadata['a1a2'] == i['a1a2'])]['file_name'].str.split('/').str[1].values[0]
    i['image'].save(os.path.join(im_path, filename))