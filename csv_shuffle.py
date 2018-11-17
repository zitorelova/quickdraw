import json
import os
import datetime as dt
from tqdm import tqdm
import pandas as pd
import numpy as np

IMAGES_PER_CLASS = 100000
NCSVS = 100

def f2cat(fname: str) -> str:
    return fname.split('.')[0]

class Simplified():
    def __init__(self, input_path='./input'):
        self.input_path = input_path
        
    def list_all_categories(self):
        files = os.listdir(os.path.join(self.input_path, 'train_simplified'))
        return sorted([f2cat(f) for f in files], key=str.lower)
    
    def read_training_csv(self, category, nrows=None, usecols=None, drawing_transform=False):
        df = pd.read_csv(os.path.join(self.input_path, 'train_simplified', category + '.csv'),
                         nrows=nrows, parse_dates=['timestamp'], usecols=usecols)
        if drawing_transform:
            df['drawing'] = df['drawing'].apply(json.loads)
        return df

start = dt.datetime.now()
s = Simplified('./')
categories = s.list_all_categories()
print("There are {} categories.".format(len(categories)))

for y, cat in tqdm(enumerate(categories)):
    df = s.read_training_csv(cat, nrows=IMAGES_PER_CLASS)
    df['y'] = y 
    df['cv'] = (df.key_id // 10 ** 7) % NCSVS
    for k in range(NCSVS):
        fname = 'train_k{}.csv'.format(k)
        chunk = df[df.cv == k]
        chunk = chunk.drop(['key_id'], axis=1)
        if not y:
            chunk.to_csv(fname, index=False)
        else:
            chunk.to_csv(fname, mode='a', header=False, index=False)

for k in tqdm(range(NCSVS)):
    fname = 'train_k{}.csv'.format(k)
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        df['rnd'] = np.random.rand(len(df))
        df = df.sort_values(by='rnd').drop('rnd', axis=1)
        df.to_csv(fname + '.gz', compression='gzip', index=False)
        os.remove(fname)

end = dt.datetime.now()
print('Ran on {}.\nTotal time {}m'.format(end, (end - start).minutes))