import numpy as np  
import pandas as pd 
import json
import cv2
import os
from pathlib import Path
from doodle_utils import *
from time import time
from tqdm import tqdm

NUM_SAMPLES_PER_CLASS = 30000
NUM_VALIDATION = 50 * 340
PATH = Path('data')

PATH.mkdir(exist_ok=True)
(PATH/'train').mkdir(exist_ok=True)
(PATH/'test').mkdir(exist_ok=True)

def create_train_txts_from_df(path):
    df = pd.read_csv(path)
    klass = '_'.join(path.stem.split())
    (PATH/'train'/klass).mkdir(exist_ok=True)
    for row in df.sample(NUM_SAMPLES_PER_CLASS).iterrows():
        example = {
            'countrycode': row[1].countrycode,
            'drawing': json.loads(row[1].drawing),
            'key_id': row[1].key_id,
            'recognized': row[1].recognized
        }
        with open(PATH/'train'/klass/f'{example["key_id"]}.txt', mode='w') as f: json.dump(example, f)

def create_test_txts_from_df(path):
    df = pd.read_csv(path)
    for row in tqdm(df.iterrows()):
        example = {
            'countrycode': row[1].countrycode,
            'drawing': json.loads(row[1].drawing),
            'key_id': row[1].key_id
        }
        with open(PATH/'test'/f'{example["key_id"]}.txt', mode='w') as f: json.dump(example, f)


start = time()
print("Creating train text files from csv")
for p in tqdm(Path('train_simplified').iterdir()): create_train_txts_from_df(p)
print(f'Finished train texts in {round(time() - start,2) / 60} minutes')

#start = time()
#print("creating test text files from csv")
#create_test_txts_from_df('test_simplified.csv')
#print(f'Finished test texts in {round(time() - start,2) / 60} minutes')


