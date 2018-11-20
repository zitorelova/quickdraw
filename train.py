import numpy as np  
import pandas as pd  
from pathlib import Path
from fastai import *
from fastai.vision import * 
#from fastai.data_block import ItemList
import json 
from doodle_utils import *
from time import time

run = 1
sz = 128
bs = 512
PATH = Path('data')
NUM_VAL = 5 * 340

def create_func(path):
    with open(path) as f: j = json.load(f)
    drawing = list2drawing(j['drawing'], size=sz)
    tensor = drawing2tensor(drawing)
    return Image(tensor.div_(255))

print("Creating dataset")

item_list = ItemList.from_folder(PATH/'train/', create_func=create_func)

idxs = np.arange(item_list.items.shape[0])
np.random.shuffle(idxs)
val_idxs = idxs[:NUM_VAL]

item_lists = item_list.split_by_idx(val_idxs)

#label_lists = item_lists.label_from_folder()
#pd.to_pickle(label_lists.train.y.classes, 'data/classes.pkl')

classes = pd.read_pickle('data/classes.pkl')

label_lists = item_lists.label_from_folder(classes=classes)

test_items = ItemList.from_folder(PATH/'test/', create_func=create_func)
label_lists.add_test(test_items)

train_dl = DataLoader(label_lists.train, bs, True, num_workers=8)
valid_dl = DataLoader(label_lists.valid, 2*bs, False, num_workers=8)
test_dl = DataLoader(label_lists.test, 2*bs, False, num_workers=8)

data_bunch = ImageDataBunch(train_dl, valid_dl, test_dl)


pd.to_pickle(data_bunch.batch_stats(), f'data/batch_stats_{sz}.pkl')
batch_stats = pd.read_pickle(f'data/batch_stats_{sz}.pkl')
data_bunch.normalize(batch_stats)

# Define the network 
name = f'resnet34-{sz}'

learn = create_cnn(data_bunch, models.resnet34, metrics=[accuracy, map3])

print(f'Starting training run on {sz} image size')
start = time()
learn.opt_fn = optim.Adam
#learn.metrics = [map3]
learn.fit_one_cycle(cyc_len=1, max_lr=2e-3)

learn.save(f'{name}-run-{run}')
print(f'Finished in {round(time() - start, 3) / 60} minutes')

