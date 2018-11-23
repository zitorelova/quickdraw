import numpy as np  
import pandas as pd  
from pathlib import Path
from fastai import *
from fastai.vision import * 
import json 
from doodle_utils import *
from time import time
from losses import * 

model_name = ''

sz = 128
bs = 256
PATH = Path('data')
NUM_VAL = 50 * 340
NCATS = 340

start = time()

def create_func(path):
    with open(path) as f: j = json.load(f)
    drawing = list2drawing(j['drawing'], size=sz)
    tensor = drawing2tensor(drawing)
    return Image(tensor.div_(255))

surr_loss = svm.SmoothTopkSVM(n_classes=NCATS, alpha=1., k=3)

print("Creating item list")

item_list = ItemList.from_folder(PATH/'train/', create_func=create_func)

idxs = np.arange(item_list.items.shape[0])
np.random.shuffle(idxs)
val_idxs = idxs[:NUM_VAL]

item_lists = item_list.split_by_idx(val_idxs)

classes = pd.read_pickle('data/classes.pkl')

label_lists = item_lists.label_from_folder(classes=classes)

test_items = ItemList.from_folder(PATH/'test/', create_func=create_func)
label_lists.add_test(test_items)

print("Creating data loaders")
train_dl = DataLoader(label_lists.train, bs, True, num_workers=8)
valid_dl = DataLoader(label_lists.valid, bs, False, num_workers=8)
test_dl = DataLoader(label_lists.test, bs, False, num_workers=8)

data_bunch = ImageDataBunch(train_dl, valid_dl, test_dl)

#pd.to_pickle(data_bunch.batch_stats(), f'data/batch_stats_{sz}.pkl')
batch_stats = pd.read_pickle(f'data/batch_stats_{sz}.pkl')
data_bunch.normalize(batch_stats)

learn = create_cnn(data_bunch, models.resnet50, metrics=[accuracy, map3])
learn.load(model_name)

preds, _ = learn.get_preds(ds_type=DatasetType.Test)
create_submission(preds, data_bunch.test_dl, name, classes)
print(f'Finished in {round(time() - start, 3) / 60} minutes')