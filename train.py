import matplotlib
matplotlib.use('agg')
import numpy as np  
import pandas as pd  
from pathlib import Path
from fastai import *
from fastai.vision import * 
import json 
from doodle_utils import *
from time import time
from losses import * 
from tqdm import tqdm
#from callbacks import *
import pretrainedmodels


use_pretrained = False
pretrain_name = 'resnet50-128-run-1'

run = 1
sz = 128
bs = 256
PATH = Path('data')
NUM_VAL = 50 * 340
NCATS = 340

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

label_lists = item_lists.label_from_folder()
pd.to_pickle(label_lists.train.y.classes, 'data/classes.pkl')

classes = pd.read_pickle('data/classes.pkl')

label_lists = item_lists.label_from_folder(classes=classes)

test_items = ItemList.from_folder(PATH/'test/', create_func=create_func)
label_lists.add_test(test_items)

print("Creating data loaders")
train_dl = DataLoader(label_lists.train, bs, True, num_workers=8)
valid_dl = DataLoader(label_lists.valid, bs, False, num_workers=8)
test_dl = DataLoader(label_lists.test, bs, False, num_workers=8)

data_bunch = ImageDataBunch(train_dl, valid_dl, test_dl)


pd.to_pickle(data_bunch.batch_stats(), f'data/batch_stats_{sz}.pkl')
batch_stats = pd.read_pickle(f'data/batch_stats_{sz}.pkl')
data_bunch.normalize(batch_stats)

# Define the network 
name = f'seresnext50-{sz}-run-{run}'

model_name = 'se_resnext50_32x4d'
mod = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

learn = create_cnn(data_bunch, mod, metrics=[accuracy, map3])

print(f'Starting training run on {sz} image size')
start = time()
learn.opt_fn = optim.Adam
#lr = 8e-3
#lr_arr = np.array([lr/100, lr/10, lr])
#learn.crit = softmax_cross_entropy_criterion
learn.crit = surr_loss
learn.models_path = './models/'

if use_pretrained:
    learn.load(pretrain_name)

#mod_checkpoint = SaveBestModel(model=learn, lr=lr, name=best_name)

#learn.fit(1, 3, use_clr=(10,10))
#
# FOR FINDING LR

print("Looking for LR")
learn.lr_find()
learn.recorder.plot()
plt.savefig('lr_plot.png')

#learn.freeze_to(1)
#learn.fit_one_cycle(2, lr, div_factor=100, pct_start=0.3)
#learn.save(name)
#learn.unfreeze()
#learn.fit_one_cycle(2, lr_arr, div_factor=25, pct_start=0.3)
#
#learn.save(name)
#
#learn.load(name)
#
#preds, _ = learn.get_preds(ds_type=DatasetType.Test)
#
#create_submission(preds, data_bunch.test_dl, name, classes)

print(f'Finished in {round(time() - start, 3) / 60} minutes')

