import os
import ast
import datetime as dt
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow_functions import *
import warnings 
from time import time
warnings.filterwarnings('ignore')

start = time()

#BASE_SIZE = 256
NCSVS = 200
NCATS = 340
DP_DIR = 'shuffled_csv/'
np.random.seed(seed=1987)
tf.set_random_seed(seed=1987)

STEPS = 1000
EPOCHS = 100
size = 128
batchsize =512

print("Setting up MobileNet")
model = MobileNet(input_shape=(size, size, 1), alpha=1., weights=None, classes=NCATS)
model.load_weights('models/mobilenet-best-run.h5')

model.compile(optimizer=Adam(lr=0.008), loss='categorical_crossentropy',
                      metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])

valid_df = pd.read_csv(os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(NCSVS - 1)), nrows=35000)
x_valid = df_to_image_array_xd(valid_df, size)
y_valid = keras.utils.to_categorical(valid_df.y, num_classes=NCATS)
print(x_valid.shape, y_valid.shape)
print('Validation array memory {:.2f} GB'.format(x_valid.nbytes / 1024.**3 ))

train_datagen = image_generator_xd(size=size, batchsize=batchsize, ks=range(NCSVS - 1))

callbacks = [ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=5,
		   min_delta=0.00005, mode='max', cooldown=3, verbose=1), 
		ModelCheckpoint('models/mobilenet-best-run-2.h5', monitor='val_categorical_accuracy',
		mode='max', save_best_only=True, verbose=1)]

hists = []
hist = model.fit_generator(
    train_datagen, steps_per_epoch=STEPS, epochs=EPOCHS, verbose=1,
    validation_data=(x_valid, y_valid),
    callbacks = callbacks
)
hists.append(hist)

valid_predictions = model.predict(x_valid, batch_size=256, verbose=1)
map3 = mapk(valid_df[['y']].values, preds2catids(valid_predictions).values)
print('Map3: {:.3f}'.format(map3))

test = pd.read_csv('test_simplified.csv')
test.head()
x_test = df_to_image_array_xd(test, size)
x_test_flip = np.flip(x_test, 2)
print(test.shape, x_test.shape)
print('Test array memory {:.2f} GB'.format(x_test.nbytes / 1024.**3 ))

# TTA
test_predictions1 = model.predict(x_test, batch_size=128, verbose=1)
test_predictions2 = model.predict(x_test_flip, batch_size=128, verbose=1)
final_predictions = np.average([test_predictions1, test_predictions2], axis=0, weights=[0.6,0.4])

top3 = preds2catids(final_predictions)
cats = list_all_categories()
id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(cats)}
top3cats = top3.replace(id2cat)
test['word'] = top3cats['a'] + ' ' + top3cats['b'] + ' ' + top3cats['c']
submission = test[['key_id', 'word']]
submission.to_csv('subs/mobilenet_sub.csv', index=False)

print("Finished in %s mins" % ((time() - start) / 60))

