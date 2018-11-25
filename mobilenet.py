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
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow_functions import *

start = dt.datetime.now()

BASE_SIZE = 256
NCSVS = 100
NCATS = 340
np.random.seed(seed=1987)
tf.set_random_seed(seed=1987)

TEPS = 1000
EPOCHS = 10
size = 128
batchsize = 1400

model = MobileNet(input_shape=(size, size, 1), alpha=1., weights=None, classes=NCATS)
model.load_weights('')
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy',
                      metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])

valid_df = pd.read_csv(os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(NCSVS - 1)), nrows=35000)
x_valid = df_to_image_array_xd(valid_df, size)
y_valid = keras.utils.to_categorical(valid_df.y, num_classes=NCATS)
print(x_valid.shape, y_valid.shape)
print('Validation array memory {:.2f} GB'.format(x_valid.nbytes / 1024.**3 ))

train_datagen = image_generator_xd(size=size, batchsize=batchsize, ks=range(NCSVS - 1))

callbacks = [ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=5,
		   min_delta=0.0005, mode='max', cooldown=3, verbose=1), 
		ModelCheckpoint('models/mobilenet-best-run.h5', monitor='val_categorical_accuracy',
		mode='max', save_best_only=True, verbose=1)]

hists = []
hist = model.fit_generator(
    train_datagen, steps_per_epoch=STEPS, epochs=EPOCHS, verbose=1,
    validation_data=(x_valid, y_valid),
    callbacks = callbacks
)
hists.append(hist)
