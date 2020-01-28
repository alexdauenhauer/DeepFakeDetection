# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import pickle
import time
from os.path import dirname, abspath

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (BatchNormalization, ConvLSTM2D, Dense,
                                     Input, LeakyReLU, Conv3D)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from_generator = tf.data.Dataset.from_generator
ROOT_DIR = dirname(abspath(__file__))
import sys
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from DataPrep import DataPrepDlib


# %%
import dlib
dlib.DLIB_USE_CUDA


# %%
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)


# %%

# TODO: add variables for frame height and width
# def input_fn(filepath, batch_size=10, segment_size=5):

filepath = 'data/train_sample_videos'
segment_size = 5
datapath = os.path.join(filepath, 'metadata.json')
data = pd.read_json(datapath).T
files = [os.path.join(filepath, f) for f in data.index]
labels = data.label.values
x_train, x_test, y_train, y_test = train_test_split(
    files, labels, test_size=0.1)
class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
for k, v in zip(np.unique(y_train), class_weights):
    print(k, v)
y_train = list(map(lambda x: 0 if x == 'REAL' else 1, y_train))
y_test = list(map(lambda x: 0 if x == 'REAL' else 1, y_test))
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)
# y_train = list(map(lambda x: 0 if x == 'REAL' else 1, y_train))
# y_test = list(map(lambda x: 0 if x == 'REAL' else 1, y_test))
print(len(x_train), len(y_train), len(x_test), len(y_test))


# %%

# @tf.function
def input_fn(files, labels, segment_size=5, batch_size=1, rsz=(128, 128)):
    def dataGenerator():
        for f, label in zip(files, labels):
            dp = DataPrepDlib(segment_size=segment_size, rsz=rsz)
            frames, flows = dp.prepVid(filepath=f)
            # frames, flows = dp.prepFullFrames(filepath=f)
            yield {'rgb_input': frames, 'flow_input': flows}, label
    dataset = from_generator(
        dataGenerator,
        output_types=(
            {
                "rgb_input": tf.int8,
                "flow_input": tf.float32
            },
            tf.int8),
        output_shapes=(
            {
                "rgb_input": (segment_size, rsz[0], rsz[1], 3),
                "flow_input": (segment_size - 1, rsz[0], rsz[1], 2)
            },
            (2,))
    )
    dataset = dataset.batch(batch_size)
    return dataset


# %%
# rgb_input = tf.keras.Input(shape=(5, rsz[0], rsz[1], 3), name='rgb_input')
# x = tf.keras.layers.Flatten()
# x = Dense(128)
# x = LeakyReLU()
# x = Dense(128)
# x = LeakyReLU()
# x = Dense(128)
# x = LeakyReLU()
# x = tf.keras.layers.Dropout(0.5)
# x = Dense(1)
# %%
batch_size = 4
segment_size = 5
rsz = (128, 128)
train_data = input_fn(
    x_train,
    y_train,
    segment_size=segment_size,
    batch_size=batch_size,
    rsz=rsz)
test_data = input_fn(
    x_test,
    y_test,
    segment_size=segment_size,
    batch_size=batch_size,
    rsz=rsz)


# %%
class InputStream(tf.keras.Model):
    def __init__(self, kernel_size, filters, name):
        super().__init__(name=name)
        self.convLstm1 = ConvLSTM2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            data_format='channels_last',
            return_sequences=True,
            dropout=0.5
        )
        self.bn1 = BatchNormalization()
        self.convLstm2 = ConvLSTM2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            data_format='channels_last',
            return_sequences=True,
            dropout=0.5
        )
        self.bn2 = BatchNormalization()
        self.convLstm3 = ConvLSTM2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            data_format='channels_last',
            return_sequences=False,
            dropout=0.5
        )
        self.bn3 = BatchNormalization()
        self.act = LeakyReLU()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = Dense(128)
        self.act1 = LeakyReLU()
        self.dense2 = Dense(128)
        self.act2 = LeakyReLU()
        self.dense3 = Dense(128)
        self.act3 = LeakyReLU()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.out_layer = Dense(2)

    # TODO: specify different behavior for training, i.e. dropout only when
    # training
    def call(self, input_tensor, training=False):
        x = self.convLstm1(input_tensor)
        x = self.bn1(x)
        x = self.convLstm2(x)
        x = self.bn2(x)
        x = self.convLstm3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.act1(x)
        if training:
            x = self.dropout(x)
        x = self.dense2(x)
        x = self.act2(x)
        if training:
            x = self.dropout(x)
        x = self.dense3(x)
        x = self.act3(x)
        if training:
            x = self.dropout(x)
        return self.out_layer(x)


# %%
# class InputStream(tf.keras.Model):
#     def __init__(self, kernel_size, filters, name):
#         super().__init__(name=name)

# kernel_size = 3
# filters = 1
# rgb_input = tf.keras.Input(shape=(5, 256, 256, 3), name='rgb_input')
# convLstm1 = ConvLSTM2D(
#     filters=filters,
#     kernel_size=kernel_size,
#     strides=1,
#     padding='same',
#     data_format='channels_last',
#     return_sequences=True,
#     dropout=0.5
# )(rgb_input)
# bn = BatchNormalization()(convLstm1)
# convLstm2 = ConvLSTM2D(
#     filters=filters,
#     kernel_size=kernel_size,
#     strides=1,
#     padding='same',
#     data_format='channels_last',
#     return_sequences=False,
#     dropout=0.5
# )(bn)
# bn = BatchNormalization()(convLstm2)
# flatten = tf.keras.layers.Flatten()(bn)
# dense = Dense(512)(flatten)
# act = LeakyReLU()(dense)
# dense = Dense(512)(act)
# act = LeakyReLU()(dense)
# dense = Dense(512)(act)
# act = LeakyReLU()(dense)
# dropout = tf.keras.layers.Dropout(0.5)(act)
# out_layer = Dense(2)(dropout)
# model = Model(inputs=rgb_input, outputs=out_layer)
# model.summary()


# %%
rgb_stream = InputStream(3, 4, 'rgb_stream')
flow_stream = InputStream(3, 4, 'flow_stream')
rgb_input = tf.keras.Input(
    shape=(segment_size, rsz[0], rsz[1], 3),
    name='rgb_input')
flow_input = tf.keras.Input(
    shape=(segment_size - 1, rsz[0], rsz[1], 2),
    name='flow_input')
rgb = rgb_stream(rgb_input)
flow = flow_stream(flow_input)
final_average = tf.keras.layers.average([rgb, flow])
x = tf.keras.layers.Flatten()(final_average)
final_output = Dense(2, activation='softmax', name='final_output')(x)
# final_output = Dense(2, activation='sigmoid', name='final_output')(x)
model = Model(
    inputs={"rgb_input": rgb_input, "flow_input": flow_input},
    outputs=final_output,
    name='my_model'
)
# model.summary()


# # %%
# rgb_stream.shape
# # %%
# rgb.shape
# # %%
# final_average.shape
# # %%
# x.shape
# # %%
# final_output.shape
# # %%
# f = x_train[0]
# dp = DataPrepDlib(segment_size=segment_size)
# frames, flows = dp.prepVid(filepath=f)
# sample = {'rgb_input': frames, 'flow_input': flows}
# x = model(sample)
# x.shape
# %%

# %%

# %%
# opt = tf.keras.optimizers.Adam()
# model.compile(
#     optimizer=opt,
#     loss='categorical_crossentropy',
#     metrics=['acc'])
# model.fit(
#     train_data,
#     epochs=25,
#     verbose=1,
#     class_weight=class_weights
# )


# %%
opt = tf.keras.optimizers.Adam()
model.compile(
    optimizer=opt,
    loss='binary_crossentropy',
    metrics=['acc'])
model.fit(
    x=train_data,
    validation_data=test_data,
    epochs=10,
    verbose=1,
    class_weight=class_weights
)


# %%
model.evaluate(
    test_data,
    #     class_weight=class_weights
)


# %%
from datetime import datetime
dt = datetime.now().strftime('%Y%m%d_%H%M%S')
dt


# %%
savepath = f'/data/models/{dt}'
os.makedirs(savepath, exist_ok=True)
model.save(savepath)


# %%
