# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import pickle
import time

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

from DataPrep import DataPrep


# %%

# TODO: add variables for frame height and width
# def input_fn(filepath, batch_size=10, segment_size=5):

filepath = 'data/train_sample_videos'
segment_size = 5
datapath = os.path.join(filepath, 'metadata.json')
data = pd.read_json(os.path.join(datapath)).T
files = [os.path.join(filepath, f) for f in data.index]
labels = data.label.values
x_train, x_test, y_train, y_test = train_test_split(
    files, labels, test_size=0.2)
class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
for k, v in zip(np.unique(y_train), class_weights):
    print(k, v)
y_train = list(map(lambda x: 0 if x == 'REAL' else 1, y_train))
y_test = list(map(lambda x: 0 if x == 'REAL' else 1, y_test))
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)
print(len(x_train), len(y_train), len(x_test), len(y_test))


# %%

# @tf.function
def input_fn(files, labels, segment_size=5, batch_size=1):
    def dataGenerator():
        for f, label in zip(files, labels):
            dp = DataPrep(segment_size=segment_size)
            frames = dp.prepFullFrames(filepath=f)
            flows = dp.getOpticalFlows()
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
                "rgb_input": (segment_size, 256, 256, 3),
                "flow_input": (segment_size - 1, 256, 256, 2)
            },
            (2,))
    )
    dataset = dataset.batch(batch_size)
    return dataset


# %%
batch_size = 10
train_data = input_fn(x_train, y_train, batch_size=batch_size)
test_data = input_fn(x_test, y_test, batch_size=batch_size)


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
        self.convLstm2 = ConvLSTM2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            data_format='channels_last',
            return_sequences=False,
            dropout=0.5
        )
        self.bn = BatchNormalization()
        self.act = LeakyReLU()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = Dense(512)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.out_layer = Dense(2)

    # TODO: specify different behavior for training, i.e. dropout only when
    # training
    def call(self, input_tensor, training=False):
        x = self.convLstm1(input_tensor)
        x = self.bn(x)
        x = self.convLstm1(x)
        x = self.bn(x)
        x = self.convLstm2(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.act(x)
        x = self.dense(x)
        x = self.act(x)
        x = self.dense(x)
        x = self.act(x)
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

# %%
rgb_stream = InputStream(3, 32, 'rgb_stream')
flow_stream = InputStream(3, 32, 'flow_stream')
rgb_input = tf.keras.Input(shape=(5, 256, 256, 3), name='rgb_input')
flow_input = tf.keras.Input(shape=(4, 256, 256, 2), name='flow_input')
rgb = rgb_stream(rgb_input)
flow = flow_stream(flow_input)
final_average = tf.keras.layers.average([rgb, flow])
x = tf.keras.layers.Flatten()(final_average)
final_output = Dense(2, activation='softmax', name='final_output')(x)
model = Model(
    inputs={"rgb_input": rgb_input, "flow_input": flow_input},
    outputs=final_output,
    name='my_model'
)
model.summary()

# %%
tf.keras.utils.plot_model(
    model,
    to_file='model.png',
    show_shapes=True,
    show_layer_names=True,
)


# %%
opt = tf.keras.optimizers.Adam()
model.compile(
    optimizer=opt,
    loss='categorical_crossentropy',
    metrics=['acc'])
model.fit(
    train_data,
    epochs=25,
    verbose=1,
    class_weight=class_weights
)


# %%
model.evaluate(
    test_data,
    #     class_weight=class_weights
)


# %%


# %%
# DEBUGGING
dataset = input_fn(filepath, batch_size=1)
start = time.time()
for i, x in enumerate(dataset):
    print(i)
    data_dict, label = x
    for k, v in data_dict.items():
        print(k, v.numpy().shape)
    print(time.time() - start)


# %%
filepath = 'data/train_sample_videos'
segment_size = 5
datapath = os.path.join(filepath, 'metadata.json')
data = pd.read_json(os.path.join(datapath)).T
files = [os.path.join(filepath, f) for f in data.index]
labels = data.label.apply(lambda x: 0 if x == 'REAL' else 1)
labels = to_categorical(labels, num_classes=2)
for i, (f, label) in enumerate(zip(files, labels)):
    try:
        dp = DataPrep(segment_size=segment_size)
        frames = dp.prepFullFrames(filepath=f)
        flows = dp.getOpticalFlows()
    except Exception as e:
        print(i, f, label)
        print(e)
        break
print('everything is working now')


# %%
f = 'data/train_sample_videos/adhsbajydo.mp4'
dp = DataPrep(segment_size=segment_size)
frames = dp.prepFullFrames(filepath=f)
