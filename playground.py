# %%
import os
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import (BatchNormalization, ConvLSTM2D, Dense,
                                     LeakyReLU, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

from DataPrep import DataPrep

# %%
dp = DataPrep(segment_size=5)
start = time.time()
dp.prepFullFrames()
dp.getOpticalFlows()
print(time.time() - start)
print(dp.filepath)
print(dp.frames.shape, dp.flows.shape, dp.labels.shape)

# %%
metadata = pd.read_json('data/train_sample_videos/metadata.json').T
metadata.shape
# %%
metadata.head()

# %%
vid = random.choice(os.listdir(dp.datapath))
metadata.at[vid, 'label']
# %%

# %%
class_weights = compute_class_weight('balanced', np.unique(
    metadata.label.values), metadata.label.values)
for k, v in zip(np.unique(metadata.label.values), class_weights):
    print(k, v)
# %%


def count(stop):
    i = 0
    while i < stop:
        yield i
        i += 1


ds_counter = tf.data.Dataset.from_generator(
    count, args=[25], output_types=tf.int32, output_shapes=(), )

for count_batch in ds_counter.repeat().batch(10).take(10):
    print(count_batch.numpy())

# %%


def generateFrames(datapath=None):
    if not datapath:
        datapath = '/home/alex/projects/DeepFakeDetection/data/train_sample_videos'
    vids = [os.path.join(datapath, v) for v in os.listdir(datapath)]
    for v in vids:
        dp = DataPrep(segment_size=5)
        yield dp.prepVid(frame_name=v)


vids = tf.data.Dataset.from_generator(
    generateVids,
    output_types=(tf.int32, tf.int32),
    output_shapes=((5, 256, 256, 3), (4, 256, 256, 2))
)

# %%
start = time.time()
for x in vids.batch(2).take(5):
    print(x[0].numpy().shape, x[1].numpy().shape)
    print(time.time() - start)


# %%
vids.shape
# %%

# %%

# %%
filepath = 'data/train_sample_videos'


def input_fn(filepath, segment_size=5):
    if 'metadata.json' in os.listdir(filepath):
        datapath = os.path.join(filepath, 'metadata.json')
        data = pd.read_json(os.path.join(datapath)).T
        files = [os.path.join(filepath, f) for f in os.listdir(filepath)
                 if 'metadata.json' not in f]
    data['categorical'] = data.label.apply(lambda x: 0 if x == 'REAL' else 1)
    labels = to_categorical(data.categorical, num_classes=2)
    # for f, l in zip(files, labels):
    #     dp = DataPrep(segment_size=5)
    #     frames = dp.prepFullFrames()
    #     flows = dp.getOpticalFlows()
    #     filename = dp.filepath.split('/')[-1]
    #     print("red_input", frames[:, :, :, 0])
    #     print("green_input", frames[:, :, :, 1])
    #     print("blue_input", frames[:, :, :, 2])
    #     print("x_flow_input", flows[:, :, :, 0])
    #     print("y_flow_input", flows[:, :, :, 1],)
    #     print(l)

    def dataGenerator():
        for f, l in zip(files, labels):
            dp = DataPrep(segment_size=segment_size)
            frames = dp.prepFullFrames()
            flows = dp.getOpticalFlows()
            filename = dp.filepath.split('/')[-1]
            yield {
                "red_input": frames[:, :, :, 0],
                "green_input": frames[:, :, :, 1],
                "blue_input": frames[:, :, :, 2],
                "x_flow_input": flows[:, :, :, 0],
                "y_flow_input": flows[:, :, :, 1],
            }, l
    dataset = tf.data.Dataset.from_generator(
        dataGenerator,
        output_types=(
            {
                "red_input": tf.int8,
                "green_input": tf.int8,
                "blue_input": tf.int8,
                "x_flow_input": tf.float32,
                "y_flow_input": tf.float32,
            },
            tf.int8)
    )
    dataset = dataset.batch(1)
    return dataset


dataset = input_fn(filepath)
start = time.time()
for x in dataset.take(5):
    data_dict, label = x
    print(label.numpy())
    for k, v in data_dict.items():
        print(k, v.numpy().shape)
    print(time.time() - start)


# %%

# %%


# %%
rgb_input = tf.keras.Input(shape=(5, 256, 256, 3), name='rgb_input')
flow_input = tf.keras.Input(shape=(4, 256, 256, 3), name='flow_input')
x = ConvLSTM2D(
    filters=4,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same',
    return_sequences=False,
    dropout=0.5
)(rgb_input)
x = LeakyReLU()(x)
x = BatchNormalization()(x)
rgb_outputs = Dense(1, activation='sigmoid', name='rgb_outputs')
flow_outputs = Dense(1, activation='sigmoid', name='flow_outputs')


# %%


# %%
inputs = tf.keras.Input(shape=rgb.shape, name='inputs')
x = tf.keras.layers.Conv3D(
    filters=32,
    kernel_size=(3, 3, 3),
    padding='same',
    data_format='channels_last',
    activation='relu')(inputs)
x.shape
x = tf.keras.layers.Conv3D(64, 3, activation='relu')(x)
block_1_output = tf.keras.layers.MaxPooling2D(3)(x)

x = tf.keras.layers.Conv3D(64, 3, activation='relu',
                           padding='same')(block_1_output)
x = tf.keras.layers.Conv3D(64, 3, activation='relu', padding='same')(x)
block_2_output = tf.keras.layers.add([x, block_1_output])

x = tf.keras.layers.Conv3D(64, 3, activation='relu',
                           padding='same')(block_2_output)
x = tf.keras.layers.Conv3D(64, 3, activation='relu', padding='same')(x)
block_3_output = tf.keras.layers.add([x, block_2_output])

x = tf.keras.layers.Conv3D(64, 3, activation='relu')(block_3_output)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs, name='toy_resnet')
model.summary()
