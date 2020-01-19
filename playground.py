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

    def dataGenerator():
        for f, l in zip(files, labels):
            dp = DataPrep(segment_size=segment_size)
            frames = dp.prepFullFrames()
            flows = dp.getOpticalFlows()
            filename = dp.filepath.split('/')[-1]
            # yield {
            #     "red_input": frames[:, :, :, 0],
            #     "green_input": frames[:, :, :, 1],
            #     "blue_input": frames[:, :, :, 2],
            #     "x_flow_input": flows[:, :, :, 0],
            #     "y_flow_input": flows[:, :, :, 1],
            # }, l
            yield {"rgb_input": frames, "flow_input": flows}, l
    # dataset = tf.data.Dataset.from_generator(
    #     dataGenerator,
    #     output_types=(
    #         {
    #             "red_input": tf.int8,
    #             "green_input": tf.int8,
    #             "blue_input": tf.int8,
    #             "x_flow_input": tf.float32,
    #             "y_flow_input": tf.float32,
    #         },
    #         tf.int8),
    #     output_shapes=(
    #         {
    #             "red_input": (segment_size, 256, 256),
    #             "green_input": (segment_size, 256, 256),
    #             "blue_input": (segment_size, 256, 256),
    #             "x_flow_input": (segment_size - 1, 256, 256),
    #             "y_flow_input": (segment_size - 1, 256, 256),
    #         },
    #         (2,))
    # )
    dataset = tf.data.Dataset.from_generator(
        dataGenerator,
        output_types=(
            {"rgb_input": tf.int8, "flow_input": tf.float32}, tf.int8),
        output_shapes=(
            {
                "rgb_input": (segment_size, 256, 256, 3),
                "flow_input": (segment_size - 1, 256, 256, 2)
            },
            (2,))
    )
    dataset = dataset.batch(1)
    return dataset


# %%
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
dataset = input_fn(filepath)
for x in dataset.take(1):
    data_dict, label = x
    print(label.numpy())
    rgb = data_dict.get('rgb_input')
    print(rgb.numpy().shape)


# %%
print(rgb.numpy().shape)
# %%
type(rgb)
# %%

# %%
# red_input = tf.keras.Input(shape=(5, 256, 256), name='red_input')
# green_input = tf.keras.Input(shape=(5, 256, 256), name='green_input')
# blue_input = tf.keras.Input(shape=(5, 256, 256), name='blue_input')
rgb_input = tf.keras.Input(shape=(5, 256, 256, 3), name='rgb_input')
inputs = rgb_input(rgb)
inputs.shape
# RGB model
# r = ConvLSTM2D(
#     filters=8,
#     kernel_size=(3, 3),
#     strides=(1, 1),
#     padding='same',
#     return_sequences=False,
#     dropout=0.5


# )(red_input)
# r = LeakyReLU()(r)
# r = BatchNormalization()(r)
# # red_output = Dense(2, activation='softmax', name='red_output')(r)
# red_output = Dense(2, name='red_output')(r)

# g = ConvLSTM2D(
#     filters=8,
#     kernel_size=(3, 3),
#     strides=(1, 1),
#     padding='same',
#     return_sequences=False,
#     dropout=0.5
# )(green_input)
# g = LeakyReLU()(g)
# g = BatchNormalization()(g)
# # green_output = Dense(2, activation='softmax', name='green_output')(g)
# green_output = Dense(2, name='green_output')(g)

x = ConvLSTM2D(
    filters=8,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same',
    data_format='channels_last',
    return_sequences=False,
    dropout=0.5
)(rgb_input)
x = LeakyReLU()(x)
x = BatchNormalization()(x)
# blue_output = Dense(2, activation='softmax', name='blue_output')(b)
rgb_output = Dense(2, name='blue_output')(x)

# rgb_output = tf.keras.layers.average([red_output, green_output, blue_output])
rgb_model = Model(inputs=rgb_input, outputs=rgb_output)
# %%
rgb_model.summary()

# %%
tf.keras.utils.plot_model(
    rgb_model,
    to_file='rgb_model.png',
    show_shapes=True,
    show_layer_names=True,
)

# %%
# x_flow_input = tf.keras.Input(shape=(4, 256, 256), name='x_flow_input')
# y_flow_input = tf.keras.Input(shape=(4, 256, 256), name='y_flow_input')
flow_input = tf.keras.Input(shape=(4, 256, 256, 2), name='flow_input')

x = ConvLSTM2D(
    filters=8,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same',
    data_format='channels_last',
    return_sequences=False,
    dropout=0.5
)(flow_input)
x = LeakyReLU()(x)
x = BatchNormalization()(x)
# flow_output = Dense(2, activation='softmax', name='flow_output')(x)
flow_output = Dense(2, name='flow_output')(x)

# y = ConvLSTM2D(
#     filters=8,
#     kernel_size=(3, 3),
#     strides=(1, 1),
#     padding='same',
#     return_sequences=False,
#     dropout=0.5
# )(y_flow_input)
# y = LeakyReLU()(y)
# y = BatchNormalization()(y)
# # y_flow_output = Dense(2, activation='softmax', name='y_flow_output')(y)
# y_flow_output = Dense(2, name='y_flow_output')(y)

# flow_output = tf.keras.layers.average([x_flow_output, y_flow_output])
flow_model = Model(inputs=flow_input, outputs=flow_output)
# %%
flow_model.summary()

# %%
tf.keras.utils.plot_model(
    flow_model,
    to_file='flow_model.png',
    show_shapes=True,
    show_layer_names=True,
)

# %%
final_average = tf.keras.layers.average([rgb_output, flow_output])
final_output = tf.keras.activations.softmax(final_average)
model = Model(
    inputs={
        "rgb_input": rgb_input,
        "flow_input": flow_input,
    },
    outputs=final_output
)
model.summary()
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
filepath = 'data/train_sample_videos'
model.fit(
    input_fn(filepath),
    epochs=2,
    verbose=1,
    class_weight=class_weights)

# %%
model.layers
# %%
for layer in model.layers:
    print(layer.shape)
# %%
# inputs = tf.keras.Input(shape=rgb.shape, name='inputs')
# x = tf.keras.layers.Conv3D(
#     filters=32,
#     kernel_size=(3, 3, 3),
#     padding='same',
#     data_format='channels_last',
#     activation='relu')(inputs)
# x.shape
# x = tf.keras.layers.Conv3D(64, 3, activation='relu')(x)
# block_1_output = tf.keras.layers.MaxPooling2D(3)(x)

# x = tf.keras.layers.Conv3D(64, 3, activation='relu',
#                            padding='same')(block_1_output)
# x = tf.keras.layers.Conv3D(64, 3, activation='relu', padding='same')(x)
# block_2_output = tf.keras.layers.add([x, block_1_output])

# x = tf.keras.layers.Conv3D(64, 3, activation='relu',
#                            padding='same')(block_2_output)
# x = tf.keras.layers.Conv3D(64, 3, activation='relu', padding='same')(x)
# block_3_output = tf.keras.layers.add([x, block_2_output])

# x = tf.keras.layers.Conv3D(64, 3, activation='relu')(block_3_output)
# x = tf.keras.layers.GlobalAveragePooling2D()(x)
# x = tf.keras.layers.Dense(256, activation='relu')(x)
# x = tf.keras.layers.Dropout(0.5)(x)
# outputs = tf.keras.layers.Dense(20, activation='softmaxx')(x)

# model = tf.keras.Model(inputs, outputs, name='toy_resnet')
# model.summary()
