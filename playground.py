# %%
import os
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from DataPrep import DataPrep

# %%
dp = DataPrep(segment_size=5)
datapath = '/home/alex/projects/DeepFakeDetection/data/train_sample_videos'
# vid = os.path.join(datapath, random.choice(os.listdir(datapath)))
frame_name = random.choice(os.listdir(datapath))
vid = os.path.join(datapath, frame_name)
start = time.time()
frames = dp.getFrameSnippet(vid, start_frame=0)
flows = dp.getOpticalFlows(frames)
rgb_rois = []
flow_rois = []
for i in range(int(frames.shape[0])):
    frame = frames[i]
    if i > 0:
        flow = flows[i - 1]
        rgb_faces, flow_faces = dp.getFaces(frame, flow=flow)
    else:
        rgb_faces, flow_faces = dp.getFaces(frame)
    rgb_rois.extend(rgb_faces)
    flow_rois.extend(flow_faces)
flow_rois = [r for r in flow_rois if r is not None]
rgb = np.stack(rgb_rois)
flow = np.stack(flow_rois)
print(f"read {dp.segment_size} frames runtime: ", time.time() - start)
print(frames.shape, flows.shape, len(rgb_rois), len(flow_rois))

# %%
dp = DataPrep(segment_size=5)
rgb, flow = dp.getRandomVid(frame_name=frame_name, start_frame=0)
# %%
rgb.shape, flow.shape
# %%

# %%
metadata = pd.read_json('data/train_sample_videos/metadata.json').T
metadata.head()

# %%
class_weights = compute_class_weight('balanced', np.unique(
    metadata.label.values), metadata.label.values)
for k, v in zip(np.unique(metadata.label.values), class_weights):
    print(k, v)
# %%
def count(stop):
  i = 0
  while i<stop:
    yield i
    i += 1

ds_counter = tf.data.Dataset.from_generator(count, args=[25], output_types=tf.int32, output_shapes = (), )

for count_batch in ds_counter.repeat().batch(10).take(10):
  print(count_batch.numpy())

# %%
def generateVids(datapath=None, stop=5):
    if not datapath:
        datapath = '/home/alex/projects/DeepFakeDetection/data/train_sample_videos'
    while i < stop:
        dp = DataPrep(segment_size=5)
        yield dp.getRandomVid()

start = time.time()
vids = tf.data.Dataset.from_generator(
    generateVids,
    # args=[None, 20]
    output_types=(tf.int32, tf.int32),
    output_shapes=((None,128,128,3), (None,128,128,2))
    )
print(time.time() - start)

# %%
for x in vids.batch(1).take(20):
    print(x[0].numpy().shape, x[1].numpy().shape)


# %%

# %%

# %%
rgb_input = tf.keras.Input(shape=rgb.shape)
flow_input = tf.keras.Input(shape=flow.shape)
print(rgb_input.shape, flow.shape)

# %%
x = tf.keras.layers.ConvLSTM2D(
    filters=64,
    kernel_size=(3,3),
    strides=(1,1),
    padding='same',
    return_sequences=False,
    dropout=0.5
)(rgb_input)
x.shape


# %%
inputs = tf.keras.Input(shape=rgb.shape, name='inputs')
x = tf.keras.layers.Conv3D(
    filters=32,
    kernel_size=(3,3,3),
    padding='same',
    data_format='channels_last',
    activation='relu')(inputs)
x.shape
x = tf.keras.layers.Conv3D(64, 3, activation='relu')(x)
block_1_output = tf.keras.layers.MaxPooling2D(3)(x)

x = tf.keras.layers.Conv3D(64, 3, activation='relu', padding='same')(block_1_output)
x = tf.keras.layers.Conv3D(64, 3, activation='relu', padding='same')(x)
block_2_output = tf.keras.layers.add([x, block_1_output])

x = tf.keras.layers.Conv3D(64, 3, activation='relu', padding='same')(block_2_output)
x = tf.keras.layers.Conv3D(64, 3, activation='relu', padding='same')(x)
block_3_output = tf.keras.layers.add([x, block_2_output])

x = tf.keras.layers.Conv3D(64, 3, activation='relu')(block_3_output)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs, name='toy_resnet')
model.summary()
