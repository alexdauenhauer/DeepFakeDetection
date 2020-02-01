# %%
import os
import pickle
import sys
import time
from os.path import abspath, dirname

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import (BatchNormalization, Conv3D, ConvLSTM2D,
                                     Dense, Input, LeakyReLU)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# ROOT_DIR = dirname(abspath(__file__))
# if ROOT_DIR not in sys.path:
#     sys.path.append(ROOT_DIR)

# from DataPrep import DataPrepDlib

from_generator = tf.data.Dataset.from_generator

# %%


class DataPrep():

    def __init__(self, segment_size=5, rsz=(128, 128)):
        self.fd = get_frontal_face_detector()
        self.segment_size = segment_size
        self.frames = None
        self.flows = None
        self.rsz = rsz

    def getFrameSnippet(self, filepath, start_frame=None):
        cap = cv2.VideoCapture(filepath)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not start_frame:
            start_frame = choice(range(int(frameCount)))
        if frameCount - start_frame < self.segment_size:
            start_frame = 0
        self.frames = np.empty(
            (self.segment_size, frameHeight, frameWidth, 3), dtype=np.uint8)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        j = 0
        while j < self.segment_size:
            ret, self.frames[j] = cap.read()
            j += 1
        cap.release()

    def getOpticalFlows(self):
        if self.frames is not None:
            self.flows = np.empty(
                (self.frames.shape[0] - 1,
                 self.frames.shape[1],
                 self.frames.shape[2],
                 2))
            prvs = cv2.cvtColor(
                self.frames[0].astype(np.uint8), cv2.COLOR_BGR2GRAY)
            for i in range(1, int(self.frames.shape[0])):
                frame = cv2.cvtColor(
                    self.frames[i].astype(np.uint8), cv2.COLOR_BGR2GRAY)
                self.flows[i - 1] = cv2.calcOpticalFlowFarneback(
                    prvs, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                prvs = frame

    def resize(self, frame):
        # TODO: will want to test different sizes here as a hyperparameter
        height, width = self.rsz
        return cv2.resize(frame, (height, width))

    def getFaces(self, frame, grayscale=True):
        orig_frame = frame
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.fd(frame, 0)
        if len(faces) < 1:
            frame = cv2.equalizeHist(frame)
            faces = self.fd(frame, 0)
        if len(faces) < 1:
            faces = orig_frame
        return faces

    def getFaceRois(self, frame, faces):
        if isinstance(faces, np.ndarray):
            return self.resize(frame)
        f = faces[0]
        h = f.bottom() - f.top()
        face_rois = None
        for face in faces:
            x, y, r = f.left(), f.top(), f.right()
            w = r - x
            roi = frame[y:y + h, x:x + w, :]
            if face_rois is None:
                face_rois = roi
            else:
                face_rois = np.hstack((face_rois, roi))
        face_rois = self.resize(face_rois)
        return face_rois

    def prepVid(self, filepath, start_frame=None):
        self.getFrameSnippet(filepath, start_frame)
        self.getOpticalFlows()
        w, h = self.rsz
        rgb_rois = np.empty((self.segment_size, w, h, 3), dtype=np.int8)
        flow_rois = np.empty(
            (self.segment_size - 1, w, h, 2), dtype=np.float32)
        for i, frame in enumerate(self.frames):
            faces = self.getFaces(frame)
            rois = self.getFaceRois(frame, faces)
            rgb_rois[i] = rois
            if i == 0:
                continue
            else:
                flow = self.flows[i - 1]
                rois = self.getFaceRois(flow, faces)
                flow_rois[i - 1] = rois
        return rgb_rois, flow_rois

    def prepFullFrames(self, filepath, start_frame=None):
        self.getFrameSnippet(filepath, start_frame)
        self.getOpticalFlows()
        w, h = self.rsz
        rgb_rois = np.empty((self.segment_size, w, h, 3), dtype=np.int8)
        flow_rois = np.empty(
            (self.segment_size - 1, w, h, 2), dtype=np.float32)
        for i, frame in enumerate(self.frames):
            rois = self.resize(frame)
            rgb_rois[i] = rois
            if i == 0:
                continue
            else:
                flow = self.flows[i - 1]
                rois = self.resize(flow)
                flow_rois[i - 1] = rois
        return rgb_rois, flow_rois


# %%

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
rgb_input = tf.keras.Input(
    shape=(segment_size, rsz[0], rsz[1], 3),
    name='rgb_input')
flow_input = tf.keras.Input(
    shape=(segment_size - 1, rsz[0], rsz[1], 2),
    name='flow_input')
# %%
x = Conv3D(
    filters=8,
    kernel_size=3,
    strides=(1, 1, 1),
    padding='same',
    data_format='channels_last',
    activation='relu',
)(rgb_input)
x = Conv3D(
    filters=8,
    kernel_size=4,
    strides=(1, 1, 1),
    padding='same',
    data_format='channels_last',
    activation='relu',
)(x)
block1_output = tf.keras.layers.MaxPool3D(
    pool_size=(2, 2, 2),
    strides=(2, 2, 2),
    padding='same'
)

x = Conv3D(
    filters=8,
    kernel_size=3,
    strides=(1, 1, 1),
    padding='same',
    data_format='channels_last',
    activation='relu',
)(block1_output)
x = Conv3D(
    filters=8,
    kernel_size=4,
    strides=(1, 1, 1),
    padding='same',
    data_format='channels_last',
    activation='relu',
)(x)
block2_output = tf.keras.layers.add([x, block1_output])

x = Conv3D(
    filters=8,
    kernel_size=3,
    strides=(1, 1, 1),
    padding='same',
    data_format='channels_last',
    activation='relu',
)(block2_output)
x = Conv3D(
    filters=8,
    kernel_size=4,
    strides=(1, 1, 1),
    padding='same',
    data_format='channels_last',
    activation='relu',
)(x)
block3_output = tf.keras.layers.add([x, block2_output])

x = Conv3D(
    filters=8,
    kernel_size=3,
    strides=(1, 1, 1),
    padding='same',
    data_format='channels_last',
    activation='relu',
)(block3_output)
x = tf.keras.layers.GlobalAveragePooling()(x)
x = Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = Dense(2, activation='softmax')(x)

model = Model(inputs=rgb_input, outputs=outputs)
model.summary()
# %%
