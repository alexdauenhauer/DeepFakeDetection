# %%
import os
import pickle
import sys
import time
from os.path import abspath, dirname

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from dlib import get_frontal_face_detector
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
# pylint: disable=import-error
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

from_generator = tf.data.Dataset.from_generator
np.random.seed(666)


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
            start_frame = np.random.choice(range(int(frameCount)), size=1)[0]
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

def input_fn(files, labels=None, segment_size=5, batch_size=1, rsz=(128, 128)):
    def dataGenerator():
        if labels is not None:
            for f, label in zip(files, labels):
                dp = DataPrep(segment_size=segment_size, rsz=rsz)
                frames, flows = dp.prepVid(filepath=f)
                yield {'rgb_input': frames, 'flow_input': flows}, label
        else:
            for f in files:
                dp = DataPrep(segment_size=segment_size, rsz=rsz)
                frames, flows = dp.prepVid(filepath=f)
                label = np.array([0, 0])
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


# def predict(model, test)
# %%


def main():
    # grab training data
    filepath = 'data/train_sample_videos'
    datapath = os.path.join(filepath, 'metadata.json')
    data = pd.read_json(datapath).T
    # files = [os.path.join(filepath, f) for f in data.index]
    # labels = data.label.values
    files = [os.path.join(filepath, f) for f in data.index][:20]
    labels = data.label.values[:20]
    x_train, x_test, y_train, y_test = train_test_split(
        files, labels, test_size=0.2)
    class_weights = compute_class_weight(
        'balanced', np.unique(y_train), y_train)
    for k, v in zip(np.unique(y_train), class_weights):
        print(k, v)
    y_train = list(map(lambda x: 0 if x == 'REAL' else 1, y_train))
    y_test = list(map(lambda x: 0 if x == 'REAL' else 1, y_test))
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)
    print(len(x_train), len(y_train), len(x_test), len(y_test))

    # validation data
    val_path = 'data/test_videos'
    val_files = [os.path.join(val_path, f) for f in os.listdir(val_path)]
    print('number of validation files', len(val_files))

    # generate datasets
    batch_size = 4
    segment_size = 2
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
    val_data = input_fn(
        files=val_files,
        segment_size=segment_size,
        batch_size=batch_size,
        rsz=rsz)
    rgb_input = tf.keras.Input(
        shape=(segment_size, rsz[0], rsz[1], 3),
        name='rgb_input')
    flow_input = tf.keras.Input(
        shape=(segment_size - 1, rsz[0], rsz[1], 2),
        name='flow_input')

    # TODO: make OO
    # RGB MODEL
    # block 1
    x = layers.Conv3D(
        filters=8,
        kernel_size=3,
        strides=(1, 1, 1),
        padding='same',
        data_format='channels_last',
        activation='relu',
    )(rgb_input)
    x = layers.Conv3D(
        filters=8,
        kernel_size=4,
        strides=(1, 1, 1),
        padding='same',
        data_format='channels_last',
        activation='relu',
    )(x)
    block1_output = layers.MaxPool3D(
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        padding='same'
    )(x)
    # block 2
    x = layers.Conv3D(
        filters=8,
        kernel_size=3,
        strides=(1, 1, 1),
        padding='same',
        data_format='channels_last',
        activation='relu',
    )(block1_output)
    x = layers.Conv3D(
        filters=8,
        kernel_size=4,
        strides=(1, 1, 1),
        padding='same',
        data_format='channels_last',
        activation='relu',
    )(x)
    block2_output = layers.add([x, block1_output])
    # block 3
    x = layers.Conv3D(
        filters=8,
        kernel_size=3,
        strides=(1, 1, 1),
        padding='same',
        data_format='channels_last',
        activation='relu',
    )(block2_output)
    x = layers.Conv3D(
        filters=8,
        kernel_size=4,
        strides=(1, 1, 1),
        padding='same',
        data_format='channels_last',
        activation='relu',
    )(x)
    block3_output = layers.add([x, block2_output])

    x = layers.Conv3D(
        filters=8,
        kernel_size=3,
        strides=(1, 1, 1),
        padding='same',
        data_format='channels_last',
        activation='relu',
    )(block3_output)
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    rgb_outputs = layers.Dense(2, activation='softmax')(x)

    rgb_model = Model(inputs=rgb_input, outputs=rgb_outputs)
    rgb_model.summary()

    # FLOW MODEL
    x = layers.ConvLSTM2D(
        filters=8,
        kernel_size=3,
        strides=1,
        padding='same',
        data_format='channels_last',
        return_sequences=True,
        dropout=0.5
    )(flow_input)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=8,
        kernel_size=3,
        strides=1,
        padding='same',
        data_format='channels_last',
        return_sequences=True,
        dropout=0.5
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=8,
        kernel_size=3,
        strides=1,
        padding='same',
        data_format='channels_last',
        return_sequences=False,
        dropout=0.5
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    flow_output = layers.Dense(2)(x)
    flow_model = Model(inputs=flow_input, outputs=flow_output)
    flow_model.summary()

    # FINAL MODEL
    final_average = layers.average([rgb_outputs, flow_output])
    x = layers.Flatten()(final_average)
    final_output = layers.Dense(
        2, activation='softmax', name='final_output')(x)
    model = Model(
        inputs={"rgb_input": rgb_input, "flow_input": flow_input},
        outputs=final_output,
        name='my_model'
    )
    model.summary()

    # tf.keras.utils.plot_model(
    #     model,
    #     to_file='model.png',
    #     show_shapes=True,
    #     show_layer_names=True
    # )

    # TRAIN
    opt = tf.keras.optimizers.Adam()
    save_path = 'data/model_checkpoints/ckpt'
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_path,
        save_best_only=False,
        save_weights_only=True
    )
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['acc'])
    model.fit(
        x=train_data.repeat(),
        validation_data=test_data.repeat(),
        epochs=1,
        verbose=1,
        class_weight=class_weights,
        steps_per_epoch=len(x_train) // batch_size,
        validation_steps=len(x_test) // batch_size,
        callbacks=[ckpt]
    )

    # EVAL
    print('\n\n---------------------------------------------------------')
    print('predicting on validation data')
    start = time.time()
    preds = model.predict(val_data)
    print('prediction time: ', time.time() - start)
    preds = np.argmax(preds, axis=1)
    df = pd.DataFrame([val_files, preds], columns=['filename', 'label'])
    df.to_csv('data/submission.csv')


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    main()
