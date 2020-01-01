# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import pickle
import random
import time
import json

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from DataPrep import DataPrep


# %%
# for _ in range(10):
dp = DataPrep(segment_size=5)
datapath = '/home/alex/projects/DeepFakeDetection/data/train_sample_videos'
vid = os.path.join(datapath, random.choice(os.listdir(datapath)))
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
metadata = pd.read_json('data/train_sample_videos/metadata.json').T
metadata.head()


# %%
from sklearn.utils.class_weight import compute_class_weight


# %%
class_weights = compute_class_weight('balanced', np.unique(
    metadata.label.values), metadata.label.values)
for k, v in zip(np.unique(metadata.label.values), class_weights):
    print(k, v)


# %%
import tensorflow as tf


# %%
rgb.shape[1:]


# %%
rgb_input = tf.keras.Input(shape=rgb.shape[1:])
flow_input = tf.keras.Input(shape=flow.shape[1:])


# %%
x = tf.keras.layers.Convolution2D(
    filters=rgb.shape[0],

)


# %%
counter = 0
for k, v in labels.items():
    if v['split'] == 'test':
        counter += 1
counter


# %%


# %%


# %%
fcPath = '/home/alex/data/opencv/data/haarcascades'

frontface = 'haarcascade_frontalface_default.xml'
profileface = 'haarcascade_profileface.xml'
FF = cv2.CascadeClassifier(os.path.join(fcPath, frontface))
FP = cv2.CascadeClassifier(os.path.join(fcPath, profileface))


# %%
datapath = '/home/alex/projects/DeepFakeDetection/data/train_sample_videos'
vid = os.path.join(datapath, random.choice(os.listdir(datapath)))
num_frames = 5
start = time.time()
frames = getFrameSnippet(vid, num_frames=num_frames)
flows = getOpticalFlows(frames)
rgb_rois = []
flow_rois = []
for i in range(int(frames.shape[0])):
    frame = frames[i]
    if i > 0:
        flow = flows[i - 1]
        rgb_faces, flow_faces = getFaces(frame, flow=flow)
    else:
        rgb_faces, flow_faces = getFaces(frame)
    rgb_rois.extend(rgb_faces)
    flow_rois.extend(flow_faces)
print(f"read {num_frames} frames runtime: ", time.time() - start)
print(frames.shape, flows.shape, len(rgb_rois), len(flow_rois))


# %%
for roi in rgb_rois:
    print(roi.shape)
for roi in flow_rois:
    print(roi.shape)


# %%


# %%
datapath = '/home/alex/projects/DeepFakeDetection/data/train_sample_videos'
vid = os.path.join(datapath, random.choice(os.listdir(datapath)))
num_frames = 5
start = time.time()
frames = getFrameSnippet(vid, num_frames=num_frames)
print(f"read {num_frames} frames runtime: ", time.time() - start)
print(frames.shape)
start = time.time()
prvs = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
flows = np.empty((frames.shape[0] - 1, frames.shape[1], frames.shape[2], 2))
for i in range(1, int(frames.shape[0])):
    frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prvs, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    print(flow.shape)
print("flow calculation runtime: ", time.time() - start)


# %%
frames = getFrameSnippet(vid)


# %%
dir(cv2)


# %%


# %%
cap.set
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = FF.detectMultiScale(
    image=gray,
    scaleFactor=1.6,
    minNeighbors=2)
if len(faces) < 1:
    faces = FP.detectMultiScale(
        image=gray,
        scaleFactor=1.3,
        minNeighbors=1)
# faces = list(facesfront) + list(facesprofile)
# face = sorted(faces, key=lambda x: -x[2])[0]
# x,y,w,h = face

fig, ax = plt.subplots()
ax.imshow(frame)
for x, y, w, h in faces:
    rect = patches.Rectangle(
        (x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)


# %%
rois = []
for x, y, w, h in faces:
    face = frame[y:y + h, x:x + w, :]
    rois.append(face)
for roi in rois:
    plt.imshow(roi)


# %%
len(rois)


# %%
plt.imshow(face)


# %%
face


# %%


# %%
folders = []
for dirname, subdirs, files in os.walk('/home/alex/anaconda3'):
    if 'opencv' in dirname.lower():
        folders.append(dirname)
        break
folders


# %%


# _, frame = cap.read()

# # # convert to grayscale
# # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# # detect faces
# faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# # for each detected face
# for (x, y, w, h) in faces:
#     # create rectangle around it
#     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

#     # crop the ROI
#     roi = gray[y:y + h, x:x + w]

#     # encode as png, convert to bytes
#     _, img = cv2.imencode('.png', roi)
#     msg = img.tobytes()

#     # publish the message
#     client.publish(topic, msg, 0)

#     # record that a face has been captured
#     counter += 1
