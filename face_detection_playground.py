# %%
import os
import pickle
import random
import time

import cv2
import dlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from DataPrep import DataPrep

# %%
datapath = 'data/train_sample_videos'

metadata = pd.read_json(os.path.join(datapath, 'metadata.json')).T
metadata.head()
# %%
fcPath = '/home/alex/projects/PublicRepos/opencv/data/haarcascades'

frontface = 'haarcascade_frontalface_default.xml'
profileface = 'haarcascade_profileface.xml'
FF = cv2.CascadeClassifier(os.path.join(fcPath, frontface))
FP = cv2.CascadeClassifier(os.path.join(fcPath, profileface))
ffScaleFactor = 1.5
fpScaleFactor = 1.5
ffMinNeigbors = 2
fpMinNeighbors = 2


def getFaces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces_rois = FF.detectMultiScale(
        image=gray,
        scaleFactor=ffScaleFactor,
        minNeighbors=ffMinNeigbors
    )
    if len(faces_rois) < 1:
        faces_rois = FP.detectMultiScale(
            image=gray,
            scaleFactor=fpScaleFactor,
            minNeighbors=fpMinNeighbors
        )
    return faces_rois


def getFrameSnippet(filepath, start_frame=None):
    cap = cv2.VideoCapture(filepath)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not start_frame:
        start_frame = random.choice(range(int(frameCount)))
    if frameCount - start_frame < 5:
        start_frame = 0
    frames = np.empty((5, frameHeight, frameWidth, 3), dtype=np.uint8)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    j = 0
    while j < 5:
        ret, frames[j] = cap.read()
        j += 1
    cap.release()
    return frames


def showFaces(frame, faces, rects=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(frame)
    if rects is not None:
        for R in rects:
            x, y, r, b = R.left(), R.top(), R.right(), R.bottom()
            w = r - x
            h = b - y
            rect = patches.Rectangle(
                (x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    else:
        for x, y, w, h in faces:
            rect = patches.Rectangle(
                (x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)


# %%
vid_name = random.choice(metadata.index)
vid = os.path.join(datapath, vid_name)
print(vid)
# %%
frames = getFrameSnippet(vid, 0)
frames.shape
# %%
frame = frames[0]
frame.shape

# %%
faces = getFaces(frame)
len(faces)
# %%
showFaces(frame, faces)
# %%

# %%
# TODO: need to handle dark videos and make them brighter


# %%
fd = dlib.get_frontal_face_detector()
# %%
dir(fd)
# %%
start = time.time()
faces2 = fd(frame, 1)
len(faces2)
print(time.time() - start)

# %%
f = faces2[0]

# %%

x, y, r, b = f.left(), f.top(), f.right(), f.bottom()
w = r - x
h = b - y
print(x, y, w, h)
# %%
showFaces(frame, faces2, rects=faces2)


# %%
metadata.shape

# %%
idx = np.random.choice(metadata.index, 20, replace=False)
for i in idx:
    vid = os.path.join(datapath, i)
    frames = getFrameSnippet(vid, 0)
    frame = frames[0]
    start = time.time()
    faces1 = getFaces(frame)
    print(f"cv2 runtime: {time.time() - start}")
    print(f'cv2 number of faces {len(faces1)}')
    start = time.time()
    faces2 = fd(frame, 1)
    print(f"dlib runtime: {time.time() - start}")
    print(f'dlib number of faces {len(faces2)}')


# %%
