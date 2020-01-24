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
from tqdm import tqdm

from DataPrep import DataPrep
pd.options.display.max_rows = 1000
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

# TODO: rewrite using dlib
# TODO: add logic for when num_faces == 0 to equalize the histogram
# TODO: add logic for when num_faces is still == 0 to grab the full frame
# TODO: add logic for when num_faces > 1 to select the largest face


def getFaces(frame, sf, mn):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.equalizeHist(frame)
    faces_rois = FF.detectMultiScale(
        image=frame,
        scaleFactor=sf,
        minNeighbors=mn
    )
    # if len(faces_rois) < 1:
    #     faces_rois = FP.detectMultiScale(
    #         image=frame,
    #         scaleFactor=fpScaleFactor,
    #         minNeighbors=fpMinNeighbors
    #     )
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
fd = dlib.get_frontal_face_detector()
idx = np.random.choice(metadata['Unnamed: 0'].values, 40, replace=False)
# fcv = []
fdl = []
# sf = 1.2
# mn = 7
for i in tqdm(metadata['Unnamed: 0'].values):
    # for i in tqdm(idx):
    vid = os.path.join(datapath, i)
    frames = getFrameSnippet(vid, 0)
    frame = frames[0]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = cv2.equalizeHist(frame)
    # start = time.time()
    # faces1 = getFaces(frame, sf, mn)
    # fcv.append(len(faces1))
    # print(f"cv2 runtime: {time.time() - start}")
    # print(f'cv2 number of faces {len(faces1)}')
    # start = time.time()
    faces2 = fd(frame, 1)
    if len(faces2) < 1:
        frame = cv2.equalizeHist(frame)
        faces2 = fd(frame, 1)
    fdl.append(len(faces2))
    # print(f"dlib runtime: {time.time() - start}")
    # print(f'dlib number of faces {len(faces2)}')
# for a, b in zip(fcv, fdl):
#     print(a, b)
# fcv = np.array(fcv)
# print('\n',np.mean(fcv), np.min(fcv), np.max(fcv), np.sum(fcv == 1), np.sum(fcv == 0))
# fdl = np.array(fdl)
# print('\n',np.mean(fdl), np.min(fdl), np.max(fdl), np.sum(fdl == 1), np.sum(fdl == 0))
# %%
# metadata['cv2_faces'] = fcv
# metadata['dlib_faces'] = fdl
# metadata.to_csv('metadata.csv')

# %%
# metadata = pd.read_csv('metadata.csv')
metadata['dlib_new2'] = fdl
# metadata
# %%
cols = [c for c in metadata.columns if 'cv2' in c or 'dlib' in c]
df = metadata.loc[:, cols]
df.describe()
# %%
for c in df.columns:
    print(c, np.sum(df[c].values == 1), np.sum(df[c].values == 0))

# %%
