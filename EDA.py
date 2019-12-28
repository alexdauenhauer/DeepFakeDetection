# %%
import os
import pickle
import random
import time

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skvideo.io

# %%
# def vid2array(filepath):
#     cap = cv2.VideoCapture(filepath)
#     frames = []
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if ret:
#             frames.append(frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         else:
#             break
#     cap.release()
#     frames = np.array(frames)
#     return frames


def vid2array(filepath):
    cap = cv2.VideoCapture(filepath)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    fc = 0
    ret = True

    while (fc < frameCount and ret):
        ret, buf[fc] = cap.read()
        fc += 1

    cap.release()
    return buf


def getMultiFrameFaces(filepath):
    cap = cv2.VideoCapture(filepath)
    all_faces = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            faces = getFaces(frame)
            all_faces.extend(faces)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    return all_faces


def getFaces(frame,
             grayscale=True,
             scaleFactor=(1.5, 1.3),
             minNeighbors=(2, 1)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ff_sf, fp_sf = scaleFactor
    ff_mn, fp_mn = minNeighbors
    faces = FF.detectMultiScale(
        image=gray,
        scaleFactor=ff_sf,
        minNeighbors=ff_mn)
    if len(faces) < 1:
        faces = FP.detectMultiScale(
            image=gray,
            scaleFactor=fp_sf,
            minNeighbors=fp_mn)
    rois = []
    for x, y, w, h in faces:
        face = frame[y:y + h, x:x + w, :]
        rois.append(face)

    return rois
# %%


# fcPath = '/home/alex/data/opencv/data/haarcascades'
# frontface = 'haarcascade_frontalface_default.xml'
# profileface = 'haarcascade_profileface.xml'
# FF = cv2.CascadeClassifier(os.path.join(fcPath, frontface))
# FP = cv2.CascadeClassifier(os.path.join(fcPath, profileface))
datapath = '/home/alex/projects/DeepFakeDetection/data/train_sample_videos'
# vid = 'aagfhgtpmv.mp4'
vid = os.path.join(datapath, random.choice(os.listdir(datapath)))
start = time.time()
frames = vid2array(os.path.join(datapath, vid))
print(time.time() - start)
print(frames.shape)
# %%
start = time.time()
videodata = skvideo.io.vread(vid)
print(time.time() - start)
print(videodata.shape)

# %%
vid
# %%
datapath = '/home/alex/projects/DeepFakeDetection/data/train_sample_videos'
vid = os.path.join(datapath, random.choice(os.listdir(datapath)))
start = time.time()
frames = vid2array(os.path.join(datapath, vid))
print("read all frames runtime: ", time.time() - start)
print(frames.shape)

# start = time.time()
# print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print(cap.get(cv2.CAP_PROP_FPS))
# print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print(time.time() - start)
# %%
% % timeit
datapath = '/home/alex/projects/DeepFakeDetection/data/train_sample_videos'
vid = os.path.join(datapath, random.choice(os.listdir(datapath)))
start = time.time()
cap = cv2.VideoCapture(os.path.join(datapath, vid))
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
start_frame = int(np.random.choice(range(int(frameCount)), size=1))
num_frames = 10
frames = np.empty((num_frames, frameHeight, frameWidth, 3), np.dtype('uint8'))
i = 0
j = 0
while j < num_frames:
    ret, frame = cap.read()
    if i != start_frame:
        i += 1
        continue
    else:
        frames[j] = frame
        j += 1
cap.release()
print(f"read {num_frames} frames runtime: ", time.time() - start)
print(frames.shape)

# %%


def getFrameSnippet(filepath, num_frames=10):
    cap = cv2.VideoCapture(filepath)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = int(np.random.choice(range(int(frameCount)), size=1))
    frames = np.empty((num_frames, frameHeight, frameWidth, 3))
    flows = np.empty((num_frames - 1, frameHeight, frameWidth, 2))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    j = 0
    while j < num_frames:
        ret, frame = cap.read()
        frames[j] = frame
        # if j > 0:
        # flows[i-1] = cv2.calcOpticalFlowFarneback(
        #     prvs,
        #     frame,
        #     flow=None,
        #     pyr_scale=0.5,
        #     levels=3,
        #     winsize=15,
        #     iterations=3,
        #     poly_n=5,
        #     poly_sigma=1.1,
        #     flags=0)
        # flows[i - 1] = cv2.calcOpticalFlowFarneback(
        #     prvs, frame, None, 0.5, 3, 15, 3, 5, 1.1, 0)
        # prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        j += 1
    cap.release()
    return frames, flows

# %%


# %%
help(cv2.calcOpticalFlowFarneback)

# %%
# % % timeit
datapath = '/home/alex/projects/DeepFakeDetection/data/train_sample_videos'
vid = os.path.join(datapath, random.choice(os.listdir(datapath)))
start = time.time()
frames, flows = getFrameSnippet(vid)
print(f"read {num_frames} frames runtime: ", time.time() - start)
print(frames.shape, flows.shape)


# %%

# %%
datapath = '/home/alex/projects/DeepFakeDetection/data/train_sample_videos'
vid = os.path.join(datapath, random.choice(os.listdir(datapath)))
start = time.time()
frames = getFrameSnippet(vid, num_frames=5)
print(f"read {num_frames} frames runtime: ", time.time() - start)
print(frames.shape)
start = time.time()
prvs = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
flows = np.empty((frames.shape[0], frames.shape[1], frames.shape[2], 2))
for i in range(1, int(frames.shape[0])):
    frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
    flows[i - 1] = cv2.calcOpticalFlowFarneback(
        prvs, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
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
