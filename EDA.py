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

from DataPrep import DataPrep


# %%


# def getFaces(frame,
#              flow=None,
#              grayscale=True,
#              scaleFactor=(1.5, 1.3),
#              minNeighbors=(2, 1)):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     ff_sf, fp_sf = scaleFactor
#     ff_mn, fp_mn = minNeighbors
#     faces_rois = FF.detectMultiScale(
#         image=gray,
#         scaleFactor=ff_sf,
#         minNeighbors=ff_mn
#     )
#     if len(faces_rois) < 1:
#         faces_rois = FP.detectMultiScale(
#             image=gray,
#             scaleFactor=fp_sf,
#             minNeighbors=fp_mn
#         )
#     rgb_faces = []
#     flow_faces = []
#     for x, y, w, h in faces_rois:
#         rgb_face = frame[y:y + h, x:x + w, :]
#         rgb_faces.append(rgb_face)
#         if flow is not None:
#             flow_face = flow[y:y + h, x:x + w, :]
#             flow_faces.append(flow_face)
#     return rgb_faces, flow_faces


# def getFrameSnippet(filepath, num_frames=10):
#     cap = cv2.VideoCapture(filepath)
#     frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     start_frame = int(np.random.choice(range(int(frameCount)), size=1))
#     frames = np.empty((num_frames, frameHeight, frameWidth, 3), dtype=np.uint8)
#     cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
#     j = 0
#     while j < num_frames:
#         ret, frames[j] = cap.read()
#         j += 1
#     cap.release()
#     return frames


# def getOpticalFlows(frames):
#     # TODO: look into cuda optical flows in cv2
#     flows = np.empty(
#         (frames.shape[0] - 1, frames.shape[1], frames.shape[2], 2))
#     prvs = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
#     for i in range(1, int(frames.shape[0])):
#         frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
#         flows[i - 1] = cv2.calcOpticalFlowFarneback(
#             prvs, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#         prvs = frame
#     return flows

# %%
# %%timeit
for _ in range(10):
    dp = DataPrep(segment_size=5)
    datapath = '/home/alex/projects/DeepFakeDetection/data/train_sample_videos'
    vid = os.path.join(datapath, random.choice(os.listdir(datapath)))
    start = time.time()
    frames = dp.getFrameSnippet(vid)
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
    print(f"read {dp.segment_size} frames runtime: ", time.time() - start)
    print(frames.shape, flows.shape, len(rgb_rois), len(flow_rois))


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
