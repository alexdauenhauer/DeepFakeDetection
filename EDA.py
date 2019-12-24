# %%
import os
import pickle
import random

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%

# %%


def getFacesSingleFrame(filepath, ff, fp):
    cap = cv2.VideoCapture(filepath)
    vid = []
    counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = ff.detectMultiScale(
                image=gray,
                scaleFactor=1.5,
                minNeighbors=2)
            if len(faces) < 1:
                faces = fp.detectMultiScale(
                    image=gray,
                    scaleFactor=1.3,
                    minNeighbors=1)
            rois = []
            for x, y, w, h in faces:
                face = frame[y:y + h, x:x + w, :]
                rois.append(face)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    return rois


# %%
fcPath = '/home/alex/data/opencv/data/haarcascades'
frontface = 'haarcascade_frontalface_default.xml'
profileface = 'haarcascade_profileface.xml'
ff = cv2.CascadeClassifier(os.path.join(fcPath, frontface))
fp = cv2.CascadeClassifier(os.path.join(fcPath, profileface))
datapath = '/home/alex/projects/DeepFakeDetection/data/train_sample_videos'
# vid = 'aagfhgtpmv.mp4'
vid = random.choice(os.listdir(datapath))
# face = getSingleFrame(os.path.join(datapath, vid), fc)
cap = cv2.VideoCapture(os.path.join(datapath, vid))
cap.set
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = ff.detectMultiScale(
    image=gray,
    scaleFactor=1.6,
    minNeighbors=2)
if len(faces) < 1:
    faces = fp.detectMultiScale(
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
