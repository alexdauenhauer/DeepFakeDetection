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
# def getMultiFrameFaces(filepath):
#     cap = cv2.VideoCapture(filepath)
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if ret:
#             faces = getFaces(frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         else:
#             break
#     cap.release()
#     return rois


def getFaces(frame, grayscale=True, scaleFactor=(1.5, 1.3), minNeighbors=(2, 1)):
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


def getFrameSnippet(filepath, num_frames=5):
    cap = cv2.VideoCapture(filepath)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = int(np.random.choice(range(int(frameCount)), size=1))
    frames = np.empty(
        (num_frames, frameHeight, frameWidth, 3), np.dtype('uint8'))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    j = 0
    while j < num_frames:
        ret, frames[j] = cap.read()
        j += 1
    cap.release()
    return frames
