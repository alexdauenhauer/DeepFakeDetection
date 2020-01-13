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


class DataPrep():
    def __init__(self, haar_cascades_path=None, segment_size=10):
        if not haar_cascades_path:
            haar_cascades_path = '/home/alex/data/opencv/data/haarcascades'
        fcPath = haar_cascades_path
        frontface = 'haarcascade_frontalface_default.xml'
        profileface = 'haarcascade_profileface.xml'
        self.ff = cv2.CascadeClassifier(os.path.join(fcPath, frontface))
        self.fp = cv2.CascadeClassifier(os.path.join(fcPath, profileface))
        self.ffScaleFactor = 1.5
        self.fpScaleFactor = 1.5
        self.ffMinNeigbors = 2
        self.fpMinNeighbors = 2
        self.segment_size = segment_size

    def getFaces(self, frame, flow=None, grayscale=True, resize=True):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_rois = self.ff.detectMultiScale(
            image=gray,
            scaleFactor=self.ffScaleFactor,
            minNeighbors=self.ffMinNeigbors
        )
        if len(faces_rois) < 1:
            faces_rois = self.fp.detectMultiScale(
                image=gray,
                scaleFactor=self.fpScaleFactor,
                minNeighbors=self.fpMinNeighbors
            )
        if len(faces_rois) == 0:
            print('no faces found')
            if resize:
                return [self.resize(frame)], [self.resize(flow)]
            else:
                return [frame], [flow]
        rgb_faces = []
        flow_faces = []
        for x, y, w, h in faces_rois:
            rgb_face = frame[y:y + h, x:x + w, :]
            if resize:
                rgb_face = self.resize(rgb_face)
            rgb_faces.append(rgb_face)
            if flow is not None:
                flow_face = flow[y:y + h, x:x + w, :]
                if resize:
                    flow_face = self.resize(flow_face)
                flow_faces.append(flow_face)
        return rgb_faces, flow_faces

    def getFrameSnippet(self, filepath, start_frame='random'):
        cap = cv2.VideoCapture(filepath)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if start_frame == 'random':
            start_frame = int(np.random.choice(range(int(frameCount)), size=1))
        frames = np.empty(
            (self.segment_size, frameHeight, frameWidth, 3), dtype=np.uint8)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        j = 0
        while j < self.segment_size:
            ret, frames[j] = cap.read()
            j += 1
        cap.release()
        return frames

    @staticmethod
    def getOpticalFlows(frames):
        flows = np.empty(
            (frames.shape[0] - 1, frames.shape[1], frames.shape[2], 2))
        prvs = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        for i in range(1, int(frames.shape[0])):
            frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            flows[i - 1] = cv2.calcOpticalFlowFarneback(
                prvs, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            prvs = frame
        return flows

    @staticmethod
    def resize(frame, height=128, width=128):
        # TODO: will want to test different sizes here as a hyperparameter
        return cv2.resize(frame, (height, width))

    # TODO: make this the full data extraction loop
    def generateData(self):
        raise NotImplementedError
