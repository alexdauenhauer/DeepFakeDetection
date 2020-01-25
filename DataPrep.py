# %%
from os import path, listdir
import pickle
from random import choice
from time import time

import cv2
# import matplotlib.patches as patches
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dlib import get_frontal_face_detector
import tensorflow as tf

# %%


class DataPrepCv2():
    def __init__(self,
                 #  haar_cascades_path=None,
                 datapath=None,
                 labelpath=None,
                 segment_size=10):
        # if not haar_cascades_path:
        #     # TODO: move this folder to repo for relative path usage
        #     haar_cascades_path = '/home/alex/projects/PublicRepos/opencv/data/haarcascades'
        if not datapath:
            self.datapath = 'data/train_sample_videos'
        if not labelpath:
            self.labelpath = 'data/train_sample_videos/metadata.json'
        # self.labels = pd.read_json(self.labelpath)
        # if self.labels.shape[1] > self.labels.shape[0]:
        #     self.labels = self.labels.T
        # fcPath = haar_cascades_path
        # frontface = 'haarcascade_frontalface_default.xml'
        # profileface = 'haarcascade_profileface.xml'
        # self.ff = cv2.CascadeClassifier(path.join(fcPath, frontface))
        # self.fp = cv2.CascadeClassifier(path.join(fcPath, profileface))
        # self.ffScaleFactor = 1.5
        # self.fpScaleFactor = 1.5
        # self.ffMinNeigbors = 2
        # self.fpMinNeighbors = 2
        self.fd = get_frontal_face_detector()
        self.segment_size = segment_size
        self.frames = None
        self.flows = None

    def generateFileList(self):
        raise NotImplementedError

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
            # if resize:
            #     frame = cv2.resize(frame, (128, 128))
            #     flow = cv2.resize(flow, (128, 128))
            # return [frame], [flow]
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
                # rgb_face = cv2.resize(rgb_face, (128, 128))
            rgb_faces.append(rgb_face)
            if flow is not None:
                flow_face = flow[y:y + h, x:x + w, :]
                if resize:
                    flow_face = self.resize(flow_face)
                    # flow_face = cv2.resize(flow_face, (128, 128))
                flow_faces.append(flow_face)
        return rgb_faces, flow_faces

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
        return self.frames

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
        return self.flows

    @staticmethod
    def resize(frame, height=256, width=256):
        # TODO: will want to test different sizes here as a hyperparameter
        return cv2.resize(frame, (height, width))

    # TODO: make this the full data extraction loop
    def generateData(self):
        raise NotImplementedError

    def prepVid(
            self,
            frame_name=None,
            start_frame=None,
            face_only=False,
            rsz=(256, 256)):
        if not frame_name:
            frame_name = choice(listdir(self.datapath))
        vid = path.join(self.datapath, frame_name)
        # print(vid)
        start = time()
        frames = self.getFrameSnippet(vid, start_frame=start_frame)
        flows = self.getOpticalFlows(frames)
        rgb_rois = []
        flow_rois = []
        if face_only:
            for i in range(int(frames.shape[0])):
                frame = frames[i]
                if i > 0:
                    flow = flows[i - 1]
                    rgb_faces, flow_faces = self.getFaces(frame, flow=flow)
                else:
                    rgb_faces, flow_faces = self.getFaces(frame)
                rgb_rois.extend(rgb_faces)
                flow_rois.extend(flow_faces)
            flow_rois = [r for r in flow_rois if r is not None]
            self.rgb = np.stack(rgb_rois)
            self.flow = np.stack(flow_rois)
        else:
            rgb_rois = np.empty((frames.shape[0], rsz[0], rsz[1], 3))
            flow_rois = np.empty((flows.shape[0], rsz[0], rsz[1], 2))
            for i, frame in enumerate(frames):
                rgb_rois[i] = self.resize(frame, *rsz)
            for i, flow in enumerate(flows):
                flow_rois[i] = self.resize(flow, *rsz)
            self.rgb = rgb_rois
            self.flow = flow_rois
        return self.rgb, self.flow

    def prepFullFrames(self, filepath=None, start_frame=None, rsz=(256, 256)):
        self.filepath = filepath
        if not filepath:
            self.filepath = path.join(
                self.datapath, choice(listdir(self.datapath)))
        self.getFrameSnippet(filepath=self.filepath, start_frame=start_frame)
        frames = np.empty((self.frames.shape[0], rsz[0], rsz[1], 3))
        for i, frame in enumerate(self.frames):
            frames[i] = self.resize(frame, *rsz)
        self.frames = frames
        return self.frames

    # TODO: review this function
    def prepOpticalFlows(self,
                         filepath=None,
                         start_frame=None,
                         rsz=(256, 256)):
        if self.frames is None:
            if not filepath:
                filepath = path.join(
                    self.datapath, choice(listdir(self.datapath)))
            self.getFrameSnippet(filepath=filepath, start_frame=start_frame)
        frames = np.empty((self.frames.shape[0], rsz[0], rsz[1], 3))
        for i, frame in enumerate(self.frames):
            frames[i] = self.resize(frame, *rsz)
        self.frames = frames


class DataPrepDlib():

    def __init__(self, segment_size=5):
        self.fd = get_frontal_face_detector()
        self.segment_size = segment_size
        self.frames = None
        self.flows = None

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

    @staticmethod
    def resize(frame, height=128, width=128):
        # TODO: will want to test different sizes here as a hyperparameter
        return cv2.resize(frame, (height, width))

    def getFaces(self, frame, grayscale=True):
        orig_frame = frame
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.fd(frame, 1)
        if len(faces) < 1:
            frame = cv2.equalizeHist(frame)
            faces = self.fd(frame, 1)
        if len(faces) < 1:
#             frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
#             frame = cv2.equalizeHist(frame)
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

    def prepVid(self, filepath, start_frame=None, rsz=(128, 128)):
        self.getFrameSnippet(filepath, start_frame)
        self.getOpticalFlows()
        rgb_rois = []
        flow_rois = []
        for i, frame in enumerate(self.frames):
            faces = self.getFaces(frame)
            rois = self.getFaceRois(frame, faces)
            rgb_rois.append(rois)
            if i == 0:
                continue
            else:
                flow = self.flows[i - 1]
                rois = self.getFaceRois(flow, faces)
                flow_rois.append(rois)
        rgb_rois = np.stack(rgb_rois)
        flow_rois = np.stack(flow_rois)
        return rgb_rois, flow_rois
