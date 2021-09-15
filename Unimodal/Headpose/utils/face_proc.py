"""
Face related processing class:
1. Face alignment
2. Face landmarks
3. ...
"""

import numpy as np
import dlib
from utils.proc_vid import parse_vid
from utils.face_utils import shape_to_np
from tqdm import tqdm


class FaceProc(object):


    def __init__(self):
        # Set up dlib face detector and landmark estimator
        self.landmark_estimatior= dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        self.face_detector = dlib.get_frontal_face_detector()
    
    def get_landmarks(self, img):   
        # return 68X2 landmarks
        img_rgb = img[:, :, (2 ,1, 0)]
        rects = self.face_detector(np.uint8(img_rgb))
        if(len(rects)==0):            return None
        marks = self.landmark_estimatior(img_rgb, rects[0])
        marks = shape_to_np(marks)

        return marks

    def get_all_face_rects(self, img):
        img_rgb = img[:, :, (2 ,1, 0)]
        rects = self.face_detector(np.uint8(img_rgb))
        if(len(rects)==0):
            return None
        return rects

    def get_landmarks_all_faces(self, img, rects):
        all_landmarks = []
        for rect in rects:
            img_rgb = img[:, :, (2, 1, 0)]
            marks = self.landmark_estimatior(img_rgb, rect)
            marks = shape_to_np(marks)
            all_landmarks.append(marks)
        return all_landmarks


    def get_landmarks_vid(self, video_path):
        print('vid_path: ' + video_path)
        imgs, frame_num, fps, width, height = parse_vid(video_path)
        mark_list = []
        for i, img in enumerate(tqdm(imgs)):
            mark = self.get_landmarks(img)
            mark_list.append(mark)
        return mark_list

