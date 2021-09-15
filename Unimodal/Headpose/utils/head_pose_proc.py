'''Estimate head pose by the facial landmarks, using point 1-17 and 18-68 separately,
single Face ONLY!
modified from work by yinguobing'''
import numpy as np
import cv2
import dlib
import utils.face_utils
from utils import pose_utils as pu
import argparse
import os


class PoseEstimator(object):
    '''Estimate head pose by the facial landmarks by mapping the detected facial landmarks
    to the landmar locations of an average face.
    2 methods are available here. It could use 6 landmarks (self.model_points) or all
    68 landmarks (self.model_points_68)'''

    def __init__(self, img_size, model_path='models/model_landmark.txt'):
        # initialzize image size
        self.size = img_size
        self.model_path = model_path
        # 3D model points for average face(68points)
        self.model_points_68 = self._get_full_model_points()
        # Cameral parameters
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")

        # Assuming no lens distortion
        self.dist_coeffs = np.zeros((4, 1))


    def _get_full_model_points(self):
        ''' To obtain the coordinates of 68 landmarks for the average face.
        Input: filename --- the path to the txt file
        Output: model_points --- 68 by 3 Numpy array of the landmarks
        '''
        raw_value = []
        with open(self.model_path) as file:
            for line in file:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T
        model_points[:, -1] *= -1
        return np.array(model_points, dtype=np.float32)

    def solve_single_pose(self, marks2D, marks_id=None):
        '''get the roation and translation vectors for a single face.
        Input: face_point --- the landmarks (6 or 68) used to find the rotation and translation
                            vectors
        Output: rotation_vector --- rotation vector for mapping 3D points on 2D plane, single face
                translation_vector --- translation vector for mapping 3D points on 2D plane, single face'''
        marks_id = pu.process_input_markID(marks_id)
        marks_3D = self.model_points_68
        if marks_id is not None:
            marks3D = pu.get_pose_marks(marks_3D, marks_id)
            marks2D = pu.get_pose_marks(marks2D, marks_id)

        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            marks3D,
            marks2D,
            self.camera_matrix, self.dist_coeffs)

        return (rotation_vector, translation_vector)

    def Rodrigues_convert(self, R):
        R = cv2.Rodrigues(R)[0]
        return R
