import numpy as np
import dlib

LANDMARKS = {"mouth": (48, 68),
             "mouth_inner": (60, 68),
             "right_eyebrow":(17, 22),
             "left_eyebrow": (22, 27),
             "right_eye": (36, 42),
             "left_eye": (42, 48),
             "nose": (27, 35),
             "jaw": (0, 17),
             }

MOUTH_LM = np.arange(LANDMARKS["mouth_inner"][0], LANDMARKS["mouth"][1])
LEYE_LM = np.arange(LANDMARKS["left_eye"][0], LANDMARKS["left_eye"][1])
REYE_LM = np.arange(LANDMARKS["right_eye"][0], LANDMARKS["right_eye"][1])


def shape_to_np(shape):
    number_of_points = shape.num_parts
    points = np.zeros((number_of_points, 2), dtype=np.int32)
    for i in range(0, number_of_points):
        points[i] = (shape.part(i).x, shape.part(i).y)

    return points


def classify_mouth_open(landmarks, threshold=0.15):
    left_corner = landmarks[60]
    right_corner = landmarks[64]

    upper = landmarks[62]
    lower = landmarks[66]

    h_dist = np.sqrt(np.sum(np.square(right_corner - left_corner)))
    v_dist = np.sqrt(np.sum(np.square(lower - upper)))

    if np.isclose(h_dist, 0.0):
        return False

    ratio = float(v_dist) / float(h_dist)

    return ratio > threshold


def eye_distances(landmarks, side):
    if side == 'right':
        eye_left = landmarks[36]
        eye_right = landmarks[39]
        eye_top = (landmarks[37] + landmarks[38]) / 2.0
        eye_bottom = (landmarks[40] + landmarks[41]) / 2.0
    if side == 'left':
        eye_left = landmarks[42]
        eye_right = landmarks[45]
        eye_top = (landmarks[43] + landmarks[44]) / 2.0
        eye_bottom = (landmarks[46] + landmarks[47]) / 2.0

    h_dist = np.sqrt(np.sum(np.square(eye_right - eye_left)))
    v_dist = np.sqrt(np.sum(np.square(eye_bottom - eye_top)))

    return h_dist, v_dist


def classify_eyes_open(landmarks, threshold=0.25):
    r_h_dist, r_v_dist = eye_distances(landmarks, 'right')
    if np.isclose(r_h_dist, 0.0):
        return False
    r_ratio = float(r_v_dist) / float(r_h_dist)

    l_h_dist, l_v_dist = eye_distances(landmarks, 'left')
    if np.isclose(l_h_dist, 0.0):
        return False
    l_ratio = float(l_v_dist) / float(l_h_dist)

    r_open = r_ratio > threshold
    l_open = l_ratio > threshold

    return r_open and l_open


def get_crops_landmarks(facedetector, sp68, img, roi_delta=0.0, min_score=0.0):
    """Detects faces and landmarks in image, crops image to face region."""
    face_crops = []
    final_landmarks = []

    dets, scores, idx = facedetector.run(img, 0, 0)

    num_faces = len(dets)
    if num_faces == 0:
        return face_crops, final_landmarks

    # extract especially frontal faces
    if min_score > 0.0:
        dets_new = []
        for i in range(len(dets)):
            if scores[i] > min_score:
                dets_new.append(dets[i])
        dets = dets_new

    # detect landmarks and transform to np array
    landmarks = []
    for detection in dets:
        sp_result = sp68(img, detection)
        landmarks.append(shape_to_np(sp_result))

    # crop faces
    for num in range(len(dets)):
        # copy landmarks and get crop
        face_roi = dets[num]
        face_roi = dlib.rectangle(max(0, face_roi.left()), max(0, face_roi.top()), max(0, face_roi.right()),
                                 max(0, face_roi.bottom()))

        # extend face ROI if needed
        delta_percent = roi_delta
        height = face_roi.bottom() - face_roi.top()
        delta = int(delta_percent * height)
        extended_roi_top = np.clip(face_roi.top() - delta, 0, img.shape[0])
        extended_roi_bottom = np.clip(face_roi.bottom() + delta, 0, img.shape[0])
        extended_roi_left = np.clip(face_roi.left() - delta, 0, img.shape[1])
        extended_roi_right = np.clip(face_roi.right() + delta, 0, img.shape[1])
        cropped_face = img[extended_roi_top:extended_roi_bottom, extended_roi_left:extended_roi_right, :]

        face_landmarks = landmarks[num].copy()

        face_landmarks[:, 0] = face_landmarks[:, 0] - extended_roi_left
        face_landmarks[:, 1] = face_landmarks[:, 1] - extended_roi_top

        final_landmarks.append(face_landmarks)
        face_crops.append(cropped_face)

    return face_crops, final_landmarks
