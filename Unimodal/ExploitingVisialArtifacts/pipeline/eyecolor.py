import cv2
import numpy as np

from scipy.spatial.distance import euclidean
from scipy.ndimage.morphology import binary_dilation

from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage import exposure

import pipeline_utils
import face_utils


def extract_reflection(img, mask):
    negative_mask = np.logical_not(mask)
    roi_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    roi_V = roi_HSV[..., 2]
    roi_V = exposure.rescale_intensity(roi_V, in_range=(0, 255))
    roi_V[negative_mask] = 0
    highlights = roi_V >= 150
    num_refl = np.sum(highlights)

    pupil = roi_V <= 50
    pupil[negative_mask] = 0
    highlights = np.logical_or(highlights, pupil)

    return highlights, num_refl


def segment_iris(face_crop, eye_mask):
    img_copy = face_crop.copy()

    mask_coords = np.where(eye_mask != 0)
    mask_min_y = np.min(mask_coords[0])
    mask_max_y = np.max(mask_coords[0])
    mask_min_x = np.min(mask_coords[1])
    mask_max_x = np.max(mask_coords[1])

    roi_top = np.clip(mask_min_y, 0, face_crop.shape[0])
    roi_bottom = np.clip(mask_max_y, 0, face_crop.shape[0])
    roit_left = np.clip(mask_min_x, 0, face_crop.shape[1])
    roi_right = np.clip(mask_max_x, 0, face_crop.shape[1])

    roi_image = img_copy[roi_top:roi_bottom, roit_left:roi_right, :]
    roi_mask = eye_mask[roi_top:roi_bottom, roit_left:roi_right]

    roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_RGB2LAB)
    roi_gray = roi_gray[..., 0]
    roi_gray = exposure.rescale_intensity(roi_gray, in_range=(0, 255))

    negative_mask = np.logical_not(roi_mask)
    roi_gray[negative_mask] = 0
    edges = canny(roi_gray, sigma=2.0, low_threshold=40, high_threshold=70)

    edges_mask = canny(roi_mask * 255, sigma=1.5, low_threshold=1, high_threshold=240)
    edges_mask = binary_dilation(edges_mask)
    edges_mask = np.logical_not(edges_mask)

    # detect circles within radius range based on landmarks
    edges = np.logical_and(edges, edges_mask)
    diam = mask_max_x - mask_min_x
    radius_min = int(diam / 4.0)
    radius_max = int(diam / 2.0)
    hough_radii = np.arange(radius_min, radius_max, 1)
    hough_res = hough_circle(edges, hough_radii)
    # select best detection
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=1, normalize=True)

    # select central point and diam/4 as fallback
    if radii is None or radii.size == 0:
        cx_glob = int(np.mean(mask_coords[1]))
        cy_glob = int(np.mean(mask_coords[0]))
        radius_glob = int(diam / 4.0)
        valid = False
    else:
        cx_glob = cx[0] + mask_min_x
        cy_glob = cy[0] + mask_min_y
        radius_glob = radii[0]
        valid = True

    # generate mask for iris
    iris_mask = np.zeros_like(eye_mask, dtype=np.uint8)
    cv2.circle(iris_mask, (cx_glob, cy_glob), radius_glob, 255, -1)
    iris_mask = np.logical_and(iris_mask, eye_mask)

    roi_iris = iris_mask[roi_top:roi_bottom, roit_left:roi_right]

    highlights, num_refl = extract_reflection(roi_image, roi_iris)
    highlights_global = np.zeros_like(eye_mask)
    highlights_coord = np.where(highlights != 0)
    highlights_coord[0][:] += mask_min_y
    highlights_coord[1][:] += mask_min_x
    highlights_global[highlights_coord] = 1

    eye_cx = int(np.mean(mask_coords[1]))
    eye_cy = int(np.mean(mask_coords[0]))

    return iris_mask, (cx_glob, cy_glob), radius_glob, (eye_cx, eye_cy), highlights_global, valid


def iris_distances(eye_marks, eye_center, iris_center):
    l_point = eye_marks[0]
    r_point = eye_marks[3]
    diam_lm = euclidean(l_point, r_point)

    diff_pos_x = (eye_center[0] - iris_center[0]) / (diam_lm / 2.0)
    diff_pos_y = (eye_center[1] - iris_center[1]) / (diam_lm / 2.0)

    return diam_lm, diff_pos_x, diff_pos_y


def compute_histograms(iris_vals):
    R_hist, _ = np.histogram(iris_vals[..., 0], bins=64, range=[0, 255], density=True)
    G_hist, _ = np.histogram(iris_vals[..., 1], bins=64, range=[0, 255], density=True)
    B_hist, _ = np.histogram(iris_vals[..., 2], bins=64, range=[0, 255], density=True)

    return R_hist, G_hist, B_hist


def extract_eyecolor_features(landmarks, face_crop):
    l_eye_marks = landmarks[face_utils.LEYE_LM]
    r_eye_marks = landmarks[face_utils.REYE_LM]

    l_eye_mask = pipeline_utils.generate_convex_mask(face_crop[..., 0].shape, l_eye_marks[..., 0], l_eye_marks[..., 1])
    r_eye_mask = pipeline_utils.generate_convex_mask(face_crop[..., 0].shape, r_eye_marks[..., 0], r_eye_marks[..., 1])

    l_iris, l_iris_center, l_radius, l_eye_center, l_highlights, l_valid = segment_iris(face_crop, l_eye_mask)
    r_iris, r_iris_center, r_radius, r_eye_center, r_highlights, r_valid = segment_iris(face_crop, r_eye_mask)



    # heuristic check of iris segmentation
    l_diam_lm, l_diff_pos_x, l_diff_pos_y = iris_distances(l_eye_marks, l_eye_center, l_iris_center)
    r_diam_lm, r_diff_pos_x, r_diff_pos_y = iris_distances(r_eye_marks, r_eye_center, r_iris_center)

    diam_diff = abs(l_diam_lm - r_diam_lm) / np.mean((l_diam_lm, r_diam_lm))
    pos_diff = (abs(l_diff_pos_x - r_diff_pos_x) + abs(l_diff_pos_y - r_diff_pos_y)) / 2
    radius_diff = abs(l_radius - r_radius) / np.mean((l_radius, r_radius))

    # if iris detection/segmentation fails return None
    valid_seg = True
    if not l_valid or not r_valid:
        valid_seg = False
        # return None, None
    # if iris check fails return None
    if pos_diff > 0.1 or diam_diff > 0.4 or radius_diff > 0.4:
        valid_seg = False
        # return None, None

    l_iris = np.logical_xor(l_iris, l_highlights)
    r_iris = np.logical_xor(r_iris, r_highlights)

    img_dis_hsv = cv2.cvtColor(face_crop, cv2.COLOR_RGB2HSV)

    l_iris_vals = face_crop[l_iris, :]
    r_iris_vals = face_crop[r_iris, :]

    l_iris_vals_hsv = img_dis_hsv[l_iris, :]
    r_iris_vals_hsv = img_dis_hsv[r_iris, :]

    # if segmentation fails fallback to zero
    if l_iris_vals.size < 1 or r_iris_vals.size < 1 or l_iris_vals_hsv.size < 1 or r_iris_vals_hsv.size < 1:
        distance_HSV = 0.0
        dist_H = 0.0
        dist_S = 0.0
        dist_V = 0.0
        correl_R = 0.0
        correl_G = 0.0
        correl_B = 0.0
    else:
        lIrisMean = np.mean(l_iris_vals_hsv, axis=0)
        rIrisMean = np.mean(r_iris_vals_hsv, axis=0)

        # distance for HSV, taking into account opencv scaling
        dist_H = min(abs(lIrisMean[0] - rIrisMean[0]), 180 - abs(lIrisMean[0] - rIrisMean[0])) / 90.0
        dist_S = abs(lIrisMean[1] - rIrisMean[1]) / 255.0
        dist_V = abs(lIrisMean[2] - rIrisMean[2]) / 255.0
        distance_HSV = dist_H * 90.0 + dist_S * 255.0 + dist_V * 255.0

        l_R_hist, l_G_hist, l_B_hist = compute_histograms(l_iris_vals)
        r_R_hist, r_G_hist, r_B_hist = compute_histograms(r_iris_vals)

        correl_R = cv2.compareHist(l_R_hist.astype(dtype=np.float32), r_R_hist.astype(dtype=np.float32),
                                   cv2.HISTCMP_CORREL)
        correl_G = cv2.compareHist(l_G_hist.astype(dtype=np.float32), r_G_hist.astype(dtype=np.float32),
                                   cv2.HISTCMP_CORREL)
        correl_B = cv2.compareHist(l_B_hist.astype(dtype=np.float32), r_B_hist.astype(dtype=np.float32),
                                   cv2.HISTCMP_CORREL)

        correl_R = ((correl_R * -1.0) + 1.0) / 2.0
        correl_G = ((correl_G * -1.0) + 1.0) / 2.0
        correl_B = ((correl_B * -1.0) + 1.0) / 2.0

    feature_vector = np.array((dist_H, dist_S, dist_V, correl_R, correl_G, correl_B))

    return feature_vector, distance_HSV, valid_seg


def process_faces(classifier, face_crop_list, landmarks_list, scale=768):
    """Processes a list of face crops and according landmarks with the given classifier.

    If list contains more than one image, the maximum score is returned.

    Args:
        classifier: Loaded scikit-learn classifer.
        face_crop_list: List of images showing crops of faces.
        landmarks_list: Landmarks consistent with face crops.
        scale: Size of larger dimension images are rescaled to. Default: 768.

    Returns:
        final_score_clf: Final score of the trained classifier.
        final_score_HSV: Final score comparing HSV directly.
        final_feature_vector: Six dimensional feature vector.
        valid_segmentation: True if segmentation and feature extraction was successful.
    """
    final_score_clf = 0.0
    final_score_HSV = 0.0
    final_feature_vector = None
    final_valid_seg = False

    for num in range(len(face_crop_list)):

        face_crop = face_crop_list[num]
        landmarks = landmarks_list[num].copy()

        out_size = pipeline_utils.new_size(face_crop.shape[1], face_crop.shape[0], large_dim=scale)
        scale_x = float(out_size[1]) / face_crop.shape[1]
        scale_y = float(out_size[0]) / face_crop.shape[0]

        landmarks_resize = landmarks.copy()
        landmarks_resize[:, 0] = landmarks_resize[:, 0] * scale_x
        landmarks_resize[:, 1] = landmarks_resize[:, 1] * scale_y

        face_crop_resize = cv2.resize(face_crop, (int(out_size[1]), int(out_size[0])), interpolation=cv2.INTER_LINEAR)

        feature_vector, distance_HSV, valid_seg = extract_eyecolor_features(landmarks_resize, face_crop_resize)
        final_valid_seg = final_valid_seg or valid_seg

        if feature_vector is not None:
            score_clf = classifier.predict_proba(feature_vector.reshape(1, -1))[0, 1]

            final_score_HSV = max(final_score_HSV, distance_HSV)
            if score_clf > final_score_clf:
                final_score_clf = score_clf
                final_feature_vector = feature_vector

    return final_score_clf, final_score_HSV, final_feature_vector, final_valid_seg
