import cv2
import numpy as np
from sklearn.cluster import KMeans

import pipeline_utils
import face_utils
import laws_texture


# laws texture
LAW_MASKS = laws_texture.generate_law_filters()


def extract_features_mask(img, mask):
    """Computes law texture features for masked area of image."""
    preprocessed_img = laws_texture.preprocess_image(img, size=15)
    law_images = laws_texture.filter_image(preprocessed_img, LAW_MASKS)
    law_energy = laws_texture.compute_energy(law_images, 10)

    energy_features_list = []
    for type, energy in law_energy.iteritems():
        # extract features for mask
        energy_masked = energy[np.where(mask != 0)]
        energy_feature = np.mean(energy_masked, dtype=np.float32)
        energy_features_list.append(energy_feature)

    return energy_features_list


def extract_features_eyes(landmarks, face_crop, scale=256):
    # generate mask for eyes
    l_eye_marks = landmarks[face_utils.LEYE_LM]
    r_eye_marks = landmarks[face_utils.REYE_LM]
    l_eye_mask = pipeline_utils.generate_convex_mask(face_crop[..., 0].shape, l_eye_marks[..., 0], l_eye_marks[..., 1])
    r_eye_mask = pipeline_utils.generate_convex_mask(face_crop[..., 0].shape, r_eye_marks[..., 0], r_eye_marks[..., 1])
    eye_mask = np.logical_or(l_eye_mask, r_eye_mask)
    eye_mask = eye_mask.astype(dtype=np.uint8)

    # resize input
    out_size = pipeline_utils.new_size(face_crop.shape[1], face_crop.shape[0], large_dim=scale)
    eye_mask = cv2.resize(eye_mask, (out_size[1], out_size[0]), interpolation=cv2.INTER_NEAREST)
    face_crop = cv2.resize(face_crop, (out_size[1], out_size[0]), interpolation=cv2.INTER_LINEAR)

    #segmentation fail
    if np.sum(eye_mask) < 10:
        return None

    energy_features = extract_features_mask(face_crop, eye_mask)

    return energy_features


def extract_features_mouth(landmarks, face_crop, scale=200):
    mouth_marks = landmarks[face_utils.MOUTH_LM]
    mouth_mask = pipeline_utils.generate_convex_mask(face_crop[..., 0].shape, mouth_marks[..., 0], mouth_marks[..., 1])
    mouth_mask = mouth_mask.astype(dtype=np.uint8)

    # resize input
    out_size = pipeline_utils.new_size(face_crop.shape[1], face_crop.shape[0], large_dim=scale)
    mouth_mask = cv2.resize(mouth_mask, (out_size[1], out_size[0]), interpolation=cv2.INTER_NEAREST)
    face_crop = cv2.resize(face_crop, (out_size[1], out_size[0]), interpolation=cv2.INTER_LINEAR)

    # segment mouth
    mouth_idxs = np.where(mouth_mask != 0)
    img_intensity = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
    mouth_values = img_intensity[mouth_idxs]

    # cluster teeth, mouth
    mouth_values = mouth_values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(mouth_values)
    kmeans_pred = kmeans.predict(mouth_values)

    mouth_idxs_array = np.asarray(mouth_idxs)
    bright_cluster = np.argmax(kmeans.cluster_centers_)
    mouth_idx_select = mouth_idxs_array[..., np.where(kmeans_pred == bright_cluster)[0]]

    teeth_mask = np.zeros_like(mouth_mask)
    teeth_mask[mouth_idx_select[0], mouth_idx_select[1]] = 1
    teeth_mask = np.logical_and(mouth_mask, teeth_mask)

    num_pix = np.sum(teeth_mask)
    total_pix = np.sum(mouth_mask)

    #segmentation fail, return None
    if total_pix <= 0:
        return None
    percentage = num_pix / float(total_pix)
    if percentage < 0.01:
        return None

    energy_features = extract_features_mask(face_crop, teeth_mask)

    return energy_features


def extract_features_faceborder(landmarks, face_crop, scale=256):
    face_marks = landmarks
    face_mask = pipeline_utils.generate_convex_mask(face_crop[..., 0].shape, face_marks[..., 0], face_marks[..., 1])
    face_mask = face_mask.astype(dtype=np.uint8)

    # resize
    out_size = pipeline_utils.new_size(face_crop.shape[1], face_crop.shape[0], large_dim=scale)
    face_mask = cv2.resize(face_mask, (out_size[1], out_size[0]), interpolation=cv2.INTER_NEAREST)
    face_crop = cv2.resize(face_crop, (out_size[1], out_size[0]), interpolation=cv2.INTER_LINEAR)

    # segmentation fail
    if np.sum(face_mask) < 10:
        return None

    energy_features = extract_features_mask(face_crop, face_mask)

    return energy_features


def extract_features_nose(landmarks, face_crop, scale=256):
    NOSE_LM = np.arange(30, 36)
    nose_marks = landmarks[NOSE_LM]
    nose_mask = pipeline_utils.generate_convex_mask(face_crop[..., 0].shape, nose_marks[..., 0], nose_marks[..., 1])
    nose_mask = nose_mask.astype(dtype=np.uint8)

    # resize
    out_size = pipeline_utils.new_size(face_crop.shape[1], face_crop.shape[0], large_dim=scale)
    nose_mask = cv2.resize(nose_mask, (out_size[1], out_size[0]), interpolation=cv2.INTER_NEAREST)
    face_crop = cv2.resize(face_crop, (out_size[1], out_size[0]), interpolation=cv2.INTER_LINEAR)

    #segmentation fail
    if np.sum(nose_mask) < 10:
        return None

    energy_features = extract_features_mask(face_crop, nose_mask)

    return energy_features


def process_faces(classifiers, face_crop_list, landmarks_list, pipeline, scale=256):
    """Processes a list of face crops and according landmarks with the given classifier.

    If list contains more than one image, the maximum score is returned.

    Args:
        classifier: List of two loaded scikit-learn classifers.
        face_crop_list: List of images showing crops of faces.
        landmarks_list: Landmarks consistent with face crops.
        pipeline: Selected pipeline for processing. Options: 'deepfake', 'face2face'
        scale: Size of larger dimension images are rescaled to. Default: 768.

    Returns:
        final_score_clf_0: Final score of the first trained classifier.
        final_score_clf_1: Final score of the second trained classifier.
        final_feature_vector: Six dimensional feature vector.
        valid_segmentation: True if segmentation and feature extraction was successful.
    """
    final_score_clf_0 = 0.0
    final_score_clf_1 = 0.0
    final_feature_vector_0 = None
    final_feature_vector_1 = None

    for num in range(len(face_crop_list)):

        landmarks = landmarks_list[num].copy()
        face_crop = face_crop_list[num]

        if pipeline == 'deepfake':
            # classify landmarks for mouth/eye open
            mouth_open = face_utils.classify_mouth_open(landmarks)
            eyes_open = face_utils.classify_eyes_open(landmarks)

            if not (eyes_open and mouth_open):
                continue

            features_0 = extract_features_eyes(landmarks, face_crop, scale=scale)
            features_1 = extract_features_mouth(landmarks, face_crop, scale=scale)
        elif pipeline == 'face2face':
            features_0 = extract_features_nose(landmarks, face_crop, scale=scale)
            features_1 = extract_features_faceborder(landmarks, face_crop, scale=scale)
        else:
            print 'Unknown pipeline argument.'
            return 0.0, None, False

        if features_0 is not None and features_1 is not None:
            #combine to single feature vector
            feature_vector = np.concatenate((features_0, features_1))

            score_clf_0 = classifiers[0].predict_proba(feature_vector.reshape(1, -1))[0, 1]
            score_clf_1 = classifiers[1].predict_proba(feature_vector.reshape(1, -1))[0, 1]

            if score_clf_0 > final_score_clf_0:
                final_score_clf_0 = score_clf_0
                final_feature_vector_0 = feature_vector

            if score_clf_1 > final_score_clf_1:
                final_score_clf_1 = score_clf_1
                final_feature_vector_1 = feature_vector

    valid_segmentation = final_feature_vector_0 is not None

    return (final_score_clf_0, final_score_clf_1), (final_feature_vector_0, final_feature_vector_1), valid_segmentation
