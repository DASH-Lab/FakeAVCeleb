import cv2
import numpy as np


LAWS = {'L5': [1, 4, 6, 4, 1], 'E5': [-1, -2, 0, 2, 1], 'S5': [-1, 0, 2, 0, -1], 'R5': [1, -4, 6, -4, 1]}


def generate_law_filters():
    law_masks = {}
    for type1, vector1 in LAWS.iteritems():
        for type2, vector2 in LAWS.iteritems():
            mask_type = type1+type2
            filter_mask = np.asarray(vector1)[:, np.newaxis].T * np.asarray(vector2)[:, np.newaxis]
            law_masks[mask_type] = filter_mask

    return law_masks


def generate_mean_kernel(size):
    mean_kernel = np.ones((size, size), dtype=np.float32)
    mean_kernel = mean_kernel / mean_kernel.size

    return mean_kernel


def preprocess_image(img, size=15):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mean_kernel = generate_mean_kernel(size)
    local_means = cv2.filter2D(img, -1, mean_kernel)
    local_zero_mean_img = img - local_means

    return local_zero_mean_img


def filter_image(img, law_masks):
    law_images = {}
    for name, filter_kernel in law_masks.iteritems():
        filtered_img = cv2.filter2D(img, -1, filter_kernel)
        law_images[name] = filtered_img

    return law_images


def compute_energy(law_images, m_size):
    laws_energy = {}
    mean_kernel = generate_mean_kernel(m_size)

    for name, law_image in law_images.iteritems():
        law_image = np.abs(law_image)
        energy_image = cv2.filter2D(law_image, -1, mean_kernel)
        laws_energy[name] = energy_image

    laws_energy_final = {}
    laws_energy_final['L5E5_2'] = (laws_energy['L5E5'] + laws_energy['E5L5']) / 2.0
    laws_energy_final['L5R5_2'] = (laws_energy['L5R5'] + laws_energy['R5L5']) / 2.0
    laws_energy_final['E5S5_2'] = (laws_energy['S5E5'] + laws_energy['E5S5']) / 2.0
    laws_energy_final['L5S5_2'] = (laws_energy['S5L5'] + laws_energy['L5S5']) / 2.0
    laws_energy_final['E5R5_2'] = (laws_energy['E5R5'] + laws_energy['R5E5']) / 2.0
    laws_energy_final['S5R5_2'] = (laws_energy['S5R5'] + laws_energy['R5S5']) / 2.0
    laws_energy_final['S5S5'] = laws_energy['S5S5']
    laws_energy_final['R5R5'] = laws_energy['R5R5']
    laws_energy_final['E5E5'] = laws_energy['E5E5']

    return laws_energy_final
