import numpy as np
from skimage.morphology import convex_hull_image


def new_size(org_x, org_y, large_dim=600):
    ratio = float(org_x) / float(org_y)

    if org_x > org_y:
        out_size = (int(large_dim / ratio), large_dim)
    else:
        out_size = (large_dim, int(large_dim * ratio))

    return out_size


def generate_convex_mask(shape, points_x, points_y):
    mask = np.zeros(shape, dtype=np.uint8)

    #clip to image size
    points_x = np.clip(points_x, 0, max(0, shape[1] - 1))
    points_y = np.clip(points_y, 0, max(0, shape[0] - 1))

    #set mask pixels
    mask[points_y, points_x] = 255
    mask = convex_hull_image(mask)

    return mask

