import cv2
import numpy as np


def process_input_markID(IDs):
    final_id = []
    dashes_include = IDs.split(',')
    for item in dashes_include:
        item = item.strip().split('-')
        if (len(item) == 1):
            final_id.append(int(str(item[0])))
        else:
            for i in range(int(str(item[0])), int(str(item[1])) + 1):
                final_id.append(i)
    return final_id


def draw_boxes(image, rotation_vctors, translation_vectors, camera_matrix, dist_coeffs, color=(0, 255, 0), line_width=1):
    '''draw annotation boxes for multiple faces on an image. The each annotation box indicate pose of
    each head. Simply by calling the function to drawn each face with its mapping (transaltion and rotation vector).
    Input: image --- the image to be drawn on
        rotation_vectors --- list of mapping from 3D to 2D plane containing n vectors for n faces
        translation_vectors --- list of mapping from 3D to 2D plane containing n vectors for n faces
        '''
    for r_vec, t_vec in zip(rotation_vctors, translation_vectors):
        # get translation and rotation vector to drawn head pose for individual faces
        draw_annotation_box(image, r_vec, t_vec, camera_matrix, dist_coeffs, color=color, line_width=line_width)


def draw_annotation_box(image, rotation_vector, translation_vector, camera_matrix, dist_coeffs, color=(0, 255, 0), line_width=1):
    """Draw a 3D box as annotation of head pose for a single face. Each head pose annotation contains
    a smaller square on face surface, and a larger sqaure in front of the smaller one. The openning
    of the 2 squares indicate the orientation of head pose.
    Input:
        image --- image to be drawn on.
        rotation_vector --- a mapping vector for a single face from 3D to 2D plane (rotation)
        translation_vector --- a mapping vector for single face from 3D to 2D plane (translation)
    Output: image --- the annotated image
        """
    point_3d = []  # a list to save the 3D points to show head pose
    rear_size = 50  # smaller square edge length
    rear_depth = 0  # distance between small squares to nose tip
    point_3d.append((-rear_size, -rear_size, rear_depth))  # get all 4 points for small squared
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = 100  # larger square edge length
    front_depth = 50  # distance between large squares to nose tip
    point_3d.append((-front_size, -front_size, front_depth))  # all 4 points for larger square
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d image points
    # print(type(self.dist_coeffs))
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))  # convert to integer for pixels
    # print('2D points', point_2d)
    # Draw all the lines
    cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)
    return image


def draw_annotation_arrow(image, rotation_vector, translation_vector, camera_matrix, dist_coeffs, color=(0, 255, 0), line_width=1):
    '''draw arrow as the head pose direction (direction facing to), single face only.
    Input:
        image --- image to be drawn on.
        rotation_vector --- a mapping vector for a single face from 3D to 2D plane (rotation)
        translation_vector --- a mapping vector for single face from 3D to 2D plane (translation)
    Output: image --- the annotated image
    '''
    points_3D = []  # a list to store the 3D points to draw
    rear_point_3D = [0, 0, 0]  # the rear point for the arrow
    front_point_3D = [0, 0, 100]  # the point for the tip of the array
    points_3D.append(rear_point_3D)
    points_3D.append(front_point_3D)
    points_3D = np.array(points_3D, dtype=np.float).reshape(-1, 3)
    # map the 3D points onto 2D image plane
    (points_2d, _) = cv2.projectPoints(points_3D,
                                       rotation_vector,
                                       translation_vector,
                                       camera_matrix,
                                       dist_coeffs)
    points_2d = np.int32(points_2d.reshape(-1, 2))  # convert to integer
    # draw on image plane
    cv2.arrowedLine(image, tuple(points_2d[0]), tuple(points_2d[1]), color, 2, tipLength=0.5)
    return image


def get_pose_marks(marks, markID):
    chosen_marks = []
    for ID in markID:
        chosen_marks.append(marks[ID - 1])
    return np.array(chosen_marks, dtype=np.float32)
