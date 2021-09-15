import cv2
import numpy as np
import os


def parse_vid(video_path):
    vidcap = cv2.VideoCapture(video_path)
    frame_num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)) # float
    height = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    imgs = []
    while True:
        success, image = vidcap.read()
        if success:
            imgs.append(image)
        else:
            break

    vidcap.release()
    if len(imgs) != frame_num:
        frame_num = len(imgs)
    return imgs, frame_num, fps, width, height


def extract_video_id(video_folder):
    all_files = sorted(os.listdir(video_folder))
    IDs = []
    for file in all_files:
        IDs.append(file.split('.')[0])
    return IDs
