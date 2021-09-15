import os, pickle, argparse
from utils.proc_vid import parse_vid
from utils.face_proc import FaceProc


def main(args):
    video_dir_dict = {}
    video_dir_dict['real'] = args.real_video_dir
    video_dir_dict['fake'] = args.fake_video_dir

    face_inst = FaceProc()
    info_dict = {}

    for tag in video_dir_dict:
        vid_list = os.listdir(video_dir_dict[tag])
        for vid_name in vid_list:
            vid_path = os.path.join(video_dir_dict[tag], vid_name)
            print ('processing video: ', vid_path)
            info = {'height': [], 'width': [], 'label': [], 'fps': [], 'frame_num': []}
            frames, frame_num, fps, width, height = parse_vid(vid_path)
            info['label'] = tag
            info['height'] = height
            info['width'] = width
            info['fps'] = fps
            info['frame_num'] = frame_num
            mark_list_all = []
            for img in frames:
                landmarks = face_inst.get_landmarks(img)
                mark_list_all.append(landmarks)
            info['landmarks'] = mark_list_all
            info_dict[vid_path] = info

    with open(args.output_landmark_path, 'wb') as f:
        pickle.dump(info_dict, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="headpose forensics: train step 1")
    parser.add_argument('--real_video_dir', type=str, default='data/real')
    parser.add_argument('--fake_video_dir', type=str, default='data/fake')
    parser.add_argument('--output_landmark_path', type=str, default='cache/landmark_info.p')
    args = parser.parse_args()
    main(args)
