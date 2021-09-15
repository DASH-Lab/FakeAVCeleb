import argparse, os, pickle
from utils.head_pose_proc import PoseEstimator


def main(args):

    with open(args.landmark_info_path, 'rb') as f:
        vids_info = pickle.load(f)

    markID_c = args.markID_c
    markID_a = args.markID_a
    save_pose_file = args.headpose_save_path

    for key, value in vids_info.items():
        print(key)
        # Load 2d landmarks
        landmark_2d = value['landmarks']
        height = value['height']
        width = value['width']
        pose_estimate = PoseEstimator([height, width])

        R_c_list, R_a_list, t_c_list, t_a_list = [], [], [], []
        R_c_matrix_list, R_a_matrix_list = [], []
        for landmark_2d_cur in landmark_2d:
            R_c, t_c = None, None
            R_a, t_a = None, None
            R_c_matrix, R_a_matrix = None, None

            # landmark_2d_cur = landmark_2d[i]
            if landmark_2d_cur is not None:
                R_c, t_c = pose_estimate.solve_single_pose(landmark_2d_cur, markID_c)
                R_a, t_a = pose_estimate.solve_single_pose(landmark_2d_cur, markID_a)

                R_c_matrix = pose_estimate.Rodrigues_convert(R_c)
                R_a_matrix = pose_estimate.Rodrigues_convert(R_a)


            R_c_list.append(R_c)
            R_a_list.append(R_a)

            t_c_list.append(t_c)
            t_a_list.append(t_a)

            R_c_matrix_list.append(R_c_matrix)
            R_a_matrix_list.append(R_a_matrix)

        value['R_c_vec'] = R_c_list
        value['R_c_mat'] = R_c_matrix_list
        value['t_c'] = t_c_list

        value['R_a_vec'] = R_a_list
        value['R_a_mat'] = R_a_matrix_list
        value['t_a'] = t_a_list

    # Save to pkl file
    with open(save_pose_file, 'wb') as f:
        pickle.dump(vids_info, f)

    print('Done!')


if __name__ == '__main__':
   parser = argparse.ArgumentParser(description="headpose forensic: train step 2")
   parser.add_argument('--landmark_info_path', type=str, default='cache/landmark_info.p', help='landmarks file from step 1')
   parser.add_argument('--markID_a', type=str, default='1-36,49,55', help='landmarks for whole face')
   parser.add_argument('--markID_c', type=str, default='18-36,49,55', help='landmarks for center face')
   parser.add_argument('--headpose_save_path', type=str, default='cache/headpose_data.p', help='file to save headpose data')
   args = parser.parse_args()
   main(args)
