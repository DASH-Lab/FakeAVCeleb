from sklearn import svm
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import pickle
import argparse
import numpy as np

def process_training_data(data):
    videos_real = []
    videos_fake = []
    video_list = []
    label_list = []

    R_vec_feat = []
    R_mat_feat = []
    R_mat_full_feat = []
    t_vec_feat = []

    for key, value in data.items():
        label = value['label']
        if label == 'real':
            label_id = 0
            videos_real.append(key)
        else:
            label_id = 1
            videos_fake.append(key)

        print(key)
        R_c_list = value['R_c_vec']
        R_c_matrix_list = value['R_c_mat']
        t_c_list = value['t_c']

        R_a_list = value['R_a_vec']
        R_a_matrix_list = value['R_a_mat']
        t_a_list = value['t_a']

        # Compute diff
        delta_R_vec_list = [R_c_list[i][:, -1] - R_a_list[i][:, -1] for i in range(len(R_c_list)) if R_c_list[i] is not None]
        delta_t_vec_list = [t_c_list[i][:, -1] - t_a_list[i][:, -1] for i in range(len(t_c_list)) if t_c_list[i] is not None]
        delta_R_mat_list = [R_c_matrix_list[i][:, -1] - R_a_matrix_list[i][:, -1] for i in range(len(R_c_matrix_list)) if R_c_matrix_list[i] is not None]
        delta_R_full_mat_list = [(R_c_matrix_list[i] - R_a_matrix_list[i]).flatten() for i in range(len(R_c_matrix_list)) if R_c_matrix_list[i] is not None]

        R_vec_feat += delta_R_vec_list
        R_mat_feat += delta_R_mat_list
        t_vec_feat += delta_t_vec_list
        R_mat_full_feat += delta_R_full_mat_list

        label_list += [label_id] * len(delta_R_mat_list)
        video_list += [key] * len(delta_R_mat_list)

    return sorted(set(videos_real)), sorted(set(videos_fake)), video_list, label_list, R_vec_feat, R_mat_feat, R_mat_full_feat, t_vec_feat

def main(args):

    with open(args.headpose_path, 'rb') as f:
        data = pickle.load(f)
    videos_real, videos_fake, video_list, label_list, R_vec_feat, R_mat_feat, R_mat_full_feat, t_vec_feat\
        = process_training_data(data)
    features = [np.concatenate([R_mat_full_feat[i], t_vec_feat[i]]) for i in range(len(R_mat_feat))]
    classifier, scaler = train_model(features, label_list)
    model = [classifier, scaler]
    with open(args.model_save_path, 'wb') as f:
        pickle.dump(model, f)


def train_model(features, label_list, random_state=0):
    X_train, y_train = shuffle(features, label_list, random_state=random_state)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    clf = svm.SVC(kernel='rbf', probability=True)
    clf.fit(X_train, y_train)
    return clf, scaler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="headpose forensic: train step 3")
    parser.add_argument('--headpose_path', type=str, default='cache/headpose_data.p')
    parser.add_argument('--model_save_path', type=str, default='models/svm_model.p')
    args = parser.parse_args()
    main(args)

