import pandas as pd
import numpy as np


def save2csv(dest_path, all_landmarks, vids_id, label_list):
    col_names = []
    col_names.append('img_name')
    for idx in range(1, 69):
        col_names.append('X' + str(idx))
        col_names.append('Y' + str(idx))
    data = pd.DataFrame(columns=col_names)
    col_names.append('label')
    n = 0
    for i, mark in enumerate(all_landmarks):
        for j, pts in enumerate(mark):
            if pts is None:
                temp_data = np.ones([1, 68 * 2]) * -1
            else:
                temp_data = np.reshape(pts, (1, -1))
            # print(temp_data)
            data.loc[n] = [vids_id[i] + '/' + str(j)] + list(temp_data[0]) + [label_list[i]]
            n += 1
    data.to_csv(dest_path)



