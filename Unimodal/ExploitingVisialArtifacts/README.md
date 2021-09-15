# Exploiting Visual Artifacts to Expose Deepfakes and Face Manipulations

This repository contains an implementation of the methods described in the paper "Exploiting Visual Artifacts to Expose Deepfakes and Face Manipulations".

Please cite the paper if you use the code.

## Cite

    @inproceedings{matern2019exploiting,
      title={Exploiting Visual Artifacts to Expose Deepfakes and Face Manipulations},
      author={Matern, Falko and Riess, Christian and Stamminger, Marc},
      booktitle={2019 IEEE Winter Applications of Computer Vision Workshops (WACVW)},
      pages={83--92},
      year={2019}
    }

## Usage

### Dependencies

- Python2.7
- Packages as listed in requirements.txt
- Dlib: shape_predictor_68_face_landmarks.dat

### Installation

    Setup and activate Python2.7 virtual environment
    pip install -r requirements.txt
    download and extract http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

### Run

#### Evaluate Images

process_data.py evaluates a single image or a folder containing multiple images.

The output is saved as a .csv file containing the scores of the classifers and a flag indicating if the segmentation for the sample was valid.
The -f flag will additionally save the feature vectors as single .npy file.

Specify the pipeline with -p. Options: 'gan', 'deepfake', 'face2face'.

Example:

    python process_data.py -i img_folder -o save_folder -p deepfake -f
    
#### Fit Classifiers
A basic implementation to fit the classifiers to new data is provided in fit_classifiers.py.

The script requires the output files of process_data.py.
Additionally, the ground-truth labels have to be saved as a .csv file with columns: 'Filename', 'Label'.
To use new classifiers change the path in process_data.py accordingly.

Example:

    python fit_classifiers.py -f features.npy -s scores.csv -l labels.csv -o save_folder