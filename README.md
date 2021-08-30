# FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset

![Header](images/teaser.png)

## Overview
FakeAVCeleb is a novel Audio-Video Multimodal Deepfake Detection dataset (FakeAVCeleb), which contains not only deepfake videos but also respective synthesized cloned audios. 


## Access
If you would like to download the FakeAVCeleb dataset, please fill out the [google request form](https://docs.google.com/forms/u/1/d/e/1FAIpQLSfPDd3oV0auqmmWEgCSaTEQ6CGpFeB-ozQJ35x-B_0Xjd93bw/viewform) and, once accepted, we will send you the link to our download script.

Once, you obtain the download link, please see the [download section](dataset/README.md). You can also find details about our FakeAVCeleb dataset.

## Requirements and Installation


## [Benchmark](TBD)




## Deepfake Dataset for Quantitative Comparison
- Quantitative comparison of FakeAVCeleb to existing publicly available **Deepfake dataset**.

| Dataset | Real Videos | Fake Videos | Total Videos | Rights Cleared | Agreeing subjects | Total subjects | Methods | Real Audio | Deepfake Audio |
|------------------|-------------------------------------------------------------|---------------------|---------------------|-----|-----|------|---|-----|-----|
| UADFV            | 49                                                          | 49                  | 98                  | No  | 0   | 49   | 1 | No  | No  |
| DeepfakeTIMIT    | 640                                                         | 320                 | 960                 | No  | 0   | 32   | 2 | No  | Yes |
| FF++             | 1000                                                        | 4,000               | 5,000               | No  | 0   | N/A  | 4 | No  | No  |
| Celeb-DF         | 590                                                         | 5,639               | 6,229               | No  | 0   | 59   | 1 | No  | No  |
| Google DFD       | 0                                                           | 3,000               | 3,000               | Yes | 28  | 28   | 5 | No  | No  |
| DeeperForensics  | 50,000                                                      | 10,000              | 60,000              | No  | 100 | 100  | 1 | No  | No  |
| DFDC             | 23,654                                                      | 104,500             | 128,154             | Yes | 960 | 960  | 8 | Yes | Yes |
| KoDF             | 62,166                                                      | 175,776             | 237,942             | Yes | 403 | 403  | 6 | No  | Yes |
| SystemName      | 490 | 25,000 | 25,500 | Yes | 0 | 600 | 5 | Yes | Yes |



## Training & Evaluation
### 1. Benchmark
To train and evaluate the model(s) in the paper, run this command:
- Unimodal
    ```train
    cd ./Unimodal/train
    python ~~
    ```
   After train the model, you can evaluate the result. 
    ```soely eval (audio and video, respectively.)
    cd ./Unimodal/eval
    python Eval_Headpose.py
    ```
    
    ```ensemble eval (paired video with audio.)
    cd ./ENSEMBLE/~~
    python ~~
    ```
  
- Multimodal
  ```train
    cd ./Multimodal/train
    python Train_~~.py
  ```
  ```eval
    cd ./Multimodal/eval
    python Eval_~~.py
  ```

## Result
- Frame-level AUC scores (%)** of various methods on compared datasets.

| Dataset | UADFV | DF-TIMIT (LQ) | DF-TIMIT (HQ) | FF-DF | DFD | DFDC | Celeb-DF | FakeAVCeleb |
|---|---|---|---|---|---|---|---|---|
| Capsule | 61.3 | 78.4 | 74.4 | 96.6 | 64.0 | 53.3 | 57.5 | 73.1 |
| HeadPose | 89.0 | 55.1 | 53.2 | 47.3 | 56.1 | 55.9 | 54.6 | 49.2 |
| VA-MLP | 70.2 | 61.4 | 62.1 | 66.4 | 69.1 | 61.9 | 55.0 | 55.8 |
| VA-LogReg | 54.0 | 77.0 | 77.3 | 78.0 | 77.2 | 66.2 | 55.1 | 65.4 |
| Xception-raw | 80.4 | 56.7 | 54.0 | 99.7 | 53.9 | 49.9 | 48.2 | 73.1 |
| Xception-comp | 91.2 | 95.9 | 94.4 | 99.7 | 85.9 | 72.2 | 65.3 | 73.4 |
| Meso4 | 84.3 | 87.8 | 68.4 | 84.7 | 76.0 | 75.3 | 54.8 | 43.1 |
| MesoInception4 | 82.1 | 80.4 | 62.7 | 83.0 | 75.9 | 73.2 | 53.6 | 77.8 |

## Citation
If you use the FakeAVCeleb data or code please cite:
```
TBD

```
  
## References


## Contect
If you have any questions, please contact us at hasam.khalid/shahroz/kimminha@g.skku.edu.

## License
The data can be released under the [FakeAVCeleb Request Forms](https://docs.google.com/forms/u/1/d/e/1FAIpQLSfPDd3oV0auqmmWEgCSaTEQ6CGpFeB-ozQJ35x-B_0Xjd93bw/viewform), and the code is released under the MIT license.

Copyright (c) 2021
