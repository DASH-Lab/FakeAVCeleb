# FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset

![Header](images/teaser.png)

## Overview
FakeAVCeleb is a novel Audio-Video Multimodal Deepfake Detection dataset (FakeAVCeleb), which contains not only deepfake videos but also respective synthesized cloned audios. 


## Access
If you would like to download the FakeAVCeleb dataset, please fill out the [google request form](https://docs.google.com/forms/u/1/d/e/1FAIpQLSfPDd3oV0auqmmWEgCSaTEQ6CGpFeB-ozQJ35x-B_0Xjd93bw/viewform) and, once accepted, we will send you the link to our download script.

Once, you obtain the download link, please see the [download section](dataset/README.md). You can also find details about our FakeAVCeleb dataset.

## Requirements and Installation


## [Benchmark](TBD)



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
  
## Results
Our model achieves the following performance on benchmark:
```
| Dataset | Real Videos | Fake Videos | | | | | | | |
|------------------|-------------------------------------------------------------|
| UADFV            | 49                                                          | 49                  | 98                  | No  | 0   | 49   | 1 | No  | No  |
| DeepfakeTIMIT    | 640                                                         | 320                 | 960                 | No  | 0   | 32   | 2 | No  | Yes |
| FF++             | 1000                                                        | 4,000               | 5,000               | No  | 0   | N/A  | 4 | No  | No  |
| Celeb-DF         | 590                                                         | 5,639               | 6,229               | No  | 0   | 59   | 1 | No  | No  |
| Google DFD       | 0                                                           | 3,000               | 3,000               | Yes | 28  | 28   | 5 | No  | No  |
| DeeperForensics  | 50,000                                                      | 10,000              | 60,000              | No  | 100 | 100  | 1 | No  | No  |
| DFDC             | 23,654                                                      | 104,500             | 128,154             | Yes | 960 | 960  | 8 | Yes | Yes |
| KoDF             | 62,166                                                      | 175,776             | 237,942             | Yes | 403 | 403  | 6 | No  | Yes |
| \SystemName      | 490+\hl{RECHECK}                                            | 25,000+\hl{RECHECK} | 25,500+\hl{RECHECK} | Yes | 0   | 600+ | 5 | Yes | Yes |
```

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
