from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os 


frames_total = 8    # each video 8 uniform samples


class Normaliztion_valtest(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x, image_ir, binary_mask, string_name = sample['image_x'], sample['image_ir'],sample['binary_mask'],sample['string_name']
        
        new_image_x = (image_x - 127.5)/128     # [-1,1]
        new_image_ir = (image_ir - 127.5)/128     # [-1,1]
        
        return {'image_x': new_image_x,'image_ir': new_image_ir, 'binary_mask': binary_mask, 'string_name': string_name}


class ToTensor_valtest(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """
    def __call__(self, sample):
        image_x, image_ir, binary_mask, spoofing_label = sample['image_x'], sample['image_ir'], sample['binary_mask'],sample['string_name']
        
        # swap color axis because
        # numpy image: (batch_size) x H x W x C
        # torch image: (batch_size) x C X H X W
        image_x = image_x[:,:,::-1].transpose((2, 0, 1))
        image_x = np.array(image_x)
        
        image_ir = image_ir[:,:,::-1].transpose((2, 0, 1))
        image_ir = np.array(image_ir)        
        
        
        binary_mask = np.array(binary_mask)

                        
        spoofing_label_np = np.array([0],dtype=np.long)
        spoofing_label_np[0] = spoofing_label
        
        
        return {'image_x': torch.from_numpy(image_x.astype(np.float)).float(), 'image_ir': torch.from_numpy(image_ir.astype(np.float)).float(), 'binary_mask': torch.from_numpy(binary_mask.astype(np.float)).float(), 'string_name': torch.from_numpy(spoofing_label_np.astype(np.float)).float()}


class Spoofing_valtest(Dataset):

    def __init__(self, root_dir,  transform=None):

        self.landmarks_frame_real = os.listdir(root_dir+"/FRAMS/REAL0") #pd.read_csv(info_list, delimiter=' ', header=None)
        self.landmarks_frame_fake = os.listdir(root_dir+"/FRAMS/FAKE0") #pd.read_csv(info_list, delimiter=' ', header=None)
        self.root_dir = root_dir
        self.transform = transform
        for xx in range( len(self.landmarks_frame_real)):
            self.landmarks_frame_real[xx] = root_dir+"/FRAMS/REAL0/" + self.landmarks_frame_real[xx]
        for xx in range( len(self.landmarks_frame_fake)):
            self.landmarks_frame_fake[xx] = root_dir+"/FRAMS/FAKE0/" + self.landmarks_frame_fake[xx]
        #self.landmarks_frame = self.landmarks_frame_fake + self.landmarks_frame_real



        self.landmarks_frame_real_s = os.listdir(root_dir+"/SPEC/REAL0") #pd.read_csv(info_list, delimiter=' ', header=None)
        self.landmarks_frame_fake_s = os.listdir(root_dir+"/SPEC/FAKE0") #pd.read_csv(info_list, delimiter=' ', header=None)
        for xx in range( len(self.landmarks_frame_real_s)):
            self.landmarks_frame_real_s[xx] = [self.landmarks_frame_real[xx], root_dir+"/SPEC/REAL0/" + self.landmarks_frame_real_s[xx]]
        for xx in range( len(self.landmarks_frame_fake_s)):
            self.landmarks_frame_fake_s[xx] = [self.landmarks_frame_fake[xx],root_dir+"/SPEC/FAKE0/" + self.landmarks_frame_fake_s[xx]]
        self.landmarks_frame = self.landmarks_frame_fake_s + self.landmarks_frame_real_s
        random.shuffle(self.landmarks_frame)



    def __len__(self):
        return len(self.landmarks_frame)

    
    def __getitem__(self, idx):
        #print(self.landmarks_frame.iloc[idx, 0])
        videoname = self.landmarks_frame[idx]
        image_path = videoname[0]
        
        ir_path = videoname[1]
        image_x, image_ir, binary_mask = self.get_single_image_x(image_path, ir_path)
        
		    
        if "REAL0" in image_path:
            spoofing_label = 1            # real
        else:
            spoofing_label = 0            # fake
            binary_mask = np.zeros((32, 32))    
        
        
        #frequency_label = self.landmarks_frame.iloc[idx, 2:2+50].values  

        sample = {'image_x': image_x, 'image_ir': image_ir, 'binary_mask': binary_mask, 'string_name': spoofing_label}

        # if self.transform:
        sample = self.transform(sample)
        return sample

    def get_single_image_x(self, image_path, ir_path):
        

        image_x = np.zeros((256, 256, 3))
        binary_mask = np.zeros((32, 32))
 
 
        image_x_temp = cv2.imread(image_path)
        image_x_temp_ir = cv2.imread(ir_path)
        # image_x_temp_depth = cv2.imread(depth_path)
        image_x_temp_gray = cv2.imread(image_path, 0)


        image_x = cv2.resize(image_x_temp, (256, 256))
        image_x_ir = cv2.resize(image_x_temp_ir, (256, 256))
        # image_x_depth = cv2.resize(image_x_temp_depth, (256, 256))
        image_x_temp_gray = cv2.resize(image_x_temp_gray, (32, 32))
        # image_x_aug_depth = seq.augment_image(image_x_depth) 
        
             
        
        for i in range(32):
            for j in range(32):
                if image_x_temp_gray[i,j]>0:
                    binary_mask[i,j]=1
                else:
                    binary_mask[i,j]=0
        
        return image_x, image_x_ir, binary_mask







if __name__ == '__main__':
    # usage
    # MAHNOB
    root_list = '/wrk/yuzitong/DONOTREMOVE/BioVid_Pain/data/cropped_frm/'
    trainval_list = '/wrk/yuzitong/DONOTREMOVE/BioVid_Pain/data/ImageSet_5fold/trainval_zitong_fold1.txt'
    

    BioVid_train = BioVid(trainval_list, root_list, transform=transforms.Compose([Normaliztion(), Rescale((133,108)),RandomCrop((125,100)),RandomHorizontalFlip(),  ToTensor()]))
    
    dataloader = DataLoader(BioVid_train, batch_size=1, shuffle=True, num_workers=8)
    
    # print first batch for evaluation
    for i_batch, sample_batched in enumerate(dataloader):
        #print(i_batch, sample_batched['image_x'].size(), sample_batched['video_label'].size())
        print(i_batch, sample_batched['image_x'], sample_batched['pain_label'], sample_batched['ecg'])
        pdb.set_trace()
        break

            
 


