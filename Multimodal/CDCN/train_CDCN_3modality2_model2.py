from __future__ import print_function, division
import torch
import matplotlib as mpl
mpl.use('TkAgg')
import argparse,os
import pandas as pd
import cv2
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support as score
from torchvision import transforms


from models.CDCNs import Conv2d_cd, CDCN_3modality2
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


from Loadtemporal_BinaryMask_train_3modality import Spoofing_train, Normaliztion, ToTensor, RandomHorizontalFlip, Cutout, RandomErasing
from Loadtemporal_valtest_3modality import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest


import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import copy
import pdb

from utils import AvgrageMeter, accuracy, performances



# Dataset root
   
train_list = '/media/data1/hsm/acm_workshop/data/'
val_list = '/media/data1/hsm/acm_workshop/testdata/'




 





def contrast_depth_conv(input):
    ''' compute contrast depth in both of (out, label) '''
    '''
        input  32x32
        output 8x32x32
    '''
    

    kernel_filter_list =[
                        [[1,0,0],[0,-1,0],[0,0,0]], [[0,1,0],[0,-1,0],[0,0,0]], [[0,0,1],[0,-1,0],[0,0,0]],
                        [[0,0,0],[1,-1,0],[0,0,0]], [[0,0,0],[0,-1,1],[0,0,0]],
                        [[0,0,0],[0,-1,0],[1,0,0]], [[0,0,0],[0,-1,0],[0,1,0]], [[0,0,0],[0,-1,0],[0,0,1]]
                        ]
    
    kernel_filter = np.array(kernel_filter_list, np.float32)
    
    kernel_filter = torch.from_numpy(kernel_filter.astype(np.float)).float().cuda()
    # weights (in_channel, out_channel, kernel, kernel)
    kernel_filter = kernel_filter.unsqueeze(dim=1)
    
    input = input.unsqueeze(dim=1).expand(input.shape[0], 8, input.shape[1],input.shape[2])
    
    contrast_depth = F.conv2d(input, weight=kernel_filter, groups=8)  # depthwise conv
    
    return contrast_depth


class Contrast_depth_loss(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Contrast_depth_loss,self).__init__()
        return
    def forward(self, out, label): 
        '''
        compute contrast depth in both of (out, label),
        then get the loss of them
        tf.atrous_convd match tf-versions: 1.4
        '''
        contrast_out = contrast_depth_conv(out)
        contrast_label = contrast_depth_conv(label)
        
        
        criterion_MSE = nn.MSELoss().cuda()
    
        loss = criterion_MSE(contrast_out, contrast_label)
        #loss = torch.pow(contrast_out - contrast_label, 2)
        #loss = torch.mean(loss)
    
        return loss




# main function
def train_test():
    # GPU  & log file  -->   if use DataParallel, please comment this command
    #os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.gpu)

    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
    log_file = open(args.log+'/'+ args.log+'_log.txt', 'w')
    
    echo_batches = args.echo_batches

    print("Oulu-NPU, P1:\n ")

    log_file.write('Oulu-NPU, P1:\n ')
    log_file.flush()

    # load the network, load the pre-trained model in UCF101?
    finetune = args.finetune
    if finetune==True:
        print('finetune!\n')

    else:
        print('train from scratch!\n')
        log_file.write('train from scratch!\n')
        log_file.flush()
         
		 
        #model = CDCN_3modality2( basic_conv=Conv2d_cd, theta=0.7)
        model = CDCN_3modality2( basic_conv=Conv2d_cd, theta=args.theta)
        

        model = model.cuda()


        lr = args.lr
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    print(model) 
    
    
    criterion_absolute_loss = nn.MSELoss().cuda()
    criterion_contrastive_loss = Contrast_depth_loss().cuda() 
    


    ACER_save = 1.0
    
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        scheduler.step()
        if (epoch + 1) % args.step_size == 0:
            lr *= args.gamma

        
        loss_absolute = AvgrageMeter()
        loss_contra =  AvgrageMeter()
        #top5 = utils.AvgrageMeter()
        
        
        ###########################################
        '''                train             '''
        ###########################################
        model.train()
        
        # load random 16-frame clip data every epoch
        train_data = Spoofing_train(train_list, transform=transforms.Compose([RandomErasing(), RandomHorizontalFlip(),  ToTensor(), Cutout(), Normaliztion()]))
        dataloader_train = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=4)

        for i, sample_batched in enumerate(dataloader_train):
            # get the inputs
            inputs, binary_mask, spoof_label = sample_batched['image_x'].cuda(), sample_batched['binary_mask'].cuda(), sample_batched['spoofing_label'].cuda() 
            inputs_ir = sample_batched['image_ir'].cuda()
            
            
            optimizer.zero_grad()

            
            # forward + backward + optimize
            map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs, inputs_ir)
            #map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs, inputs_depth)

            #pdb.set_trace()
            absolute_loss = criterion_absolute_loss(map_x, binary_mask)
            contrastive_loss = criterion_contrastive_loss(map_x, binary_mask)
            
            loss =  absolute_loss + contrastive_loss
             
            loss.backward()
            
            optimizer.step()
            
            n = inputs.size(0)
            loss_absolute.update(absolute_loss.data, n)
            loss_contra.update(contrastive_loss.data, n)
            
            
        

            if i % echo_batches == echo_batches-1:    # print every 50 mini-batches
                
                # visualization
                #FeatureMap2Heatmap(x_input, x_Block1, x_Block2, x_Block3, map_x)

                # log written
                print('epoch:%d, mini-batch:%3d, lr=%f, Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f' % (epoch + 1, i + 1, lr,  loss_absolute.avg, loss_contra.avg))
        
            #break            
            
        # whole epoch average
        print('epoch:%d, Train:  Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f\n' % (epoch + 1, loss_absolute.avg, loss_contra.avg))
        log_file.write('epoch:%d, Train: Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f \n' % (epoch + 1, loss_absolute.avg, loss_contra.avg))
        log_file.flush()
           
    
            
        epoch_test = 1
        if epoch >= 0:   
        #if epoch>-1 and epoch % epoch_test == epoch_test-1:  
            model.eval()
            
            with torch.no_grad():
                ###########################################
                '''                val             '''
                ###########################################
                # val for threshold
                val_data = Spoofing_valtest(val_list, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_val = DataLoader(val_data, batch_size=2, shuffle=False, num_workers=4)
                
                map_score_list = []
                real_count = []
                fake_count = []
                totall = 0
                for i, sample_batched in enumerate(dataloader_val):
                    # get the inputs
                    inputs = sample_batched['image_x'].cuda()
                    inputs_ir = sample_batched['image_ir'].cuda()
                    string_name, binary_mask = sample_batched['string_name'], sample_batched['binary_mask'].cuda()
        
                    optimizer.zero_grad()
                    
                    
                    map_score = 0.0
                    map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs, inputs_ir)

                    for mapp in range(map_x.shape[0]):
                        #map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs[:,frame_t,:,:,:], inputs_depth[:,frame_t,:,:,:])
                        score_norm = torch.sum(map_x[mapp,:,:])/torch.sum(binary_mask[mapp,:,:])
                        map_score = score_norm
                        #print(score_norm)
                        if map_score>1:
                            score_norm = 1
                        else:
                            score_norm = 0 
                        if score_norm == 0 :
                            fake_count.append(1)
                            if string_name[mapp] > 0:
                                real_count.append(1)
                            else:
                                real_count.append(0)
                                
                        if score_norm == 0 :
                            fake_count.append(0)
                            if string_name[mapp] < 1:
                                real_count.append(0)
                            else:
                                real_count.append(1)
                        totall+=1
                        #print(string_name[mapp],score_norm)
                    map_score = map_score/map_x.shape[0]
                    
 
    
                    map_score_list.append('{} {}\n'.format( string_name[0], map_score ))
                    #print(string_name[0], map_score )
                    
                map_score_val_filename = args.log+'/'+ args.log+ '_map_score_val_%d.txt'% (epoch + 1)
                with open(map_score_val_filename, 'w') as file:
                    file.writelines(map_score_list)                
                #print("Accuracy: ",((real_count+fake_count)*100)/totall)
                print(f1_score( fake_count, real_count, average="binary"))
                print(precision_score(fake_count, real_count, average="binary"))
                print(recall_score(fake_count, real_count, average="binary"))
                
                
                precision, recall, fscore, support = score(fake_count, real_count)

                print('precision: {}'.format(precision))
                print('recall: {}'.format(recall))
                print('fscore: {}'.format(fscore))
                print('support: {}'.format(support))
                
            # save the model until the next improvement     
            torch.save(model.state_dict(), args.log+'/'+args.log+'_%d.pkl' % (epoch + 1))


    print('Finished Training')
    log_file.close()
  

  
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=3, help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')  #default=0.0001
    parser.add_argument('--batchsize', type=int, default=16, help='initial batchsize')  #default=9  
    parser.add_argument('--step_size', type=int, default=20, help='how many epochs lr decays once')  # 500  | DPC = 400
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 50
    parser.add_argument('--epochs', type=int, default=50, help='total training epochs')
    parser.add_argument('--log', type=str, default="CDCN_3modality2_P1", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')
    parser.add_argument('--theta', type=float, default=0.7, help='hyper-parameters in CDCNpp')
	
	
    args = parser.parse_args()
    train_test()
