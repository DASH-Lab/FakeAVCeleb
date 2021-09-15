"""
Copyright (c) 2019, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------
Script for testing Capsule-Forensics-v2 on FaceForensics++ database (Real, DeepFakes, Face2Face, FaceSwap)
"""

import sys
sys.setrecursionlimit(15000)
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torch.autograd import Variable
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import model_big
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn import metrics
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default ='/media/data1/hsm/acm_workshop/', help='path to dataset')
parser.add_argument('--test_set', default ='testdata/FRAMSc20', help='test set')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imageSize', type=int, default=300, help='the height / width of the input image to network')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--outf', default='checkpoints/binary_faceforensicspp', help='folder to output model checkpoints')
parser.add_argument('--random', action='store_true', default=False, help='enable randomness for routing matrix')
parser.add_argument('--id', type=int, default=1, help='checkpoint ID')

opt = parser.parse_args()
print(opt)

if __name__ == '__main__':

    text_writer = open(os.path.join(opt.outf, 'test.txt'), 'w')

    transform_fwd = transforms.Compose([
        transforms.Resize((opt.imageSize, opt.imageSize)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


    dataset_test = dset.ImageFolder(root=os.path.join(opt.dataset, opt.test_set), transform=transform_fwd)
    assert dataset_test
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))

    vgg_ext = model_big.VggExtractor()
    capnet = model_big.CapsuleNet(2, opt.gpu_id)

    capnet.load_state_dict(torch.load(os.path.join(opt.outf,'capsule_' + str(opt.id) + '.pt')))
    capnet.eval()

    if opt.gpu_id >= 0:
        vgg_ext.cuda(opt.gpu_id)
        capnet.cuda(opt.gpu_id)


    ##################################################################################

    tol_label = np.array([], dtype=np.float)
    tol_pred = np.array([], dtype=np.float)
    tol_pred_prob = np.array([], dtype=np.float)

    count = 0
    loss_test = 0

    for img_data, labels_data in tqdm(dataloader_test):

        labels_data[labels_data > 1] = 1
        img_label = labels_data.numpy().astype(np.float)

        if opt.gpu_id >= 0:
            img_data = img_data.cuda(opt.gpu_id)
            labels_data = labels_data.cuda(opt.gpu_id)

        input_v = Variable(img_data)

        x = vgg_ext(input_v)
        classes, class_ = capnet(x, random=opt.random)

        output_dis = class_.data.cpu()
        output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)

        for i in range(output_dis.shape[0]):
            if output_dis[i,1] >= output_dis[i,0]:
                output_pred[i] = 1.0
            else:
                output_pred[i] = 0.0

        tol_label = np.concatenate((tol_label, img_label))
        tol_pred = np.concatenate((tol_pred, output_pred))
        
        pred_prob = torch.softmax(output_dis, dim=1)
        tol_pred_prob = np.concatenate((tol_pred_prob, pred_prob[:,1].data.numpy()))

        count += 1

    acc_test = metrics.accuracy_score(tol_label, tol_pred)
    loss_test /= count

    precision, recall, fscore, support = score(tol_label, tol_pred)

    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))
                
    fpr, tpr, thresholds = roc_curve(tol_label, tol_pred_prob, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    # fnr = 1 - tpr
    # hter = (fpr + fnr)/2

    print('[Epoch %d] Test acc: %.2f   EER: %.2f' % (opt.id, acc_test*100, eer*100))
    text_writer.write('%d,%.2f,%.2f\n'% (opt.id, acc_test*100, eer*100))

    text_writer.flush()
    text_writer.close()

    y_true = tol_label
    y_pred = tol_pred
    result = metrics.classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=4, output_dict=False, zero_division='warn')    
    print(result)
    print(f'ACC is {metrics.accuracy_score(y_true, y_pred)}')
    y_true, y_pred = np.array(y_true),np.array(y_pred)
    print(y_true.shape, y_pred.shape)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true,y_pred)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_true,y_pred)
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    print(roc_auc[1])
    plt.plot(fpr[1], tpr[1], color='darkred', lw=lw, label='ROC curve ({:.2f})'.format(roc_auc[1]))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('EfficientNet-B0')
    plt.legend(loc="lower right")
    plt.show()
