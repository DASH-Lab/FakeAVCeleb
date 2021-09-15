import os
from utils.face_proc import FaceProc
import argparse
import pickle
from forensic_test import exam_img, exam_video
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

def main(args):
    tol_pred = []
    labs = []
    for dirr in os.listdir(args.input_dir):
        print(os.path.join(args.input_dir,dirr))
        pth = os.path.join(args.input_dir,dirr)
        all_paths = os.listdir(pth)
        proba_list = []

        # initiate face process class, used to detect face and extract landmarks
        face_inst = FaceProc()

        # initialize SVM classifier for face forensics
        with open(args.classifier_path, 'rb') as f:
            model = pickle.load(f)
        classifier = model[0]
        scaler = model[1]

        for f_name in all_paths:
            f_path = os.path.join(pth, f_name)
            print('_'*20)
            print('Testing: ' + f_name)
            suffix = f_path.split('.')[-1]
            if suffix.lower() in ['jpg', 'png', 'jpeg', 'bmp']:
                proba, optout = exam_img(args, f_path, face_inst, classifier, scaler)
            elif suffix.lower() in ['mp4', 'avi', 'mov', 'mts']:
                proba, optout = exam_video(args, f_path, face_inst, classifier, scaler)
            print('fake_proba: {},   optout: {}'.format(str(proba), optout))
            tmp_dict = dict()
            tmp_dict['file_name'] = f_name
            tmp_dict['probability'] = proba
            tmp_dict['optout'] = optout
            proba_list.append(tmp_dict)
            
            if dirr == 'REAL0':
                labs.append(0.0)
                if proba < 0.5:
                    tol_pred.append(0.0)
                else:
                    tol_pred.append(1.0)
            else:
                labs.append(1.0)                
                if proba > 0.5:
                    tol_pred.append(1.0)
                else:
                    tol_pred.append(0.0)
                
    #precision, recall, fscore, support = score(labs, tol_pred)
    #print('precision: {}'.format(precision))
    #print('recall: {}'.format(recall))
    #print('fscore: {}'.format(fscore))
    #print('support: {}'.format(support))
    #print(metrics.accuracy_score(labs, tol_pred))
    
    pickle.dump(proba_list, open(args.save_file, 'wb'))
    
    y_true = labs
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



if __name__ == '__main__':
   parser = argparse.ArgumentParser(description="headpose forensics")
   parser.add_argument('--input_dir', type=str, default='/TEST/VIDEOSFRAMES')
   parser.add_argument('--markID_c', type=str, default='18-36,49,55', help='landmark ids to estimate CENTRAL face region')
   parser.add_argument('--markID_a', type=str, default='1-36,49,55', help='landmark ids to estimate WHOLE face region')
   parser.add_argument('--classifier_path', type=str, default='./model')
   parser.add_argument('--save_file', type=str, default='proba_list.p')
   args = parser.parse_args()
   main(args)