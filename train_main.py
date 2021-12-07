import argparse
from Unimodal.Training import *
from utils.Common_Function import *
import torch.multiprocessing

parser = argparse.ArgumentParser(description='PyTorch CONTINUAL LEARNING')
parser.add_argument('--model', '-m',type=str, default="MESO4", help='model name = [MESO4, MESOINCEPTION4, XCEPTION, EFFICIENTB0, F3NET, LIPS, XRAY, HEADPOSE, EXPLOTING, CAPSULE]')#TO BE MODIFIED
# parser.add_argument('--path_video', '-v',type=str, default="", help='path of path of frame (video)')#TO BE MODIFIED
# parser.add_argument('--path_audio', '-a',type=str, default="/media/data1/mhkim/FakeAVCeleb_PREPROCESSED/SPECTROGRAM/B/TRAIN", help="path of spectogram (audio)") #TO BE MODIFIED
parser.add_argument('--path_video', '-v',type=str, default="/media/data1/mhkim/FakeAVCeleb_PREPROCESSED/FRAMES_PNG/C/TRAIN", help='path of path of frame (video)')#TO BE MODIFIED
parser.add_argument('--path_audio', '-a',type=str, default="", help="path of spectogram (audio)") #TO BE MODIFIED
parser.add_argument('--path_save', '-sm',type=str, default='./', help='path to save model while training')
parser.add_argument('--lr', '-l', type=float, default=1e-5, help='initial learning rate')
parser.add_argument('--epochs', '-me', type=int, default=50, help='epochs')
parser.add_argument('--batch_size', '-nb', type=int, default=200, help='batch size')
parser.add_argument('--num_gpu', '-ng', type=str, default='0', help='excuted gpu number')
parser.add_argument('--val_ratio', '-vr', type=float, default=0.3, help='validation ratio on trainset')
parser.add_argument('--n_early', '-ne', type=int, default=10, help='patient number of early stopping')

args = parser.parse_args()
set_seeds()
torch.multiprocessing.set_sharing_strategy('file_system')
print('GPU num is' , args.num_gpu)
os.environ['CUDA_VISIBLE_DEVICES'] =str(args.num_gpu)
MODEL = args.model

if MODEL == 'MESO4':
   Train_MesoNet.TrainMesoNet(args)
elif MODEL == 'MESOINCEPTION4':
   Train_MesoInceptionNet.TrainMesoInceptionNet(args)
elif MODEL == 'XCEPTION' or MODEL == 'XRAY' or MODEL == 'LIPS':
      if MODEL == 'LIPS':
            args.path_video = args.path_video.replace('FRAMES_PNG','FRAMES_LIPS_PNG')
      if MODEL == 'XRAY':
            args.path_video = args.path_video.replace('FRAMES_PNG','FRAMES_XRAY_PNG')     
      Train_Xception.TrainXception(args)
elif MODEL == 'EFFICIENTB0':
      Train_EfficientB0.TrainEfficientB0(args)
elif MODEL == 'VGG':
      Train_VGG16.TrainVGG16(args)
elif MODEL == 'F3NET':
      Train_F3Net.TrainF3Net(args)