import argparse
from Unimodal.Eval import *
import os

parser = argparse.ArgumentParser(description='PyTorch CONTINUAL LEARNING')
parser.add_argument('--model', '-m',type=str, default="MESO4", help='model name = [MESO4, MESOINCEPTION4, XCEPTION, EADPOSE, EXPLOTING, CAPSULE]')#TO BE MODIFIED
parser.add_argument('--path_video', '-v',type=str, default="/media/data1/mhkim/FAKEVV_hasam/test/FRAMES/real_A_fake_others", help='path of video')#TO BE MODIFIED
parser.add_argument('--path_video_model', '-vm',type=str, default="./Unimodal/weights/video/Meso4_realA_fakeC.pt", help='path of the video model')
parser.add_argument('--path_audio', '-a',type=str, default='', help="/media/data1/mhkim/FAKEVV_hasam/test/SPECTOGRAMS/real_A_fake_others") #TO BE MODIFIED
parser.add_argument('--path_audio_model', '-am',type=str, default='./Unimodal/weights/audio/Meso4_realA_fakeB.pt', help='path of the audio model')
parser.add_argument('--lr', '-l', type=float, default=1e-5, help='initial learning rate')
parser.add_argument('--epochs', '-me', type=int, default=50, help='epochs')
parser.add_argument('--batch_size', '-nb', type=int, default=200, help='batch size')
parser.add_argument('--num_gpu', '-ng', type=str, default='2', help='excuted gpu number')
parser.add_argument('--val_ratio', '-vr', type=float, default=0.3, help='validation ratio on trainset')
parser.add_argument('--n_early', '-ne', type=int, default=10, help='patient number of early stopping')

args = parser.parse_args()
print('GPU num is' , args.num_gpu)
os.environ['CUDA_VISIBLE_DEVICES'] =str(args.num_gpu)

MODEL = args.model
if MODEL == 'MESO4':
    Eval_MesoNet.EvalMesoNet(args)
#TO BE MODIFIED
#TO BE MODIFIED
#TO BE MODIFIED
#TO BE MODIFIED
#TO BE MODIFIED