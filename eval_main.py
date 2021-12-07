import argparse
from Unimodal.Eval import *
import os

parser = argparse.ArgumentParser(description='PyTorch CONTINUAL LEARNING')
parser.add_argument('--model', '-m',type=str, default="LIPS", help='model name = [MESO4, MESOINCEPTION4, XCEPTION, EADPOSE]')#TO BE MODIFIED
parser.add_argument('--path_video_model', '-vm',type=str, default="", help='path of the video model')
parser.add_argument('--path_audio_model', '-am',type=str, default="", help='path of the audio model')
parser.add_argument('--path_audio', '-a',type=str, default="/media/data1/mhkim/FakeAVCeleb_PREPROCESSED/SPECTROGRAM/B/TEST", help="path of spectogram (audio)") #TO BE MODIFIED
parser.add_argument('--path_video', '-v',type=str, default="/media/data1/mhkim/FakeAVCeleb_PREPROCESSED/FRAMES_PNG/C/TEST", help='path of path of frame (video)')#TO BE MODIFIED

parser.add_argument('--batch_size', '-nb', type=int, default=200, help='batch size for evaluation')
parser.add_argument('--num_gpu', '-ng', type=str, default='3', help='excuted gpu number')

args = parser.parse_args()
print('GPU num is' , args.num_gpu)
os.environ['CUDA_VISIBLE_DEVICES'] =str(args.num_gpu)

MODEL = args.model
if not args.path_video_model : args.path_video_model = f'./best_{MODEL}_VIDEO.pt'
args.path_audio_model = f'./best_{MODEL}_AUDIO.pt'

if MODEL == 'MESO4':
   Eval_MesoNet.Eval(args)
elif MODEL == 'MESOINCEPTION4': 
   Eval_MesoInceptionNet.Eval(args)
elif MODEL == 'XCEPTION' or MODEL == 'XRAY' or MODEL == 'LIPS':
   if MODEL == 'LIPS':
      args.path_video = args.path_video.replace('FRAMES_PNG','FRAMES_LIPS_PNG')
   if MODEL == 'XRAY':
      args.path_video = args.path_video.replace('FRAMES_PNG','FRAMES_XRAY_PNG')       
   Eval_Xception.Eval(args)
elif MODEL == 'VGG':
   Eval_VGG16.Eval(args)
elif MODEL == 'EFFICIENTB0':
   Eval_EfficientB0.Eval(args)
elif MODEL == 'F3NET':
   Eval_F3Net.Eval(args)