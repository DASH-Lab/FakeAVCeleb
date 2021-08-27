from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from Common_Function_ import *
import torch.multiprocessing
from models import xception_origin

from PIL import Image

torch.multiprocessing.set_sharing_strategy('file_system')
GPU = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
calculate = False
EPOCHS = 50
BATCH_SIZE = 64
VALID_RATIO = 0.3
N_IMAGES = 100
START_LR = 1e-5
END_LR = 10
NUM_ITER = 100
PATIENCE_EARLYSTOP=10

pretrained_size = 224
pretrained_means = [0.4489, 0.3352, 0.3106]#[0.485, 0.456, 0.406]
pretrained_stds= [0.2380, 0.1965, 0.1962]#[0.229, 0.224, 0.225]

class CustumDataset(Dataset):
    def __init__(self, data, target, data_2=None, target_2=None, transform=None):
        self.data = data
        self.target = target
        self.data_video = data_2
        self.target_video = target_2
        self.transform = transform

        if self.data_video:
            self.len_data2 = len(self.data_video)
        print(self.len_data2)
        print(len(self.data_video))
        print(len(self.data))

        assert (self.len_data2 == len(self.target) == len(self.target_video) == len(self.data) == len(self.data_video))

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        path = self.data[idx]
        img = Image.open(path)
        img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)

        if self.data_video:
            path_video = self.data[idx]
            img_video = Image.open(path_video)
            img_video = img_video.convert('RGB')
            if self.transform:
                img_video = self.transform(img_video)
            return img, self.target[idx], img_video, self.target_video[idx]

train_transforms = transforms.Compose([
                           transforms.Resize((pretrained_size,pretrained_size)),
                           transforms.RandomHorizontalFlip(0.5),
                           # transforms.RandomCrop(pretrained_size, padding = 10),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = pretrained_means,
                                                std = pretrained_stds)
                       ])

test_transforms = transforms.Compose([
                           transforms.Resize((pretrained_size,pretrained_size)),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = pretrained_means,
                                                std = pretrained_stds)
                       ])

####
def getnum_of_files(path):
    _dict = {}
    for (a,b,c) in os.walk(path):
        if not b:
            _dict[a.split('/')[-1]] = len(c)
    return _dict


####
test_dir =  ["/media/data1/mhkim/FAKEVV_hasam/test/SPECTOGRAMS/real_A_fake_others",
             "/media/data1/mhkim/FAKEVV_hasam/test/FRAMES/real_A_fake_others"]

list_test = [datasets.ImageFolder(root = test_dir[0],transform = None),
            datasets.ImageFolder(root = test_dir[1],transform = None)]
print(list_test[0].targets)

#test
list_glob_testpath = [list_test[1].samples[i][0] for i in range(len(list_test[1].samples))]
list_targets_testpath = [list_test[1].targets[i] for i in range(len(list_test[1].targets))]

list_num_test = getnum_of_files(test_dir[1])
list_glob_testpath_video=[]; list_targets_testpath_video=[]
for i in range(len(list_test[0].samples)):
    _str = list_test[0].samples[i][0].split('/')[-2]
    num_repeat = int(list_num_test[_str])
    list_glob_testpath_video += [list_test[0].samples[i][0]] * num_repeat
    list_targets_testpath_video += [list_test[0].targets[i]] * num_repeat
    i = i + num_repeat
    # print(f'{str} / {num_repeat}')

assert(list_targets_testpath_video == list_targets_testpath)
test_data = CustumDataset(list_glob_testpath, list_targets_testpath, list_glob_testpath_video, list_targets_testpath_video, test_transforms)
print(f'Number of testing examples: {len(test_data)}')



pretrained_size = 224
pretrained_means = [0.4489, 0.3352, 0.3106]#[0.485, 0.456, 0.406]
pretrained_stds= [0.2380, 0.1965, 0.1962]#[0.229, 0.224, 0.225]

class Ensemble(nn.Module):
    def __init__(self, models=[], device='cuda', training=True):
        super().__init__()
        self.model1 = None
        self.model2 = None
        assert len(models) >= 2
        if models:
            self.model1 = models[0]  # FOR VIDEO
            self.model2 = models[1]  # FOR FRAME IMAGE

    def forward(self, frame, video):
        feat1 = self.model1(video)
        feat2 = self.model2(frame)
        max_feat1 = torch.argmax(feat1,1)
        max_feat2 = torch.argmax(feat2,1)
        print(feat2.size())
        out = max_feat1 + max_feat2
        out = out / 2
        out
        final = torch.zeros(out[0],2)

        # out = self.fc(out)
        print('out??????????')
        print(out)
        out = torch.argmax(out,1)
        print(out)

models = [xception_origin.xception(num_classes=2, pretrained=''), xception_origin.xception(num_classes=2, pretrained='')]
MODELS_NAME = 'Xception'
# checkpoinsts for model loaders :  [VIDEO(A&B), FRAME(A&C)]
list_checkpoint = [torch.load(f'/home/mhkim/DFVV/PRETRAINING/{MODELS_NAME}_realA_fakeB.pt',map_location='cuda')['state_dict'],
                   torch.load(f'/home/mhkim/DFVV/PRETRAINING/{MODELS_NAME}_realA_fakeC.pt',map_location='cuda')['state_dict']]
models[0].load_state_dict(list_checkpoint[0])
models[1].load_state_dict(list_checkpoint[1])

ecls = Ensemble(models,device)
# ecls = nn.DataParallel(ecls)  # 4개의 GPU를 이용할 경우
ecls = ecls.to(device)

label_encoder = LabelEncoder()
enc = OneHotEncoder(sparse=False)
y_true = np.zeros((0, 2), dtype=np.int8)
y_pred = np.zeros((0, 2), dtype=np.int8)
ecls.eval()
models[0].eval()
models[1].eval()
test_iterator = data.DataLoader(test_data,
                                shuffle = True,
                                batch_size = BATCH_SIZE)
def count(x):
    return x.value_counts().sort_values(ascending=False).index[0]


import pandas as pd
df = pd.DataFrame()
targets = [] ;y_preds_1=[] ; y_preds_2=[] ; y_preds_3=[]
for i, data in enumerate(test_iterator):
    with torch.no_grad():
        in_1 = data[0].to(device)
        target = data[1].cpu().detach().numpy() ; targets.append(target)
        in_2 = data[2].to(device)
        # _y_pred = ecls(in_1, in_2).cpu().detach()
        # int_pred_encode = label_encoder.fit_transform(_y_pred.detach().cpu())
        # int_pred_encode = int_pred_encode.reshape(len(int_pred_encode), 1)
        # onehot_pred_encode = enc.fit_transform(int_pred_encode)
        # _y_pred = torch.tensor(onehot_pred_encode)
        # print(_y_pred)
        """spectograms(video) and frames are must be matched. So, the number 2 is only True label."""
        #y_true
        y_pred_1 = models[0](in_1)
        y_pred_2 = models[1](in_2)
        y_pred_3 = (y_pred_1+y_pred_2)/2

        y_pred_1 = y_pred_1.argmax(1).detach().cpu().numpy() ; y_preds_1.append(y_pred_1)
        y_pred_2 = y_pred_2.argmax(1).detach().cpu().numpy(); y_preds_2.append(y_pred_2)
        y_pred_3 = y_pred_3.argmax(1).detach().cpu().numpy(); y_preds_3.append(y_pred_3)

y_preds_1 = np.concatenate(y_preds_1)
y_preds_2 = np.concatenate(y_preds_2)
y_preds_3 = np.concatenate(y_preds_3)

df['pred1'] = y_preds_1
df['pred2'] = y_preds_2
df['pred3'] = y_preds_3
df['hard_vote'] = df.apply(lambda x: count(x), 1)

soft = df.loc[(df['pred1']!=df['pred2'])]['pred3'].copy()
df.loc[(df['pred1'] != df['pred2']), 'hard_vote'] = soft

targets = np.concatenate(targets) ; print(targets.shape)
print(df.shape)
df['target'] = targets
print(f'accuracy : {accuracy_score(df["target"], df["hard_vote"])*100:.2f}')
print(df["hard_vote"])

result = classification_report(df["target"], df["hard_vote"], labels=None, target_names=None, sample_weight=None, digits=5,
                               output_dict=False, zero_division='warn')

print(result)
