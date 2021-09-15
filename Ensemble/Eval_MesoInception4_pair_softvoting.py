from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils.Common_Function import *
import torch.multiprocessing
from models.MesoNet4_forEnsemble import MesoInception4 as MesoNet
from PIL import Image
import copy
import torch.nn as nn

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
        return img, self.target[idx]

def getnum_of_files(path):
    _dict = {}
    for (a,b,c) in os.walk(path):
        if not b:
            _dict[a.split('/')[-1]] = len(c)
    return _dict

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
        out = feat1 + feat2
        out = out / 2
        return out

def Eval(args):
    BATCH_SIZE = args.batch_size
    test_dir = ["/media/data1/mhkim/FAKEVV_hasam/test/SPECTOGRAMS/real_A_fake_others",
                "/media/data1/mhkim/FAKEVV_hasam/test/FRAMES/real_A_fake_others"]

    list_checkpoint = [torch.load(f'/home/mhkim/DFVV/PRETRAINING/MesoInception4_realA_fakeB.pt')['state_dict'],
                       torch.load(f'/home/mhkim/DFVV/PRETRAINING/MesoInception4_realA_fakeC.pt')['state_dict']]
    # test_dir = [args.path_audio, args.path_video]
    # list_checkpoint = [args.path_audio_model, args.path_video_model]
    assert (not test_dir or not list_checkpoint, 'wrong path param !!!')

    pretrained_size = 224
    pretrained_means = [0.4489, 0.3352, 0.3106]#[0.485, 0.456, 0.406]
    pretrained_stds= [0.2380, 0.1965, 0.1962]#[0.229, 0.224, 0.225]

    test_transforms = transforms.Compose([
                               transforms.Resize((pretrained_size,pretrained_size)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean = pretrained_means,
                                                    std = pretrained_stds)
                           ])
    list_test = [datasets.ImageFolder(root=test_dir[0], transform=None),
                 datasets.ImageFolder(root=test_dir[1], transform=None)]

    # test
    list_glob_testpath = [list_test[1].samples[i][0] for i in range(len(list_test[1].samples))]
    list_targets_testpath = [list_test[1].targets[i] for i in range(len(list_test[1].targets))]

    list_num_test = getnum_of_files(test_dir[1])
    list_glob_testpath_video = [];
    list_targets_testpath_video = []
    for i in range(len(list_test[0].samples)):
        _str = list_test[0].samples[i][0].split('/')[-2]
        num_repeat = int(list_num_test[_str])
        list_glob_testpath_video += [list_test[0].samples[i][0]] * num_repeat
        list_targets_testpath_video += [list_test[0].targets[i]] * num_repeat
        i = i + num_repeat

    assert (list_targets_testpath_video == list_targets_testpath)
    test_data = CustumDataset(list_glob_testpath, list_targets_testpath, list_glob_testpath_video,
                              list_targets_testpath_video, test_transforms)
    print(f'Number of testing examples: {len(test_data)}')

    models = [MesoNet(), MesoNet()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #checkpoinsts for model loaders :  [VIDEO(A&B), FRAME(A&C)]
    models[0].load_state_dict(list_checkpoint[0])
    models[1].load_state_dict(list_checkpoint[1])

    ecls = Ensemble(models,device)
    ecls = nn.DataParallel(ecls)  # 4개의 GPU를 이용할 경우
    ecls = ecls.to(device)
    label_encoder = LabelEncoder()
    enc = OneHotEncoder(sparse=False)

    y_true = np.zeros((0, 2), dtype=np.int8)
    y_pred = np.zeros((0, 2), dtype=np.int8)
    y_true_auc = []
    y_pred_auc = []
    ecls.eval()
    test_iterator = DataLoader(test_data,
                                    shuffle = True,
                                    batch_size = BATCH_SIZE)

    for i, data in enumerate(test_iterator):
        with torch.no_grad():
            in_1 = data[0].to(device)
            in_2 = data[2].to(device)
            _y_pred = ecls(in_1, in_2).cpu().detach()

            _pred = copy.deepcopy(_y_pred).detach().cpu()  # .tolist()
            _true = copy.deepcopy(data[1]).detach().cpu().float().tolist()
            [y_pred_auc.append(_a) for _a in _pred[:, 1]]
            [y_true_auc.append(_a) for _a in _true]
            integer_encoded = label_encoder.fit_transform(data[1].detach().cpu())
            integer_encoded_2 = label_encoder.fit_transform(data[3].detach().cpu())
            integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
            integer_encoded_2 = integer_encoded_2.reshape(len(integer_encoded_2), 1)

            onehot_encoded = enc.fit_transform(integer_encoded)
            onehot_encoded_2 = enc.fit_transform(integer_encoded_2)
            onehot_encoded = onehot_encoded.astype(np.int8)
            onehot_encoded_2 = onehot_encoded_2.astype(np.int8)

            _y_true = torch.tensor(onehot_encoded + onehot_encoded_2)
            _y_true_argmax = _y_true.argmax(1)
            _y_true = np.array(torch.zeros(_y_true.shape).scatter(1, _y_true_argmax.unsqueeze(1), 1), dtype=np.int8)
            y_true = np.concatenate((y_true, _y_true))
            a = _y_pred.argmax(1)
            _y_pred = np.array(torch.zeros(_y_pred.shape).scatter(1, a.unsqueeze(1), 1), dtype=np.int8)

            y_pred = np.concatenate((y_pred, _y_pred))

    result = classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=5,
                                   output_dict=False, zero_division='warn')

    print(result)
    print(f'ACC is {accuracy_score(y_true, y_pred)}')
    import sklearn.metrics as metrics
    import matplotlib.pyplot as plt
    y_true_auc, y_pred_auc = np.array(y_true_auc), np.array(y_pred_auc)
    print(y_true_auc.shape, y_pred_auc.shape)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true_auc, y_pred_auc)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_true_auc, y_pred_auc)
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    print('ROC : {:.3f}'.format(roc_auc[1]))
    plt.plot(fpr[1], tpr[1], color='darkred', lw=lw, label='ROC curve ({:.3f})'.format(roc_auc[1]))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{args.model}')
    plt.legend(loc="lower right")
    plt.show()
