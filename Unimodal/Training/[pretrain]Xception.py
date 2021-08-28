import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import copy
import time
from torch.cuda.amp import GradScaler
from EarlyStopping import EarlyStopping
from Common_Function import *
from models import xception_origin

set_seeds()
#
# train_dir = "/media/data1/mhkim/FAKEVV_hasam/train/FRAMES/real_A_fake_MergedC"
# test_dir =  "/media/data1/mhkim/FAKEVV_hasam/test/FRAMES/real_A_fake_MergedC"

train_dir = "/media/data1/mhkim/FAKEVV_hasam/train/SPECTOGRAMS/real_A_fake_B"
test_dir =  "/media/data1/mhkim/FAKEVV_hasam/test/SPECTOGRAMS/real_A_fake_B"
train_data = datasets.ImageFolder(root = train_dir,
                                  transform = transforms.ToTensor())
calculate = False
EPOCHS = 50
BATCH_SIZE = 200
VALID_RATIO = 0.3
N_IMAGES = 100
START_LR = 1e-5
END_LR = 10
NUM_ITER = 100
PATIENCE_EARLYSTOP=10

pretrained_size = 224
pretrained_means = [0.4489, 0.3352, 0.3106]#[0.485, 0.456, 0.406]
pretrained_stds= [0.2380, 0.1965, 0.1962]#[0.229, 0.224, 0.225]
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
train_data = datasets.ImageFolder(root = train_dir, 
                                  transform = train_transforms)

test_data = datasets.ImageFolder(root = test_dir, 
                                 transform = test_transforms)
                                 

n_valid_examples = int(len(train_data) * VALID_RATIO)#기존 test data자체가 너무 적어서 train기준으로 비율조정
n_train_examples = len(train_data) - n_valid_examples

train_data, valid_data = data.random_split(train_data, 
                                           [n_train_examples, n_valid_examples])
valid_data = copy.deepcopy(valid_data)
valid_data.dataset.transform = test_transforms

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

train_iterator = data.DataLoader(train_data, 
                                 shuffle = True, 
                                 batch_size = BATCH_SIZE)

valid_iterator = data.DataLoader(valid_data, 
                                 shuffle = True, 
                                 batch_size = BATCH_SIZE)

test_iterator = data.DataLoader(test_data, 
                                shuffle = True, 
                                batch_size = BATCH_SIZE)


print(f'number of train/val/test loader : {len(train_iterator), len(valid_iterator), len(test_iterator)}')
model = xception_origin.xception(num_classes=2, pretrained='')
OUTPUT_DIM = 2
print(f'OUTPUT_DIM is {OUTPUT_DIM}')

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss().to(device)
model = model.to(device)
scaler = GradScaler()
early_stopping = EarlyStopping(patience=PATIENCE_EARLYSTOP, verbose=True)

optimizer = optim.Adam(model.parameters(), lr = START_LR)
STEPS_PER_EPOCH = len(train_iterator)
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH
best_valid_loss = float('inf')

print("training...")
for epoch in range(EPOCHS):
    
    start_time = time.monotonic()
    train_loss, train_acc_1, train_acc_5 = train(model, train_iterator, optimizer, criterion, scaler, device)
    valid_loss, valid_acc_1, valid_acc_5 = evaluate(model, valid_iterator, criterion, device)
        
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save({'state_dict':model.state_dict(),
                   'best_acc':valid_acc_1,
                   'val_loss':valid_loss,
                   'epoch':epoch,
                   'lr':START_LR,
                   'best_acc':valid_acc_1,
                   }, 'Xception_realA_fakeB.pt')

    end_time = time.monotonic()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc @1: {train_acc_1*100:6.2f}% | ' \
          f'Train Acc @5: {train_acc_5*100:6.2f}%')
    print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc @1: {valid_acc_1*100:6.2f}% | ' \
          f'Valid Acc @5: {valid_acc_5*100:6.2f}%')
    
    if early_stopping:
        early_stopping(valid_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    