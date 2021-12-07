import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import copy
import time
from torch.cuda.amp import GradScaler
from utils.EarlyStopping import EarlyStopping
from utils.Common_Function import *
from models.MesoNet import Meso4


def TrainMesoNet(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LIST_SELECT = ('VIDEO' if os.path.exists(args.path_video) else '',
                   'AUDIO' if os.path.exists(args.path_audio) else '')
    assert (LIST_SELECT[0]!='' and LIST_SELECT[1]!='', 'At least one path must be typed')
    print(LIST_SELECT)
    tu_video, tu_audio = None, None
    if args.path_video:
        tu_video = (args.path_video)
    if args.path_audio:
        tu_audio = (args.path_audio)

    for MODE in LIST_SELECT:
        train_dir = None
        if MODE == 'VIDEO':
            train_dir = tu_video
        elif MODE == 'AUDIO':
            train_dir = tu_audio
        
        if train_dir is None:
            continue

        EPOCHS = args.epochs
        BATCH_SIZE = args.batch_size
        VALID_RATIO = args.val_ratio
        START_LR = args.lr
        PATIENCE_EARLYSTOP = args.n_early
        SAVE_PATH = args.path_save

        pretrained_size = 224
        pretrained_means = [0.4489, 0.3352, 0.3106]  # [0.485, 0.456, 0.406]
        pretrained_stds = [0.2380, 0.1965, 0.1962]  # [0.229, 0.224, 0.225]
        train_transforms = transforms.Compose([
            transforms.Resize((pretrained_size, pretrained_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=pretrained_means,
                                 std=pretrained_stds)
        ])

        test_transforms = transforms.Compose([
            transforms.Resize((pretrained_size, pretrained_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=pretrained_means,
                                 std=pretrained_stds)
        ])
        train_data = datasets.ImageFolder(root=train_dir,
                                          transform=train_transforms)

        n_valid_examples = int(len(train_data) * VALID_RATIO)  # 기존 test data자체가 너무 적어서 train기준으로 비율조정
        n_train_examples = len(train_data) - n_valid_examples

        train_data, valid_data = data.random_split(train_data,
                                                   [n_train_examples, n_valid_examples])
        valid_data = copy.deepcopy(valid_data)
        valid_data.dataset.transform = test_transforms

        print(f'Number of training examples: {len(train_data)}')
        print(f'Number of validation examples: {len(valid_data)}')
        train_iterator = data.DataLoader(train_data,
                                         shuffle=True,
                                         batch_size=BATCH_SIZE)

        valid_iterator = data.DataLoader(valid_data,
                                         shuffle=True,
                                         batch_size=BATCH_SIZE)

        print(f'number of train/val/test loader : {len(train_iterator), len(valid_iterator)}')

        model = Meso4()
        if len(args.num_gpu) > 1:
            model = nn.DataParallel(model)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        scaler = GradScaler()
        early_stopping = EarlyStopping(patience=PATIENCE_EARLYSTOP, verbose=True)

        optimizer = optim.Adam(model.parameters(), lr=START_LR)
        best_valid_loss = float('inf')
        print("training...")
        for epoch in range(EPOCHS):

            start_time = time.monotonic()

            train_loss, train_acc_1, train_acc_5 = train(model, train_iterator, optimizer, criterion, scaler, device)
            valid_loss, valid_acc_1, valid_acc_5 = evaluate(model, valid_iterator, criterion, device)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save({'state_dict': model.state_dict(),
                            'best_acc': valid_acc_1,
                            'val_loss': valid_loss,
                            'epoch': epoch,
                            'lr': START_LR,
                            'best_acc': valid_acc_1,
                            }, f'{SAVE_PATH}/best_{args.model}_{MODE}.pt')

            end_time = time.monotonic()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc @1: {train_acc_1 * 100:6.2f}% | ' \
                  f'Train Acc @5: {train_acc_5 * 100:6.2f}%')
            print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc @1: {valid_acc_1 * 100:6.2f}% | ' \
                  f'Valid Acc @5: {valid_acc_5 * 100:6.2f}%')

            if early_stopping:
                early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
