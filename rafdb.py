import os
import warnings
from tqdm import tqdm
import argparse

from PIL import Image
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from sklearn.metrics import confusion_matrix

from sklearn.metrics import balanced_accuracy_score

from networks.dan import DAN

def warn(*args, **kwargs):
    pass
warnings.warn = warn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='/content/drive/MyDrive/FER/DAN/raf-basic/', help='Raf-DB dataset path.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate for sgd.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=40, help='Total training epochs.')
    parser.add_argument('--num_head', type=int, default=4, help='Number of attention head.')
    parser.add_argument('--load_model', type=str, default='models/rafdb_epoch21_acc0.897_bacc0.8532.pth', help='Load pretrained model on RAF-DB')
    parser.add_argument('--test_on', type=str, default='raf', help='Test on RAF DB or other dataset')
    parser.add_argument('--cm_save_path', type=str, default='/content/DAN/', help='Path to save confusion matrix text file')
    parser.add_argument('--mode', type=str, default='test', help='Set to train or test mode')
    return parser.parse_args()


class EvpDataSet(data.Dataset):
    def __init__(self, evp_path, transform = None):
        self.transform = transform
        self.evp_path = evp_path
        self.txt_path = os.path.join(self.evp_path, 'dan-evp.txt')

        self.file_paths, self.label = self.parse_file(self.txt_path)
        
        _, self.sample_counts = np.unique(self.label, return_counts=True)
        

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')        
        label = self.label[idx]

        if self.transform is not None:
            image = self.transform(image)
           
        return image, label

    def parse_file(self, txt_path):
      file_names = []
      labels = np.array([])
      with open(txt_path, "r") as f:
        contents = f.read().splitlines()
      
      i = 0
      while i < len(contents):
        if contents[i] == "vid_dir":
          root_path = contents[i + 1]
          i += 2
        
        fname, label = contents[i].split()
        file_names += [os.path.join(root_path, fname)]
        labels = np.append(labels, int(label))
        i += 1

      assert len(file_names) == len(labels), "Length of images not equal to labels"
      return file_names, labels




class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform = None):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path

        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None,names=['name','label'])

        if phase == 'train':
            self.data = df[df['name'].str.startswith('train')]
        else:
            self.data = df[df['name'].str.startswith('test')]

        file_names = self.data.loc[:, 'name'].values
        self.label = self.data.loc[:, 'label'].values - 1 # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

        _, self.sample_counts = np.unique(self.label, return_counts=True)
        # print(f' distribution of {phase} samples: {self.sample_counts}')

        self.file_paths = []
        for f in file_names:
            f = f.split(".")[0]
            f = f +"_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.label[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

class AffinityLoss(nn.Module):
    def __init__(self, device, num_class=8, feat_dim=512):
        super(AffinityLoss, self).__init__()
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.device = device

        self.centers = nn.Parameter(torch.randn(self.num_class, self.feat_dim).to(device))

    def forward(self, x, labels):
        x = self.gap(x).view(x.size(0), -1)

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_class) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_class, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_class).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_class)
        mask = labels.eq(classes.expand(batch_size, self.num_class))

        dist = distmat * mask.float()
        dist = dist / self.centers.var(dim=0).sum()

        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

class PartitionLoss(nn.Module):
    def __init__(self, ):
        super(PartitionLoss, self).__init__()
    
    def forward(self, x):
        num_head = x.size(1)

        if num_head > 1:
            var = x.var(dim=1).mean()
            loss = torch.log(1+num_head/var)
        else:
            loss = 0
            
        return loss


def write_cm_log(args, cm, log_name, mode, norm):
    if not norm:
        header = "Confusion matrix"
    else:
        header = "Confusion matrix(normalized)"
        cm = np.around(cm, 4)
        
    conf_m_str = "\n" + header + "\n" + "".join([str(row) + "\n" for row in cm])
    with open(os.path.join(args.cm_save_path, log_name), mode) as f:
      f.write(conf_m_str)


def run_training():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    model = DAN(num_head=args.num_head)
    model.to(device)

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
                transforms.RandomRotation(20),
                transforms.RandomCrop(224, padding=32)
            ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02,0.25)),
        ])
    
    train_dataset = RafDataSet(args.raf_path, phase = 'train', transform = data_transforms)    
    
    print('Whole train set size:', train_dataset.__len__())

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = True,  
                                               pin_memory = True)

    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])   

    val_dataset = RafDataSet(args.raf_path, phase = 'test', transform = data_transforms_val)   

    print('Validation set size:', val_dataset.__len__())
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)

    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_af = AffinityLoss(device)
    criterion_pt = PartitionLoss()

    params = list(model.parameters()) + list(criterion_af.parameters())
    optimizer = torch.optim.SGD(params,lr=args.lr, weight_decay = 1e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # write empty file
    log_name = "train_log_"  + args.test_on
    with open(os.path.join(args.cm_save_path, log_name), "w") as f:
      f.write("")

    best_acc = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        print("\nEpoch", epoch)
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()

        for (imgs, targets) in train_loader:
            print("   training ...")
            iter_cnt += 1
            optimizer.zero_grad()

            imgs = imgs.to(device)
            targets = targets.to(device)
            
            out,feat,heads = model(imgs)

            loss = criterion_cls(out,targets) + 1* criterion_af(feat,targets) + 1*criterion_pt(heads)  #89.3 89.4

            loss.backward()
            optimizer.step()
            
            running_loss += loss
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss/iter_cnt
        tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (epoch, acc, running_loss,optimizer.param_groups[0]['lr']))

        
        all_predictions = torch.Tensor().to(device)
        all_targets = torch.Tensor().to(device)
        with torch.no_grad():
            print("   validating ...")
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            baccs = []

            model.eval()
            for (imgs, targets) in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                
                out,feat,heads = model(imgs)
                loss = criterion_cls(out,targets) + criterion_af(feat,targets) + criterion_pt(heads)

                running_loss += loss
                iter_cnt+=1
                _, predicts = torch.max(out, 1)
                correct_num  = torch.eq(predicts,targets)
                all_predictions = torch.cat((all_predictions, predicts), dim=0)
                all_targets = torch.cat((all_targets, targets), dim=0)

                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += out.size(0)
                
                baccs.append(balanced_accuracy_score(targets.cpu().numpy(),predicts.cpu().numpy()))
            running_loss = running_loss/iter_cnt   
            scheduler.step()

            acc = bingo_cnt.float()/float(sample_cnt)
            acc = np.around(acc.numpy(),4)
            best_acc = max(acc,best_acc)
            
            bacc = np.around(np.mean(baccs),4)
            tqdm.write("[Epoch %d] Validation accuracy:%.4f. bacc:%.4f. Loss:%.3f" % (epoch, acc, bacc, running_loss))
            tqdm.write("best_acc:" + str(best_acc))
            
            all_targets = all_targets.cpu()
            all_predictions = all_predictions.cpu()
            cm = confusion_matrix(all_targets, all_predictions, labels=[0,1,2,3,4,5,6])
            cm_norm = confusion_matrix(all_targets, all_predictions, labels=[0,1,2,3,4,5,6], normalize='true')
        
            write_cm_log(args, cm, log_name, "a", norm=False)
            write_cm_log(args, cm_norm, log_name, "a", norm=True)

            if acc > 0.89 and acc == best_acc:
                if not os.path.isdir('checkpoints'):
                    os.system('mkdir ' + 'checkpoints')
                torch.save({'iter': epoch,
                            'model_state_dict': model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),},
                            os.path.join('checkpoints', "rafdb_epoch"+str(epoch)+"_acc"+str(acc)+"_bacc"+str(bacc)+".pth"))
                tqdm.write('Model saved.')
                
            if epoch == args.epochs:
                print("Confusion matrix")
                print(cm)
                
                print("Confusion matrix (Normalized)")
                print(cm_norm)


def run_testing():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    model = DAN(num_head=args.num_head)
    model.to(device)
    state_dict = torch.load(args.load_model)['model_state_dict']
    model.load_state_dict(state_dict)

    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])   


    if args.test_on == 'raf':
        val_dataset = RafDataSet(args.raf_path, phase = 'test', transform = data_transforms_val)   
    else:
        val_dataset = EvpDataSet('/content/drive/MyDrive/FER/DAN/', transform = data_transforms_val)

    print('Test set size:', val_dataset.__len__())
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)

    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_af = AffinityLoss(device)
    criterion_pt = PartitionLoss()

    all_predictions = torch.Tensor().to(device)
    all_targets = torch.Tensor().to(device)
    with torch.no_grad():
        iter_cnt = 0
        bingo_cnt = 0
        sample_cnt = 0
        
        model.eval()
        for (imgs, targets) in val_loader:
            print("iter", iter_cnt + 1)
            imgs = imgs.to(device)
            targets = targets.to(device)    
            out,feat,heads = model(imgs)

            iter_cnt+=1
            _, predicts = torch.max(out, 1)
            correct_num  = torch.eq(predicts,targets)
            all_predictions = torch.cat((all_predictions, predicts), dim=0)
            all_targets = torch.cat((all_targets, targets), dim=0)

            bingo_cnt += correct_num.sum().cpu()
            sample_cnt += out.size(0)
            
        acc = bingo_cnt.float()/float(sample_cnt)
        acc = np.around(acc.numpy(),4)

        print("Validation accuracy: {}".format(acc))
        all_targets = all_targets.cpu()
        all_predictions = all_predictions.cpu()
        cm = confusion_matrix(all_targets, all_predictions, labels=[0,1,2,3,4,5,6])
        cm_norm = confusion_matrix(all_targets, all_predictions, labels=[0,1,2,3,4,5,6], normalize='true')
        
        write_cm_log(args, cm, "test_log_" + args.test_on, "w", norm=False)
        write_cm_log(args, cm_norm, "test_log_" + args.test_on, "a", norm=True)


def driver():
  args = parse_args()
  if args.mode == "train":
    run_training()
  else:
    run_testing()
        
if __name__ == "__main__":    
    print("RUNning")    
    driver()
