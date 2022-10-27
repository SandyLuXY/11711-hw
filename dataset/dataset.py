from torch.utils.data import Dataset
import numpy as np
import torch

class TXTDataset(Dataset):
    def __init__(self, path, train=True):
        self.path = path
        self.txtdat = np.load(path, allow_pickle=True)
        self.train = train

    def __len__(self):
        return len(self.txtdat)

    def __getitem__(self, idx):
        if self.train:
            # 0 feats 2 token
            return [self.txtdat[idx][0], self.txtdat[idx][2]]
        else:
            return self.txtdat[idx]
class TXTDataset2(Dataset):
    def __init__(self, path1, path2, train=True):
        self.path1 = path1
        self.txtdat1 = np.load(path2, allow_pickle=True)
        self.path2 = path2
        self.txtdat2 = np.load(path2, allow_pickle=True)
        self.train = train

    def __len__(self):
        return len(self.txtdat1)

    def __getitem__(self, idx):
        if self.train:
            # 0 feats 2 token
            return [self.txtdat1[idx][0], self.txtdat2[idx][0], self.txtdat1[idx][2]]
        else:
            return self.txtdat1[idx], self.txtdat2[idx]

def collate_train(batch):
    maxlen = 0
    for pairs in batch:
        feat, label = pairs
        maxlen = max(maxlen, len(label))
    feats = []
    labels = []
    for pairs in batch:
        feat, label = pairs
        feats.append(np.pad(feat, ((0,maxlen-feat.shape[0]), (0, 0)), 'constant', constant_values=0))
        labels.append(np.pad(label, (0,maxlen-len(label)), 'constant', constant_values=0))
    feats = torch.Tensor(np.array(feats))
    labels = torch.Tensor(np.array(labels)).type(torch.LongTensor)
    return feats, labels

def collate_train2(batch):
    maxlen = 0
    for pairs in batch:
        feat, _, label = pairs
        maxlen = max(maxlen, len(label))
    feats1 = []
    feats2 = []
    labels = []
    for pairs in batch:
        feat1, feat2, label = pairs
        feats1.append(np.pad(feat1, ((0,maxlen-feat1.shape[0]), (0, 0)), 'constant', constant_values=0))
        feats2.append(np.pad(feat2, ((0,maxlen-feat2.shape[0]), (0, 0)), 'constant', constant_values=0))
        labels.append(np.pad(label, (0,maxlen-len(label)), 'constant', constant_values=0))
    feats1 = torch.Tensor(np.array(feats1))
    feats2 = torch.Tensor(np.array(feats2))
    labels = torch.Tensor(np.array(labels)).type(torch.LongTensor)
    return feats1, feats2, labels
