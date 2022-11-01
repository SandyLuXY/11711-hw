import math, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn import Parameter

# acc 49 lr 1e-4 randseet 41
class CLF_TDNN(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate = 0.3):
        super(CLF_TDNN, self).__init__()
        self.in_feat = in_features
        self.out_feat = out_features
        self.conv1_out = 256
        self.conv2_out = 256
        self.conv3_out = 256
        self.ln1_out = 128
        self.conv1 = nn.Conv1d(in_channels= in_features, out_channels= self.conv1_out, kernel_size= 3, padding=1)
        self.conv3 = nn.Conv1d(in_channels= self.conv3_out, out_channels= self.conv3_out, kernel_size= 3, padding=1)
        self.nonLinear = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(self.conv1_out)
        self.bn3 = nn.BatchNorm1d(self.conv3_out)
        self.clf = torch.nn.Linear(self.conv3_out, self.ln1_out)
        self.clf2 = torch.nn.Linear(self.ln1_out,out_features)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        sizes = x.size(1)
        x = x.permute(0,2,1)
        x = self.conv1(x)
        x = self.nonLinear(x)
        x = self.bn1(x)
        x = self.conv3(x)
        x = self.nonLinear(x)
        x = self.bn3(x)
        x = x.permute(0,2,1)
        x = self.dropout(x)
        x = self.clf(x)
        x = self.clf2(self.nonLinear(x))

        # print(x.size())
        return F.softmax(x, dim = 2)

class CLF(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate = 0.3):
        super(CLF, self).__init__()
        self.in_feat = in_features
        self.out_feat = out_features
        self.cls_hidden_size = 256
        self.clf = nn.Sequential(
            nn.Linear(self.in_feat, self.cls_hidden_size),
            nn.ReLU(),
            nn.Linear(self.cls_hidden_size, self.out_feat)
        )
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(x)
        return F.softmax(self.clf(x),dim = -1)
