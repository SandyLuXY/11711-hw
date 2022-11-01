import torch
from torch.utils.data import DataLoader
from dataset.dataset import  collate_train, TXTDataset, TXTDataset2, collate_train2
from modules.CLF import CLF, CLF_TDNN
import torch.nn as nn, torch.nn.functional as F
import numpy as np
from utils.utils import save_ckpt, accuracy, AvgMeter, get_lr
import random
import os
import time
import sys
from sklearn.metrics import f1_score

# front end model name
if sys.argv[1] == 'sci':
    model_name = 'scibert_uncase'
else:
    model_name = 'nerbert'


path = 'data/train_10_l8_non0_'+model_name+'.npy'

testpath = 'data/test_2_'+model_name+'.npy'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


out_feat = 8
in_feat = 768
epochs = 160
start_epoch = 0
dropout_rate = 0.1

SEED = 1048596 # steins;gate no sentaku
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
lr = 1e-3
weight_decay = 0
warm_up = 0
batch_size = 1

criterion = nn.CrossEntropyLoss().cuda()


train_set = TXTDataset(path)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn = collate_train)
test_set = TXTDataset(testpath)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn = collate_train)

if sys.argv[2] == 'TDNN':
    model = CLF_TDNN(in_features=in_feat, out_features = out_feat, dropout_rate=dropout_rate).cuda()
    model_name = model_name +'_TDNN'
else:
    model = CLF(in_features=in_feat, out_features = out_feat, dropout_rate=dropout_rate).cuda()
    model_name = model_name +'_Linear'

ckpt_path = 'checkpoints/'+model_name+'/model'
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr,weight_decay=weight_decay)

lambda1 = lambda epoch: 0.98 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
model.train()
loss_train = []
loss_test = []
for epoch in range(start_epoch, epochs):
    loss_meter, acc_meter = AvgMeter(), AvgMeter()
    pred_save = []
    labels_save = []
    pred_save_all= []
    model.train()
    for i, (feats, labels) in enumerate(train_loader):
        feats, labels = feats.cuda(), labels.cuda()
        out = model(feats)
        # print(feats.size(), out.size())
        loss = criterion(out[0], labels[0])
        for uttid in range(1, len(labels)):
            loss += criterion(out[uttid], labels[uttid])
        loss /= len(labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for uttid in range(0, len(labels)):
            pred_save.append( list(torch.argmax(out[uttid],dim = 1).cpu().numpy()))
            labels_save += list(labels[uttid].cpu().numpy())

        for uttid in range(len(labels)):
            acc = accuracy(out[uttid].data,labels[uttid].data)
            acc_meter.update(acc.data.item(), feats.size(1))

        loss_meter.update(loss.data.item(), feats.size(0))
    lbs_tmp = np.array(labels_save)
    indx = (labels_save!= 0)
    pred_tmp = np.array(pred_save)[indx]
    lbs_tmp = lbs_tmp[indx]
    loss_train.append(loss_meter.avg)


    # print(
    #         'Epoch [%d][%d/%d]\t ' % (epoch, i, len(train_loader)) +
    #         'Loss %.4f %.4f\t' % (loss_meter.val, loss_meter.avg) +
    #         'F1 %3.3f\t' % (f1_score(labels_save,pred_save, average='micro')) +
    #         'LR %.6f' % get_lr(optimizer)
    #         )

    if epoch >= warm_up:
        scheduler.step()
    save_ckpt(ckpt_path, model, lr, optimizer,epoch,None)
    np.save('outputs/'+model_name+'/predict_'+str(epoch), pred_save, allow_pickle=True)
    np.save('outputs/'+model_name+'/labels_'+str(epoch), labels_save, allow_pickle=True)

    model.eval()
    #  testing
    loss_meter, acc_meter = AvgMeter(), AvgMeter()
    pred_save = []
    labels_save = []
    for i, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.cuda(), labels.cuda()
        out = model(feats)
        out = out.view(out.size(1),-1)
        pred_save.append(list(torch.argmax(out,dim = 1).cpu().numpy()))
        pred_save_all += list(torch.argmax(out,dim = 1).cpu().numpy())
        loss = criterion(out,labels[0])
        loss_meter.update(loss.data.item(), len(labels[0]))

        labels_save += list(labels[0].cpu().numpy())
        acc = accuracy(out.data,labels[0].data)
        acc_meter.update(acc.data.item(), feats.size(1))
        # print(feats.size(0))
    print(
        'Epoch [%d][%d/%d]\t ' % (epoch, len(train_loader), len(train_loader)) +
        'F1 %3.3f\t' % (f1_score(labels_save,pred_save_all, average = 'micro'))
    )
    loss_test.append(loss_meter.avg)
    np.save('outputs/'+model_name+'/test_predict_'+str(epoch), pred_save, allow_pickle=True)
    np.save('outputs/'+model_name+'/test_labels_'+str(epoch), labels_save, allow_pickle=True)
    np.save('outputs/'+model_name+'/test_labels_all_'+str(epoch), pred_save_all, allow_pickle=True)

np.save('outputs/'+model_name+'/train_loss', np.array(loss_train), allow_pickle=True)
np.save('outputs/'+model_name+'/test_loss', np.array(loss_test), allow_pickle=True)
