import torch, os

def save_ckpt(path, model, lr, optimizer, epoch, scheduler):
    torch.save({
        'model':model.state_dict(),
        'lr': lr,
        'optimizer': optimizer,
        'scheduler': scheduler
    },path+str(epoch)+'.pkl')


class AvgMeter(object):
    def __init__(self):
        self.renew()

    def renew(self):
        self.val = 0
        self.count = 0
        self.avg = 0
        self.sum = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    batch_size = target.size(0)

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_1 = correct[:1].view(-1).float().sum(0, keepdim=True)
    res=correct_1.mul_(100.0 / batch_size)
    return res

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']