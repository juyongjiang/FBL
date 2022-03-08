import torch
import torch.nn as nn
import shutil
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import math
from pathlib import Path
import time


class LearnableWeightScaling(nn.Module):
    def __init__(self, num_classes):
        super(LearnableWeightScaling, self).__init__()
        self.learned_norm = nn.Parameter(torch.ones(1, num_classes))

    def forward(self, x):
        return self.learned_norm * x

class AverageMeter(object):
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def mixup_criterion(criterion, pred, y_a, y_b, lam, curr=0):
    return lam * criterion(pred, y_a, curr) + (1 - lam) * criterion(pred, y_b, curr)

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def get_category_list(annotations, num_classes):
    num_list = [0] * num_classes
    cat_list = []
    print("Weight List has been produced")
    for anno in annotations:
        category_id = anno["category_id"]
        num_list[category_id] += 1
        cat_list.append(category_id)
    return num_list, cat_list


def prepare_folders(args):
    time_str = time.strftime('%Y%m%d%H%M')
    store_name = '_'.join([args.dataset, args.loss_type, args.train_rule, str(args.imb_factor), 
                                args.exp_str, time_str])
    store_name = os.path.join('saved', store_name)
    
    folders_util = ['saved', store_name, 
                    os.path.join(store_name, 'log'),
                    os.path.join(store_name,'checkpoint')]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.makedirs(folder, exist_ok=True)
            
    return store_name

def save_checkpoint(store_name, state, is_best):   
    filename = '%s/checkpoint/ckpt.pth.tar' % (store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))

def set_color(log, color, highlight=True):
    color_set = ['black', 'red', 'green', 'yellow', 'blue', 'pink', 'cyan', 'white']
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = '\033['
    if highlight:
        prev_log += '1;3'
    else:
        prev_log += '0;3'
    prev_log += str(index) + 'm'
    return prev_log + log + '\033[0m'

def accuracy(output, target, topk=(1,)):
    
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def adjust_learning_rate(args, optimizer, epoch):
    epoch = epoch + 1
    
    if args.dataset.lower().startswith('cifar'):
        if epoch <= 5:
            lr = args.lr * epoch / 5
        elif epoch > 180:
            lr = args.lr * 0.0001
        elif epoch > 160:
            lr = args.lr * 0.01
        else:
            lr = args.lr
    elif args.dataset.lower() == 'place365':
        if epoch <= 5:
            lr = args.lr * epoch / 5
        elif epoch > 80:
            lr = args.lr * 0.01
        elif epoch > 60:
            lr = args.lr * 0.1
        else:
            lr = args.lr
    else:
        if epoch <= 5:
            lr = args.lr * epoch / 5
        elif epoch > 160:
            lr = args.lr * 0.01
        elif epoch > 120:
            lr = args.lr * 0.1
        else:
            lr = args.lr

    # update
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def calc_confusion_mat(args, val_loader, model, cls_num_list):
    
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    cf = confusion_matrix(all_targets, all_preds).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    cls_acc = cls_hit / cls_cnt

    print('Class Accuracy : ')
    print(cls_acc)
    classes = [str(x) for x in cls_num_list]
    plot_confusion_matrix(all_targets, all_preds, classes)
    plt.savefig(os.path.join(args.root_log, args.store_name, 'confusion_matrix.png'))

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax