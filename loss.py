import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import normal

def get_loss(args, cls_num_list, per_cls_weights):
    # Default Linear
    if args.loss_type == 'CE':
        criterion = CELoss(weight=per_cls_weights).cuda(args.gpu) #nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
    elif args.loss_type == 'Focal':
        criterion = FocalLoss(weight=per_cls_weights, gamma=1).cuda(args.gpu)
    elif args.loss_type == 'FeaBal':
        criterion = FeaBalLoss(cls_num_list=cls_num_list, weight=per_cls_weights, lambda_=args.lambda_).cuda(args.gpu) # hyper-parameter A=60                 
    elif args.loss_type == 'LDAM':
        criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(args.gpu) 

    else:
        raise NotImplementedError("Error:Loss function {} is not implemented! Please re-choose loss type!".format(args.loss_type))

    return criterion

class CELoss(nn.Module):
    def __init__(self, weight):
        super(CELoss, self).__init__()
        self.weight = weight
        
    def forward(self, out, labels, curr=0):
        """
        Args:
            out: dict out['feat'], embedding; out['score'], logit    
            labels: ground truth labels with shape (batch_size).
        """
        feat, out = out['feature'], out['score']  
        return F.cross_entropy(out, labels, weight=self.weight)



def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)   # transfer to probability
    loss = (1 - p.detach()) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target, curr=None):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=False, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 /np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target, curr=None):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)                             #one-hot
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))  #取得对应位置的m   self.m_list
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)                                       #x的index位置换成x_m
        
        return F.cross_entropy(self.s*output, target, weight=self.weight)  #weight=self.weight
   
class FeaBalLoss(nn.Module):
    def __init__(self, cls_num_list, weight, lambda_=1., classifier = False, gamma=0.):
        super(FeaBalLoss, self).__init__()
        self.num_classes = len(cls_num_list)
        self.weight = weight
        self.classisier = classifier
        self.lambda_ = lambda_
        
        lam_list = torch.cuda.FloatTensor(cls_num_list)
        lam_list = torch.log(lam_list)  #s_list = s_list**(1/4)
        lam_list = lam_list.max()-lam_list        
        self.lam_list = lam_list*(1/lam_list.max()) #归一化 lambda_：限制强度
        
        self.gamma = gamma
        
    def forward(self, out, labels, curr=0):
        """
        Args:
            out: dict out['feat'], embedding; out['score'], logit    
            labels: ground truth labels with shape (batch_size).
        """
        feat, out = out['feature'], out['score']  
        feat_norm = torch.norm(feat,dim=1).unsqueeze(1).repeat([1,len(self.lam_list)])
        
        logit = out - curr*self.lambda_*self.lam_list/(feat_norm+1e-12)
        
        if self.classisier:#classifier re-balance model
            return F.cross_entropy(out, labels, weight=self.weight)       
        else:  
            return F.cross_entropy(logit, labels, weight=self.weight)