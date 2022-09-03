import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter
from torch.distributions import normal

__all__ = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        cosine = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        #out = x.mm(self.weight)
        return cosine

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)      
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, classifier=True, linear_type='Default'):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.classifier = classifier
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
    
        if self.classifier:
            if linear_type == 'Norm':
                self.linear = NormedLinear(64, num_classes)
            elif linear_type == 'Default':
                self.linear = nn.Linear(64, num_classes)
            else:
                raise NotImplementedError("Error:Linear {} is not implemented! Please re-choose linear type!".format(linear_type))
                
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, get_feat=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        feat = F.avg_pool2d(out, out.size()[3])
        #feat=F.dropout2d(feat, p=self.dropout_rate)  #0831 加了dropput,
        feat = feat.view(feat.size(0), -1)
        
        if self.classifier:
            score = self.linear(feat)
            if get_feat == True:
                out = dict()  
                out['feature'] = feat
                out['score'] = score
            else:
                out = score
            return out
        else:
            return feat
            
def resnet20(num_classes=10, classifier=True, linear_type='Default', pretrained=False):
    if pretrained:
        print("Sorry! In our implementation, the series of CIFAR doesn't support loading pre-trained models!")
    
    return ResNet(BasicBlock, [3, 3, 3], num_classes, classifier, linear_type)

def resnet32(num_classes=10, classifier=True, linear_type='Default', pretrained=False):
    if pretrained:
        print("Sorry! In our implementation, the series of CIFAR doesn't support loading pre-trained models!")

    return ResNet(BasicBlock, [5, 5, 5], num_classes, classifier, linear_type)

def resnet44(num_classes=10, classifier=True, linear_type='Default', pretrained=False):
    if pretrained:
        print("Sorry! In our implementation, the series of CIFAR doesn't support loading pre-trained models!")
    
    return ResNet(BasicBlock, [7, 7, 7], num_classes, classifier, linear_type)

def resnet56(num_classes=10, classifier=True, linear_type='Default', pretrained=False):
    if pretrained:
        print("Sorry! In our implementation, the series of CIFAR doesn't support loading pre-trained models!")

    return ResNet(BasicBlock, [9, 9, 9], num_classes, classifier, linear_type)

def resnet110(num_classes=10, classifier=True, linear_type='Default', pretrained=False):
    if pretrained:
        print("Sorry! In our implementation, the series of CIFAR doesn't support loading pre-trained models!")

    return ResNet(BasicBlock, [18, 18, 18], num_classes, classifier, linear_type)

def resnet1202(num_classes=10, classifier=True, linear_type='Default', pretrained=False):
    if pretrained:
        print("Sorry! In our implementation, the series of CIFAR doesn't support loading pre-trained models!")

    return ResNet(BasicBlock, [200, 200, 200], num_classes, classifier, linear_type)

def net_test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            net_test(globals()[net_name]())
            print()
