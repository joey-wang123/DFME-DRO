# This part is borrowed from https://github.com/huawei-noah/Data-Efficient-Model-Compression

import torch
import torch.nn as nn
import torch.nn.functional as F
from cifar10_models.noise_layer import noise_Conv2d
import random
from perturbnet import *

class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = noise_Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = noise_Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                noise_Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
 
 
class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = noise_Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = noise_Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = noise_Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                noise_Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
 
 
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, args, num_classes=10, normalize_coefs=None, normalize=False):
        super(ResNet, self).__init__()

        if normalize_coefs is not None:
            self.mean, self.std = normalize_coefs

        self.normalize = normalize

        self.in_planes = 64
 
        perturbnet_batch_size = int(args.batch_size)
        self.perturbnet = Res2Net(epsilon=0.50, hidden_planes=2, batch_size=perturbnet_batch_size).train().cuda()
        self.conv1 = noise_Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        #self.conv1 = data_dependent_Conv2d(3, 64, kernel_size=3, input_size=3072, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, noise_Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.args = args



    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
 
    def forward(self, x, out_feature=False):

        noisenet_max_eps = 0.6
        self.perturbnet.reload_parameters()
        self.perturbnet.set_epsilon(random.uniform(noisenet_max_eps / 2.0, noisenet_max_eps))

        new_fake = x.view(1, -1, 32, 32)
        fake_noise = self.perturbnet(new_fake)
        fake_noise = fake_noise.view(self.args.batch_size, 3, 32, 32)
        x = x + self.args.scale*fake_noise
        # output = net(images + scale*fake_noise)
        # print('self.args.scale', self.args.scale)

        if self.normalize:
            # Normalize according to the training data normalization statistics
            x -= self.mean
            x /= self.std

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)
        if out_feature == False:
            return out
        else:
            return out,feature
 
 
def ResNet18_8x(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)
 
def ResNet34_8x(args, num_classes=10, normalize_coefs=None, normalize=False):
    print('loading ResNet34_8x')
    return ResNet(BasicBlock, [3,4,6,3], args, num_classes,  normalize_coefs=normalize_coefs, normalize=normalize)
 
def ResNet50_8x(num_classes=10):
    return ResNet(Bottleneck, [3,4,6,3], num_classes)
 
def ResNet101_8x(num_classes=10):
    return ResNet(Bottleneck, [3,4,23,3], num_classes)
 
def ResNet152_8x(num_classes=10):
    return ResNet(Bottleneck, [3,8,36,3], num_classes)
 
