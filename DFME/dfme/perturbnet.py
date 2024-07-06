import sys
import os
import numpy as np
import os
import shutil
import tempfile
from PIL import Image
import random
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms as trn
from torchvision import datasets
import torchvision.transforms.functional as trnF 
from torch.nn.functional import gelu, conv2d
import torch.nn.functional as F
import random
import torch.utils.data as data
from torchvision.datasets import ImageFolder
from tqdm import tqdm


class GELU(torch.nn.Module):
    def forward(self, x):
        return F.gelu(x)

class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, hidden_planes=9, scale = 1, batch_size=5):
        """ Constructor
        Args:
            inplanes: input channel dimensionality (multiply by batch_size)
            planes: output channel dimensionality (multiply by batch_size)
            stride: conv stride. Replaces pooling layer.
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = hidden_planes * batch_size
        self.conv1 = nn.Conv2d(inplanes * batch_size, width*scale, kernel_size=1, bias=False, groups=batch_size)
        self.bn1 = nn.BatchNorm2d(width*scale)
        
        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale -1
        
        convs = []
        bns = []
        for i in range(self.nums):
            # K = random.choice([1, 3])
            # D = random.choice([1, 2, 3])
            K = 3
            D = 2
            P = int(((K - 1) / 2) * D)

            convs.append(nn.Conv2d(width, width, kernel_size=K, stride = stride, padding=P, dilation=D, bias=True, groups=batch_size))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes * batch_size, kernel_size=1, bias=False, groups=batch_size)
        self.bn3 = nn.BatchNorm2d(planes * batch_size)

        self.act = nn.ReLU(inplace=True)
        self.scale = scale
        self.width  = width
        self.hidden_planes = hidden_planes
        self.batch_size = batch_size

    def forward(self, x):
        _, _, H, W = x.shape
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out) # [1, hidden_planes*batch_size*scale, H, W]
        
        # Hack to make different scales work with the hacky batches
        out = out.view(1, self.batch_size, self.scale, self.hidden_planes, H, W)
        out = torch.transpose(out, 1, 2)
        out = torch.flatten(out, start_dim=1, end_dim=3)
        
        spx = torch.split(out, self.width, 1) # [ ... (1, hidden_planes*batch_size, H, W) ... ]
        
        for i in range(self.nums):
            if i==0:
                sp = spx[i]
            else:
                sp = sp + spx[i]

            sp = self.convs[i](sp)
            sp = self.act(self.bns[i](sp))
          
            if i==0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        
        if self.scale != 1:
            out = torch.cat((out, spx[self.nums]),1)
        
        # Undo hack to make different scales work with the hacky batches
        out = out.view(1, self.scale, self.batch_size, self.hidden_planes, H, W)
        out = torch.transpose(out, 1, 2)
        out = torch.flatten(out, start_dim=1, end_dim=3)

        out = self.conv3(out)
        out = self.bn3(out)

        return out

class Res2Net(torch.nn.Module):
    def __init__(self, epsilon=0.2, hidden_planes=16, batch_size=5):
        super(Res2Net, self).__init__()
        
        self.epsilon = epsilon
        self.hidden_planes = hidden_planes
                
        self.block1 = Bottle2neck(3, 3, hidden_planes=hidden_planes, batch_size=batch_size)
        self.block2 = Bottle2neck(3, 3, hidden_planes=hidden_planes, batch_size=batch_size)
        # self.block3 = Bottle2neck(3, 3, hidden_planes=hidden_planes, batch_size=batch_size)
        # self.block4 = Bottle2neck(3, 3, hidden_planes=hidden_planes, batch_size=batch_size)

    def reload_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d):
                layer.reset_parameters()
 
    def set_epsilon(self, new_eps):
        self.epsilon = new_eps

    def forward_original(self, x):                
        x = (self.block1(x) * self.epsilon) 
        x = (self.block2(x) * self.epsilon) 

        return x

    def forward(self, x):
        return self.forward_original(x)