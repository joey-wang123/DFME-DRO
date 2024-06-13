import torch.nn as nn
import math
import torch.nn.functional as F
import torch
import numpy as np
# from network.lenet import Encoder

class noise_Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, pni='layerwise', w_noise=True):
        super(noise_Linear, self).__init__(in_features, out_features, bias)
        new = 1.0
        self.pni = pni
        if self.pni == 'layerwise':
            self.alpha_w = nn.Parameter(torch.Tensor([new]), requires_grad = True)
        elif self.pni == 'channelwise':
            self.alpha_w = nn.Parameter(torch.ones(self.out_features).view(-1,1)*new,
                                        requires_grad=True)
        elif self.pni == 'elementwise':
            self.alpha_w = nn.Parameter(torch.ones(self.weight.size())*new, requires_grad = True)
        
        self.w_noise = w_noise

    def forward(self, input):
        scale = 1.0
        with torch.no_grad():
            std = self.weight.std().item()
            noise = self.weight.clone().normal_(0,std)*scale

        noise_weight = self.weight + self.alpha_w * noise * self.w_noise
        output = F.linear(input, noise_weight, self.bias)
        
        return output 



class data_dependent_Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, w_noise=True):
        super(data_dependent_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                         padding, dilation, groups, bias)

        new = 1.0
        self.alpha_w = nn.Parameter(torch.Tensor([new]), requires_grad = True)        
        self.w_noise = w_noise   


    def forward(self, input):

        scale = 1.0
        with torch.no_grad():
            std = self.weight.std().item()
            noise = self.weight.clone().normal_(0,std)*scale

        noise_weight = self.weight + self.alpha_w * noise * self.w_noise
        output = F.conv2d(input, noise_weight, self.bias, self.stride, self.padding, self.dilation,
                        self.groups)

        return output


class noise_Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, w_noise=True):
        super(noise_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                         padding, dilation, groups, bias)

        new = 1.0
        self.alpha_w = nn.Parameter(torch.Tensor([new]), requires_grad = True)        
        self.w_noise = w_noise   


    def forward(self, input):

        scale = 1.0
        with torch.no_grad():
            std = self.weight.std().item()
            noise = self.weight.clone().normal_(0,std)*scale

        # noise_weight = self.weight + self.alpha_w * noise * self.w_noise
        # print('determinnistic')
        noise_weight = self.weight 
        output = F.conv2d(input, noise_weight, self.bias, self.stride, self.padding, self.dilation,
                        self.groups)

        return output
                      
                      