import torch.nn as nn
import torch.nn.functional as F
import torch





class noise_Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, w_noise=True, new = 1.0):
        super(noise_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                         padding, dilation, groups, bias)

        self.new = new
        self.alpha_w = nn.Parameter(torch.Tensor([self.new]), requires_grad = True)          
        self.w_noise = w_noise    


    def forward(self, input):

        scale = 1.0
        with torch.no_grad():
            std = self.weight.std().item()
            noise = self.weight.clone().normal_(0,std)*scale

        # noise_weight = self.weight + self.alpha_w * noise * self.w_noise
        print('no random')
        noise_weight = self.weight
        output = F.conv2d(input, noise_weight, self.bias, self.stride, self.padding, self.dilation,
                        self.groups)

        return output
                      
                      