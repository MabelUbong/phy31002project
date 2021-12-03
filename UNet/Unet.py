# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 16:38:17 2021

@author: tim
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict



class Network(nn.Module):
    
    def __init__(self, in_ch, mid_ch, out_ch):
        super(Network, self).__init__()
        
        self.conv_layers = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=in_ch, out_channels=mid_ch, kernel_size=3, padding=0)),
            ('norm1', nn.BatchNorm2d(mid_ch)),
            ('relu1', nn.ReLU(mid_ch)),
            ('conv2', nn.Conv2d(in_channels=mid_ch, out_channels=out_ch, kernel_size=3, padding=0)),
            ('norm2', nn.BatchNorm2d(out_ch)),
            ('relu2', nn.ReLU(out_ch))
            
        ]))
        
    def forward(self, x):
        return self.conv_layers(x)
    


class Down(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(OrderedDict([
            ('Maxpooling', nn.MaxPool2d(kernel_size=2, stride=2)), 
            ('Conv_layers', Network(in_ch, mid_ch, out_ch))
            
        ]))

    def forward(self, x):
        return self.maxpool_conv(x)
    
    

class Up(nn.Module):
    
    def __init__(self, in_ch, mid_ch, out_ch, bilinear = True):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = Network(in_ch, mid_ch, out_ch)
        
    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        


        x2 = F.pad(x2, [-diffX // 2, -diffX + diffX // 2,
                        -diffY // 2, -diffY + diffY // 2])
        
       
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)
        
    
    
class Output(nn.Module):
    
    def __init__(self, in_ch, out_ch):
        super(Output, self).__init__()
        
        self.end = nn.Sequential(OrderedDict([
            ('end', nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0))
            
            
        ]))
        
    def forward(self, x):
        return self.end(x)



class unet(nn.Module):
    def __init__(self, bilinear = True):
        super().__init__()
        
        self.bilinear = bilinear

        self.start = Network(1, 64, 64)
        self.down1 = Down(64, 128, 128)
        self.down2 = Down(128, 256, 256)
        self.down3 = Down(256, 512, 512)
        self.down4 = Down(512, 1024, 1024)
        self.up1 = Up(1024, 512, 512)
        self.up2 = Up(512, 256, 256)
        self.up3 = Up(256, 128, 128)
        self.up4 = Up(128, 64, 64)
        self.out = Output(64, 2)
        
        
    def forward(self, x):
       x1 = self.start(x)
       x2 = self.down1(x1)
       x3 = self.down2(x2)
       x4 = self.down3(x3)
       x5 = self.down4(x4)
       x6 = self.up1(x5, x4)
       x7 = self.up2(x6, x3)
       x8 = self.up3(x7, x2)
       x9 = self.up4(x8, x1)
       x10 = self.out(x9)
       
      
       
       return x10

print('Cuda is available?' , torch.cuda.is_available())
    
Unet = unet().cuda()

Input = torch.rand([1,1,572,572], dtype=torch.float32).cuda()

print(Unet(Input).shape)

