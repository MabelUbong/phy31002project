# import standard PyTorch modules 
import torch
import torch.nn as nn
import torch.nn.functional as f
from collections import OrderedDict 

class Network(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, mid_ch, out_ch, kernel_size):
        super(Network, self).__init__()

        self.conv_layers = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=in_ch, out_channels=mid_ch, kernel_size=kernel_size, padding=0)),
            ('norm1', nn.BatchNorm2d(mid_ch)),
            ('relu1', nn.ReLU(mid_ch)),
            ('conv2', nn.Conv2d(in_channels=mid_ch, out_channels=out_ch, kernel_size=kernel_size, padding=0)),
            ('norm2', nn.BatchNorm2d(out_ch)),
            ('relu2', nn.ReLU(out_ch))

        ]))

    def forward(self, x):
      """
      Function to add each convolutional layers
      """
        return self.conv_layers(x)


class Down(nn.Module):
  """
  Down Convolution Block
  """

    def __init__(self, in_ch, mid_ch, out_ch, kernel_size):
        super().__init__()
        self.maxpool_conv = nn.Sequential(OrderedDict([
            ('Maxpooling', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('Conv_layers', Network(in_ch, mid_ch, out_ch, kernel_size))

        ]))

    def forward(self, x):
        return self.maxpool_conv(x)


class Final_Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(size=(1200, 1200), mode='bilinear', align_corners=True)

    def forward(self, t):
        return self.up(t)


class Up(nn.Module):
  """
  Up Convolution Block
  """

    def __init__(self, in_ch, mid_ch, out_ch, kernel_size, bilinear=True):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = Network(in_ch, mid_ch, out_ch, kernel_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        pdX = x2.size()[2] - x1.size()[2]
        pdY = x2.size()[3] - x1.size()[3]

        x2 = f.pad(x2, [-pdX, 0,
                        -pdY, 0])

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class Output(nn.Module):
  ''' 
  Output block
  '''

    def __init__(self, in_ch, out_ch):
        super(Output, self).__init__()

        self.end = nn.Sequential(OrderedDict([
            ('end', nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0))

        ]))

    def forward(self, x):
        return self.end(x)

# Unet Architecture 
class unet(nn.Module):
  """
   Implementation of U-Net

   Args:
   --------
   n_channels (int): number of input channels
   n_classes (int): number of output channels
   kernal_size (int): size of kernal

  """
    def __init__(self, n_channels, n_classes, kernel_size, bilinear=True):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.kernel_size = kernel_size

        self.start = Network(n_channels, 64, 64, kernel_size)
        self.down1 = Down(64, 128, 128, kernel_size)
        self.down2 = Down(128, 256, 256, kernel_size)
        self.down3 = Down(256, 512, 512, kernel_size)
        self.down4 = Down(512, 1024, 1024, kernel_size)
        self.up1 = Up(1024, 512, 512, kernel_size)
        self.up2 = Up(512, 256, 256, kernel_size)
        self.up3 = Up(256, 128, 128, kernel_size)
        self.up4 = Up(128, 64, 64, kernel_size)
        self.out = Output(64, n_classes)
        self.Final = Final_Up(n_classes, n_classes)

    def forward(self, x):
      '''
      Forward Pass
      '''
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
        x11 = self.Final(x10)

        return x11
