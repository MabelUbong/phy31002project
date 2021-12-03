import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn  # Import the neural network function from pytorch
import torch.nn.functional as F
import torch.optim as optim


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
    def forward(self,t):
        t = t

        t = self.conv1(t)
        t = F.relu(t)

        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=1, stride=2)

        return(t)


T = torch.zeros([1,1,572,572], dtype=torch.float32)

print(T.shape)

network1 = Network()
print(network1(T).shape)




