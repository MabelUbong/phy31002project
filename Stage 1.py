import torch
import torch.nn as nn

network1 = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3), 
      nn.ReLU(),
      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3), 
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2)
      
      )


#Example random tensor used to see if the output from 
#this specific module works

Input = torch.rand([572,572]).unsqueeze(0).unsqueeze(0)

print(Input.shape)

    
Output = network1(Input)

print(Output.shape)

