import torch
import torch.nn.functional as F
import torch.nn as nn

filter = torch.tensor(
    [[0,-1,0],
     [-1,5,-1],
     [0,-1,0]], dtype=torch.float32
)

filter = filter.reshape(1,1,3,3)
inputs = torch.tensor(
    [[1,2,3,4],
     [5,6,7,8],
     [9,10,11,12],
     [13,14,15,16]], dtype=torch.float32
)

inputs = inputs.reshape(1,1,4,4)
print("filter =", filter)
print("inputs = ", inputs)

o1 = F.conv2d(inputs, filter, padding=0, bias=None)
print("F.conv2d output: ", o1)

conv = nn.Conv2d(1,1,kernel_size=3,padding=0,
bias=False)
conv.weight = nn.Parameter(filter)
o2=conv(inputs)
print("nn.Conv2d output=", o2)



'''
conv2d from nn.functional AND Conv2d from nn
import torch.nn.functional as F
F.conv2d(image, kernel, stride=1, padding=0)
import torch.nn as nn
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=True)
'''