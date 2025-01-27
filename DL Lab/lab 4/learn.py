import torch
from torch import nn
torch.manual_seed(42)
## Creating an object for the linear class
linear = nn.Linear(in_features=3, out_features=1)
print('network structure : ',linear)
print('Weight of network :\n',linear.weight)
print('Bias of network :\n',linear.bias)
## Passing input to the linear layer
output = linear(torch.tensor([1,2,3],
dtype=torch.float32))
print(output)