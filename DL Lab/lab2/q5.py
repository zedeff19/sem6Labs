import torch

x = torch.tensor(2.0, requires_grad=True)

a = torch.pow(x, 4) * 8
b = torch.pow(x, 3) * 3
c = torch.pow(x, 2) * 7
d = 6*x
e = 3

y = a+b+c+d+e

y.backward()
print(x.grad)