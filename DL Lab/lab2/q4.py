import torch

x = torch.tensor(3.0, requires_grad=True)

a = torch.pow(x, 2)*-1
b = -2 * x
c = -1 * torch.sin(x)
z = a+b+c

z.backward()

print("value of gradient at x = 3.0: ", x.grad)