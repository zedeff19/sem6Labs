import torch
x = torch.tensor(3.0, requires_grad=True)

a = -x
b = torch.exp(a)
c = 1 + b
s = 1/c

s.backward()

print("value of gradient at x = 3.0: ", x.grad)