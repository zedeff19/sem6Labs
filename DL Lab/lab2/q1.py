import torch

a = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(4.0, requires_grad=False)

x = 2*a  + 3*b
y = 5*a*a + 3*b*b*b
z = 2*x + 3*y

z.backward()
print("gradient at a = 3.0: ", a.grad)



