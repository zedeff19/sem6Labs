import torch

w = torch.tensor(3.0, requires_grad=True)
x = torch.tensor(4.0, requires_grad=False)
b = torch.tensor(5.0, requires_grad=False)

# Use PyTorch's built-in ReLU function instead of a custom function
u = w * x
v = u + b

# Apply the ReLU function using PyTorch's built-in method
a = torch.relu(v)

# Compute gradients using the backward pass
a.backward()

print("value of gradient at w = 3.0: ", w.grad)
