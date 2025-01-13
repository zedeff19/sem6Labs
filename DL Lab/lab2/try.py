import torch
x = torch.tensor(3.0, requires_grad=True)
'''
key part is the argument requires_grad=True. This tells PyTorch that you want to 
track operations on x so you can compute gradients later, which is 
necessary for backpropagation in machine learning models.
'''


y = x*x
z = 2*y + 3

print("x: " , x)
print("y= x*x: ", y)
print("z = 2*y + 3: ",z)

z.backward()
print("gradient at x = 3.5: ", x.grad)

with torch.no_grad():
    dy_dx = 2*x
'''torch.no_grad(): Disables gradient tracking temporarily for operations where you don't 
need gradients, which can be helpful for saving memory or manually calculating gradients.'''


print("analytical gradient: ", dy_dx)
