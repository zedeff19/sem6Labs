import matplotlib.pyplot as plt
import torch

x = torch.tensor( [2,4])

y = torch.tensor( [20,40])

b = torch.rand([1], requires_grad=True)
w = torch.rand([1], requires_grad=True)
print("the params are: {} and {} ". format(w,b))

alpha = torch.tensor(0.005)

loss_list = []

for epochs in range(110):
    loss = 0.0

    for j in range(len(x)):
        # a = w*x[j]
        y_p = (w*x[j]) + b # y_p = wx + b
        loss += (y_p - y[j]) **2

    loss = loss / len(x)
    loss_list.append(loss.item())

    loss.backward()

    with torch.no_grad():
        w -= alpha * w.grad
        b -= alpha * b.grad

    w.grad.zero_()
    b.grad.zero_()

    print("updated params: {}, {}. loss = {}".format(w,b,loss.item()))

plt.plot(loss_list)
plt.show()
