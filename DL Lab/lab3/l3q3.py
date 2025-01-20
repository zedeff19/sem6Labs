import matplotlib.pyplot as plt
import torch

x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0])
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0])

class RegressionModel:
    def __init__(self):
        self.w = torch.rand([1], requires_grad=True)
        self.b = torch.rand([1], requires_grad=True)

    def forward(self, x):
        return (self.w*x) + self.b

    def update(self):
        alpha = 0.005
        self.w -= alpha * self.w.grad
        self.b -= alpha * self.b.grad

    def reset_grad(self):
        self.w.grad.zero_()
        self.b.grad.zero_()

def criterion(y, y_p):
    return (y-y_p)**2

model = RegressionModel()
losses=[]

for epochs in range(500):
    loss = 0.0

    for j in range(len(x)):
        y_pred = model.forward(x[j])
        loss+=criterion(y[j],y_pred)

    loss = loss/len(x)
    losses.append(loss.item())

    loss.backward()

    with torch.no_grad():
        model.update()

    model.reset_grad()

    print("updated params: {}, {}. loss = {}".format(model.w, model.b, loss.item()))

plt.plot(losses)
plt.show()
