import matplotlib.pyplot as plt
import torch

x1= torch.tensor([3.0, 4.0, 5.0, 6.0, 2.0])
x2= torch.tensor([8.0, 5.0, 7.0, 3.0, 1.0])
y = torch.tensor([-3.7, 3.5, 2.5, 11.5, 5.7])

class RegressionModel:
    def __init__(self):
        self.w1 = torch.rand([1], requires_grad=True)
        self.w2 = torch.rand([1], requires_grad=True)
        self.b = torch.rand([1], requires_grad=True)

    def forward(self, x1, x2):
        return (self.w1*x1) + (self.w2*x2) + self.b

    def update(self):
        alpha = 0.005
        self.w1 -= alpha * self.w1.grad
        self.w2 -= alpha * self.w2.grad
        self.b -= alpha * self.b.grad

    def reset_grad(self):
        self.w1.grad.zero_()
        self.w2.grad.zero_()
        self.b.grad.zero_()

def criterion(y, y_p):
    return (y-y_p)**2

model = RegressionModel()
losses=[]

for epochs in range(500):
    loss = 0.0

    for j in range(len(x1)):
        y_pred = model.forward(x1[j], x2[j])
        loss+=criterion(y[j],y_pred)

    loss = loss/len(x1)
    losses.append(loss.item())

    loss.backward()

    with torch.no_grad():
        model.update()

    model.reset_grad()

    print("updated params: {}, {}, {}. loss = {}".format(model.w1,model.w2, model.b, loss.item()))

print('final parameters: ', model.w1, model.w2, model.b)
print('for x1 = 3, x2 = 2: y = ', model.w1*3 + model.w2*2 + model.b)

plt.plot(losses)
plt.show()
