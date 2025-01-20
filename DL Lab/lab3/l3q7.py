import matplotlib.pyplot as plt
import torch

# Sample data
x = torch.tensor([1, 5, 10, 10, 25, 50, 70, 75, 100], dtype=torch.float32)
y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32)


# Logistic Regression Model
class LogisticRegressionModel:
    def __init__(self):
        self.w = torch.rand([1], requires_grad=True)  # weight initialized randomly
        self.b = torch.rand([1], requires_grad=True)  # bias initialized randomly

    def forward(self, x):
        # Sigmoid activation function for logistic regression
        z = self.w * x + self.b
        return torch.sigmoid(z)

    def update(self):
        # Gradient descent update with learning rate
        alpha = 0.005
        self.w -= alpha * self.w.grad
        self.b -= alpha * self.b.grad

    def reset_grad(self):
        # Zero the gradients
        self.w.grad.zero_()
        self.b.grad.zero_()


# Binary Cross-Entropy Loss (Log Loss)
def criterion(y, y_pred):
    epsilon = 1e-15  # To avoid log(0) errors
    y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon)  # Clamping to prevent log(0)
    return -y * torch.log(y_pred) - (1 - y) * torch.log(1 - y_pred)


# Initialize the model
model = LogisticRegressionModel()
losses = []

# Training loop (500 epochs)
for epoch in range(500):
    loss = 0.0

    for i in range(len(x)):
        # Forward pass: Compute predicted y using the model
        y_pred = model.forward(x[i])

        # Compute the loss
        loss += criterion(y[i], y_pred)

    # Average loss
    loss = loss / len(x)
    losses.append(loss.item())

    # Backward pass: Compute gradients of loss w.r.t. parameters
    loss.backward()

    with torch.no_grad():
        model.update()  # Update model parameters using gradients

    model.reset_grad()  # Zero the gradients

    if epoch % 50 == 0:
        print(
            f"Epoch {epoch}, Updated params: w = {model.w.item():.4f}, b = {model.b.item():.4f}, Loss = {loss.item():.4f}")

# Plot the loss over epochs
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Logistic Regression - Loss over Epochs')
plt.show()
