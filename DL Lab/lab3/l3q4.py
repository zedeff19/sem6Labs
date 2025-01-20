import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Define the Dataset class
class LinearDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

# Define the regression model by extending nn.Module
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.w = nn.Parameter(torch.rand(1))  # weight as a learnable parameter
        self.b = nn.Parameter(torch.rand(1))  # bias as a learnable parameter

    def forward(self, x):
        return self.w * x + self.b

# Data
x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0])
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0])

# Create Dataset and DataLoader
dataset = LinearDataset(x, y)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Instantiate model and define loss function and optimizer
model = RegressionModel()
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.SGD(model.parameters(), lr=0.005)  # Stochastic Gradient Descent optimizer

# Training loop
losses = []
for epoch in range(500):
    epoch_loss = 0.0

    for x_batch, y_batch in dataloader:
        # Forward pass
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        epoch_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Backpropagate the gradients
        optimizer.step()  # Update the parameters

    losses.append(epoch_loss / len(dataloader))

    if epoch % 50 == 0:  # Print the loss every 50 epochs
        print(f"Epoch {epoch}, Loss: {epoch_loss / len(dataloader):.4f}, Params: w={model.w.item():.4f}, b={model.b.item():.4f}")

# Plot the loss over epochs
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.show()

# Final parameters after training
print(f"Final parameters: w={model.w.item():.4f}, b={model.b.item():.4f}")
