import torch
import torch.nn as nn
import numpy as np

# Create Fibonacci sequence
fibo_list = []

def fibo(n):
    if n == 0 or n == 1:
        return n
    else:
        return fibo(n - 1) + fibo(n - 2)

for i in range(10):
    fibo_list.append(fibo(i))

print("Fibonacci Sequence:", fibo_list)

# Prepare input data (groups of 3 consecutive Fibonacci values)
input_data = []

for i in range(len(fibo_list) - 3):
    input_data.append(fibo_list[i:i + 3])  # Grouping consecutive 3 values

input_data = np.array(input_data)  # Shape: (7, 3), 7 sequences of 3 values
target_data = fibo_list[3:]  # Target values will be the next Fibonacci value for each group

# Convert to torch tensors
X = torch.tensor(input_data, dtype=torch.float32)  # Shape: (7, 3)
y = torch.tensor(target_data, dtype=torch.float32).view(-1, 1)  # Shape: (7, 1), we want a single output per sequence

# Reshape X to have an additional dimension for the sequence length (seq_length=3)
X = X.unsqueeze(2)  # Shape: (7, 3, 1), now seq_length=3 and input_size=1

# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        # Input size is 1 (group of 1 Fibonacci number per time step), hidden size is 5 (size of the RNN's hidden state)
        self.rnn = nn.RNN(input_size=1, hidden_size=5, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(in_features=5, out_features=1)  # Output size is 1 (single predicted value)

    def forward(self, x):
        # Pass through RNN
        output, _status = self.rnn(x)  # _status is not used here
        output = output[:, -1, :]  # Only take the last hidden state (after the entire sequence)
        output = self.fc1(torch.relu(output))  # Apply ReLU and then linear layer
        return output

# Create and train the model
model = RNNModel()

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop (Increase epochs to 1000 for better learning)
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    output = model(X)

    # Compute the loss
    loss = criterion(output, y)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}')

# Test the model on the same data
model.eval()
with torch.no_grad():
    predicted = model(X)
    print("\nPredicted Values:", predicted.squeeze().numpy())  # Remove extra dimensions for display
    print("Actual Values:", y.numpy())
