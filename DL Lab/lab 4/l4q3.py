from itertools import batched

import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np

loss_list = []
torch.manual_seed(42)

X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
Y = torch.tensor([0,1,1,0], dtype=torch.float32)

#defining xor model
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        # self.w = torch.nn.Parameter(torch.rand([1]))
        # self.b = torch.nn.Parameter(torch.rand([1]))

        self.linear1 = nn.Linear(2,2,bias=True)
        self.activation1 = nn.Sigmoid()

        self.linear2 = nn.Linear(2,1, bias=True)
        self.activation2 = nn.ReLU()

    def forward(self, x):
        x=self.linear1(x) #pass the input through first layer
        x=self.activation1(x) #first activation (second/hidden layer)

        x= self.linear2(x) #pass input thru second layer
        x=self.activation2(x) #second activation (output layer)
        return x

class myDataset(Dataset):
    def __init__(self, X, Y):  # Fixed typo: __int__ to __init__
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_item = self.X[idx]
        Y_item = self.Y[idx]
        return X_item, Y_item

full_dataset = myDataset(X,Y)
batch_size = 1

train_data_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load the model to GPU
model = XORModel().to(device)
print(model)
#Add the criterion which is the MSELoss
loss_fn = torch.nn.MSELoss()
#Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


def train_one_epoch(epoch_index):
    total_loss = 0

    for i,data in enumerate(train_data_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_fn(outputs.flatten(), labels)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    return total_loss/(len(train_data_loader)*batch_size)


epochs = 100
for epoch in range(epochs):
    model.train(True)

    avg_loss = train_one_epoch(epoch)
    loss_list.append(avg_loss)

print('printing model paramaters: ')
for param in model.named_parameters():
    print(param)

import torch
import torch.nn.functional as F

# Sample input X (e.g., [1, 0])
X = torch.tensor([1.0, 1.0], dtype=torch.float32).unsqueeze(0)  # Add batch dimension

# Manually extract weights and biases after training
W1 = model.linear1.weight
b1 = model.linear1.bias
W2 = model.linear2.weight
b2 = model.linear2.bias

# Perform the linear transformation for layer 1
output_1 = torch.matmul(X, W1.T) + b1
print("Output after Linear1:", output_1)

# Apply Sigmoid activation (for hidden layer)
output_1 = torch.sigmoid(output_1)
print("Output after Sigmoid activation (Linear1):", output_1)

# Perform the linear transformation for layer 2
output_2 = torch.matmul(output_1, W2.T) + b2
print("Output after Linear2:", output_2)

# Apply ReLU activation (for output layer)
output_2 = F.relu(output_2)
print("Output after ReLU activation (Linear2):", output_2)

