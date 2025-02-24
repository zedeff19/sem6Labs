import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# Bernoulli Dropout implementation
class BernoulliDropout(nn.Module):
    def __init__(self, p: float):
        """
        :param p: The probability of keeping a neuron (1 - dropout rate)
        """
        super(BernoulliDropout, self).__init__()
        self.p = p  # Probability of keeping the neuron

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Bernoulli Dropout during training. No dropout is applied during inference.
        """
        if self.training:  # Only apply dropout during training
            # Generate a Bernoulli random variable mask with probability (1 - p) for keeping the neuron
            mask = torch.rand_like(x) > self.p  # Shape same as x, 0 or 1 based on p
            x = x * mask  # Apply the mask to the input (drop neurons by zeroing them)
            # Scale the output by (1 / (1 - p)) to maintain the expected value during training
            x = x / (1 - self.p)
        return x


# Data transformations
transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}

# Datasets and data loaders
train = datasets.ImageFolder('./dog-cat-full-dataset-master/data/train', transform=transform['train'])
test = datasets.ImageFolder('./dog-cat-full-dataset-master/data/test', transform=transform['val'])
train_loader = DataLoader(train, batch_size=128, shuffle=True)
test_loader = DataLoader(test, batch_size=128, shuffle=False)

# Load pre-trained AlexNet
model = models.alexnet(weights='IMAGENET1K_V1')

# Freeze all layers except the final classifier
for param in model.parameters():
    param.requires_grad = False

# Replace the final classifier layer with a new one for binary classification (dog vs cat)
model.classifier[6] = nn.Sequential(
    BernoulliDropout(p=0.5),  # Adding Bernoulli Dropout with 50% probability
    nn.Linear(4096, 2)
)

# Print model structure
print(model)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Move model to GPU
model.to('cuda')

# Early Stopping variables
best_val_loss = float('inf')  # Initialize best validation loss to a large value
patience = 3  # Number of epochs to wait for improvement in validation loss
epochs_without_improvement = 0
best_model_weights = None

# Train the model with Bernoulli Dropout
for epoch in range(5):
    model.train()
    running_loss = 0.0
    for input, target in train_loader:
        input, target = input.to('cuda'), target.to('cuda')
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # After each epoch, calculate validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.to('cuda'), target.to('cuda')
            output = model(input)
            loss = criterion(output, target)
            val_loss += loss.item()

    val_loss /= len(test_loader)  # Average validation loss

    print(f'Epoch {epoch + 1} - Training Loss: {running_loss}, Validation Loss: {val_loss}')

    # Check if validation loss improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_weights = model.state_dict()  # Save the best model weights
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    # Early stopping condition: stop training if validation loss doesn't improve
    if epochs_without_improvement >= patience:
        print(f'Early stopping at epoch {epoch + 1}')
        model.load_state_dict(best_model_weights)  # Load the best weights
        break

# Evaluate model accuracy on the test set
correct, total = 0, 0
model.eval()
with torch.no_grad():
    for input, target in test_loader:
        input, target = input.to('cuda'), target.to('cuda')
        output = model(input)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy_with_bernoulli_dropout = 100 * correct / total
print(f'Accuracy with Bernoulli Dropout: {accuracy_with_bernoulli_dropout:.2f}%')

# ------------------------------------------
# Now, let's train the model without Dropout (remove Dropout layer)
model = models.alexnet(weights='IMAGENET1K_V1')

# Freeze all layers except the final classifier
for param in model.parameters():
    param.requires_grad = False

# Replace the final classifier layer with a new one for binary classification (dog vs cat) without Dropout
model.classifier[6] = nn.Linear(4096, 2)

# Loss function and optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Move model to GPU
model.to('cuda')

# Early Stopping variables (same as before)
best_val_loss = float('inf')
epochs_without_improvement = 0
best_model_weights = None

# Train the model without Dropout
for epoch in range(5):
    model.train()
    running_loss = 0.0
    for input, target in train_loader:
        input, target = input.to('cuda'), target.to('cuda')
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # After each epoch, calculate validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.to('cuda'), target.to('cuda')
            output = model(input)
            loss = criterion(output, target)
            val_loss += loss.item()

    val_loss /= len(test_loader)  # Average validation loss

    print(f'Epoch {epoch + 1} - Training Loss: {running_loss}, Validation Loss: {val_loss}')

    # Check if validation loss improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_weights = model.state_dict()  # Save the best model weights
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    # Early stopping condition: stop training if validation loss doesn't improve
    if epochs_without_improvement >= patience:
        print(f'Early stopping at epoch {epoch + 1}')
        model.load_state_dict(best_model_weights)  # Load the best weights
        break

# Evaluate model accuracy on the test set without Dropout
correct, total = 0, 0
model.eval()
with torch.no_grad():
    for input, target in test_loader:
        input, target = input.to('cuda'), target.to('cuda')
        output = model(input)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy_without_dropout = 100 * correct / total
print(f'Accuracy without Dropout: {accuracy_without_dropout:.2f}%')
