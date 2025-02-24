import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Gaussian Noise transformation
class Gaussian(object):
    def __init__(self, mean: float, var: float):
        self.mean = mean
        self.var = var

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        noise = torch.normal(self.mean, self.var, img.size())
        return img + noise


# Preprocessing pipeline
preprocess = T.Compose([
    T.Resize((224, 224)),  # Resize all images to 224x224
    T.ToTensor(),
    T.RandomHorizontalFlip(),
    T.RandomRotation(45),
    Gaussian(0, 0.15),
])


class MyDataset(Dataset):
    def __init__(self, transform=None, strin="train"):
        self.imgs_path = f"./dog-cat-full-dataset-master/data/{strin}/"
        file_list = glob.glob(self.imgs_path + "*")
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for img_path in glob.glob(class_path + "/*.jpg"):
                self.data.append([img_path, class_name])
        self.class_map = {"dogs": 0, "cats": 1}
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = Image.open(img_path).convert('RGB')  # Ensure RGB format
        class_id = self.class_map[class_name]
        class_id = torch.tensor(class_id)
        if self.transform:
            img = self.transform(img)
        return img, class_id


# Dataset and DataLoader
dataset = MyDataset(transform=preprocess, strin="train")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# CNN Model Definition
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 64),  # Adjust this based on the final image size
            nn.ReLU(),
            nn.Linear(64, 2)  # 2 classes: cats and dogs
        )

    def forward(self, x):
        print("Input shape:", x.shape)  # Debug: check input shape
        x = self.model(x)
        print("After conv layers:", x.shape)  # Debug: check shape after conv layers
        x = self.fc_layers(x)
        return x


# Training Loop

# Model, Optimizer, and Loss Function
model = CNN()
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

loss_lost = []

for epoch in range(5):
    epoch_loss = 0.0
    model.train()

    for img, label in dataloader:
        # Move tensors to GPU if available
        img, label = img.cuda(), label.cuda() if torch.cuda.is_available() else (img, label)

        # Forward pass
        outputs = model(img)  # This is where the error occurs
        loss = criterion(outputs, label)
        epoch_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_lost.append(epoch_loss / len(dataloader))
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader)}")

# Optionally, plot the loss curve
plt.plot(loss_lost)
plt.title("Training Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
