import os
import string
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Data path
data_path = "q2data/data/names/"
languages = os.listdir(data_path)

names = []
labels = []

# Go through each language file
for language in languages:
    if language.endswith(".txt"):
        language_name = language.split(".")[0]  # Extract language name
        file_path = os.path.join(data_path, language)

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                name = line.strip()
                if name:  # Avoid empty lines
                    names.append(name)
                    labels.append(language_name)

# Create a character-to-index mapping
alphabet = string.ascii_lowercase + " "  # You can also add punctuation if needed

char_to_int = {char: i + 1 for i, char in enumerate(alphabet)}  # 0 is reserved for padding
int_to_char = {i + 1: char for i, char in enumerate(alphabet)}


# Convert names to sequences of integers
def name_to_sequence(name):
    return [char_to_int[char.lower()] for char in name if char.lower() in char_to_int]


X = [name_to_sequence(name) for name in names]

# Pad the sequences to make them of equal length (use a maximum length based on your data)
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable

max_length = max(len(seq) for seq in X)  # You can set a fixed length based on the maximum name length

# Pad sequences manually
X_padded = [seq + [0] * (max_length - len(seq)) for seq in X]  # Pad with 0 (for padding token)

# Convert the labels (languages) to numeric values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)  # Convert language names to integer labels

# Convert to tensors
X_tensor = torch.tensor(X_padded, dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.long)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Create DataLoader for training and testing sets
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


class NameLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, max_length):
        super(NameLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        out = rnn_out[:, -1, :]  # Use the last output of the RNN
        out = self.fc(out)
        return out


# Hyperparameters
vocab_size = len(char_to_int) + 1  # +1 because we have padding
embed_size = 64
hidden_size = 128
output_size = len(label_encoder.classes_)
max_length = max(len(seq) for seq in X)

# Initialize the model
model = NameLanguageModel(vocab_size, embed_size, hidden_size, output_size, max_length)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for inputs, targets in train_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == targets).sum().item()
        total_predictions += targets.size(0)

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct_predictions / total_predictions

    # Validation
    model.eval()
    with torch.no_grad():
        correct_predictions = 0
        total_predictions = 0
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_predictions += targets.size(0)

        test_accuracy = correct_predictions / total_predictions

    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')


# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    correct_predictions = 0
    total_predictions = 0
    for inputs, targets in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == targets).sum().item()
        total_predictions += targets.size(0)

    test_accuracy = correct_predictions / total_predictions
    print(f'Test Accuracy: {test_accuracy:.4f}')

