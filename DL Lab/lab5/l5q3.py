from itertools import batched

import torch
import torch.nn as nn

class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3),
                                 nn.ReLU(),
                                 nn.MaxPool2d((2,2), stride=2),
                                 nn.Conv2d(64,128, kernel_size=3),
                                 nn.ReLU(),
                                 nn.MaxPool2d((2,2), stride=2),
                                 nn.Conv2d(128,64,kernel_size=3),
                                 nn.ReLU(),
                                 nn.MaxPool2d((2,2), stride=2))

        self.classification_head = nn.Sequential(nn.Linear(64,20, bias=True),
                                                 nn.ReLU(),
                                                 nn.Linear(20,10,bias=True))

    def forward(self, x):
        features = self.net(x)
        return self.classification_head(features.view(batch_size,-1))