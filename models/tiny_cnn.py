import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MaxPool2d


class Tiny_CNN(nn.Module):
    def __init__(self):
        super(Tiny_CNN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3)
        # Second convolutional layer (fix: use different name, not conv1 again!)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3)

        # Maxpooling
        self.pool = MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(4 * 12 * 12, 32)  # After 2 convs + pool
        self.fc2 = nn.Linear(32, 10)  # Output 10 logits (digits)

    def forward(self, x):
        x = F.relu(self.conv1(x))    # Conv1 + ReLU
        x = F.relu(self.conv2(x))    # Conv2 + ReLU
        x = self.pool(x)             # MaxPool
        x = x.view(x.size(0), -1)    # Flatten
        x = F.relu(self.fc1(x))      # FC1 + ReLU
        x = self.fc2(x)              # FC2 (no softmax)
        return x


model = Tiny_CNN()
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")
