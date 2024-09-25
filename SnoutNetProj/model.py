import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# Define the CNN Model
class SnoutNet(nn.Module):
    def __init__(self):
        super(SnoutNet, self).__init__()

        # First convolutional layer (input channels = 3 for RGB image like MNIST) total input is N*B*h*w = 154 587
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=0)
        # Max pooling layer (2x2 pool size)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)

        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Second convolutional layer
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)

        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Fully connected layer at the end (flatten and then pass through FC)
        self.fc1 = nn.Linear(4096, 1024)  # Fc1 is 1x1024
        self.fc2 = nn.Linear(1024, 1024)  # Fc2 is 1x1024
        self.fc3 = nn.Linear(1024, 2)  # output layer

    def forward(self, x):
        # Convolutional layer 1 + ReLU + MaxPooling
        x = self.pool1(F.relu(self.conv1(x)))
        # Convolutional layer 2 + ReLU + MaxPooling
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 4*4*256)
        # Fully connected layer 1 + ReLU
        x = F.relu(self.fc1(x))

        # Fully connected layer 2 (output layer)
        x = self.fc2(x)
        x=self.fc3(x)
        return x


# Instantiate the model, define the loss function and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SnoutNet().to(device)
criterion = nn.CrossEntropyLoss()  # Use cross-entropy loss for multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(summary(model, (3, 227, 227)))
