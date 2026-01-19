import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_size,out_channels=64,kernel_size=3,stride=5,padding=1)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=2,padding=1)

        self.lin = nn.Linear(64 * 10 * 4, 32)
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, output_size)

        self.dropOut = nn.Dropout(p = 0.5)
        self.maxPool = nn.MaxPool2d((1, 3),stride=(1,1),padding=(0,1))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropOut(self.relu(self.conv1(x)))
        x = self.maxPool(x)
        x = self.dropOut(self.relu(self.conv2(x)))
        x = self.relu(self.lin(x.view(x.size(0), -1)))
        x = self.dropOut(self.relu(self.fc1(x)))
        x = self.relu(self.fc2(x))
        return x