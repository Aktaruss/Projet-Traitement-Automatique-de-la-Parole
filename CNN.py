import torch.nn as nn
import torch.nn.functional as F
import torch

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=True)

    def forward(self, x):
        return self.lora_B(self.lora_A(x))

class CNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_size,out_channels=64,kernel_size=(20,8),stride=(1,3),padding=(10,4))
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(10,4),stride=1,padding=(5,2))

        self.lin = LowRankLinear(64 * 100 * 5, 32, rank=2)
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, output_size)

        self.dropOut = nn.Dropout(p = 0.5)
        self.maxPool = nn.MaxPool2d((1, 3))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropOut(self.relu(self.conv1(x)))
        x = self.maxPool(x)
        x = self.dropOut(self.relu(self.conv2(x)))
        x = self.relu(self.lin(x.view(x.size(0), -1)))
        x = self.dropOut(self.relu(self.fc1(x)))
        x = self.relu(self.fc2(x))
        return x

    def check_size(self):
        x = torch.rand(1, 1, 98, 40)
        x = self.dropOut(self.relu(self.conv1(x)))
        print(f"Taille après le conv1 : {x.size()}")
        x = self.maxPool(x)
        print(f"Taille après le maxPool : {x.size()}")
        x = self.dropOut(self.relu(self.conv2(x)))
        print(f"Taille après le conv2 : {x.size()}")
        x = self.relu(self.lin(x.view(x.size(0), -1)))
        print(f"Taille après le lin : {x.size()}")
        x = self.dropOut(self.relu(self.fc1(x)))
        print(f"Taille après le FC1 : {x.size()}")
        x = self.relu(self.fc2(x))
        print(f"Taille après le FC2 : {x.size()}")