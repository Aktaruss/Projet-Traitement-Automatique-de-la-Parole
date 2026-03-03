import torch
import torch.nn as nn
import torch.nn.functional as F

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=True)

    def forward(self, x):
        return self.lora_B(self.lora_A(x))

class TDNN(nn.Module):
    def __init__(self, input_size, output_size):
        """
        input_size: 1 (car ton format est 100, 1, 40, 97)
        output_size: nombre de classes
        """
        super(TDNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=40, out_channels=256, kernel_size=5, dilation=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1)
        self.lin = LowRankLinear(128 * 2, 32, rank=2)
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, output_size)

        self.dropOut = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        x = self.dropOut(self.relu(self.conv1(x)))
        x = self.dropOut(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)
        x = torch.cat((mean, std), dim=1)
        x = self.relu(self.lin(x))
        x = self.dropOut(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def check_size(self, batch_size=100):
        x = torch.rand(batch_size, 1, 40, 97)
        print(f"Entrée : {x.size()}")
        x = x.squeeze(1)
        x = self.conv1(x)
        print(f"Après TDNN-1 (Conv1d) : {x.size()}")
        x = self.conv2(x)
        print(f"Après TDNN-2 (Conv1d) : {x.size()}")
        x = self.conv3(x)
        print(f"Après TDNN-3 (Conv1d) : {x.size()}")
        x = torch.randn(batch_size, 256)
        x = self.lin(x)
        print(f"Après LowRankLinear : {x.size()}")