import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            ConvBlock(1, 32),
            nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True),
            ConvBlock(32, 64),
            nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True),
            ConvBlock(64, 128),
            nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True),
            ConvBlock(128, 256),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.fc(x)
        return x.view(x.size(0), -1)
