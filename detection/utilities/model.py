import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor, Resize, Grayscale, Normalize


transforms = torchvision.transforms.Compose([ToTensor(),
                                             Grayscale(1),
                                             Resize([32, 32]),
                                             Normalize(mean=0, std=1)])


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        return self.block(x)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.N_CLASSES = 32  # Is not a parameter for reason save/load class from different files
        self.model = nn.Sequential(
            ConvBlock(1, 16),
            nn.MaxPool2d(2),
            ConvBlock(16, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            nn.Flatten()
        )

        cnn_out = self.model(torch.ones(1, 1, 32, 32))
        print(cnn_out.size())

        self.fc = nn.Sequential(
            nn.Linear(cnn_out.size()[1], 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, self.N_CLASSES)
        )

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x
