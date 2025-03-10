import torch
import torch.nn as nn
import torch.nn.functional as F
from .bic_layer import BiCLayer

class CNN1(nn.Module):
    def __init__(self, numclass=10):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1)  # 28x28 → 28x28
        self.norm1 = nn.GroupNorm(32, 128)
        self.avg_pooling1 = nn.AvgPool2d(kernel_size=2, stride=2)  # 28x28 → 14x14

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)  # 14x14 → 14x14
        self.norm2 = nn.GroupNorm(32, 128)
        self.avg_pooling2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 14x14 → 7x7

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)  # 7x7 → 7x7
        self.norm3 = nn.GroupNorm(32, 128)
        self.avg_pooling3 = nn.AvgPool2d(kernel_size=2, stride=2)  # 7x7 → 3x3

        self.classifier = nn.Linear(2048, numclass)

        self.bic = BiCLayer(numclass)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.avg_pooling1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.avg_pooling2(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = F.relu(out)
        out = self.avg_pooling3(out)

        out = out.view(out.size(0), -1)
        out = self.classifier(out)


        out = self.bic(out)
        return out
    
class ETC_CNN(nn.Module):
  def __init__(self, numclass=3):
    super(ETC_CNN, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=5, padding='same')
    self.relu1 = nn.ReLU()
    self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, padding='same')
    self.relu2 = nn.ReLU()
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.drop1 = nn.Dropout(0.25)

    self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
    self.relu3 = nn.ReLU()
    self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding='same')
    self.relu4 = nn.ReLU()
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.drop2 = nn.Dropout(0.25)

    self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
    self.relu5 = nn.ReLU()
    self.conv6 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding='same')
    self.relu6 = nn.ReLU()
    self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.drop3 = nn.Dropout(0.25)

    self.fc1 = nn.Linear(in_features=16*2*16, out_features=256)
    self.drop4 = nn.Dropout(0.1)
    self.fc2 = nn.Linear(in_features=256, out_features=numclass)

    self.bic = BiCLayer(numclass)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = self.pool1(x)
    x = self.drop1(x)

    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    x = self.pool2(x)
    x = self.drop2(x)

    x = F.relu(self.conv5(x))
    x = F.relu(self.conv6(x))
    x = self.pool3(x)
    x = self.drop3(x)

    x = torch.flatten(x, start_dim=1)
    x = F.relu(self.fc1(x))
    x = self.drop4(x)
    x = self.fc2(x)

    x = self.bic(x)
    return x