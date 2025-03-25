import torch
import torch.nn as nn
import torch.nn.functional as F
from .bic_layer import BiCLayer
import torchvision.models as models


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

class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(weights=None)

        # Thay đổi input layer vì CIFAR-10 có ảnh 32x32 (ResNet-50 mặc định dùng 224x224)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Điều chỉnh Fully Connected Layer cho 10 classes của CIFAR-10
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        self.bic = BiCLayer(num_classes)
    def forward(self, x):
        x = self.model(x)
        x = self.bic(x)
        return x
    

#Mo hinh cho phan loai luu luong mang
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

    self.fc1 = nn.Linear(in_features=16*2*8, out_features=256)
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

class ETC_CNN_0_1(nn.Module):
    def __init__(self, numclass=3):
        super(ETC_CNN_0_1, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=5, padding='same')
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding='same')
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(0.25)

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()
        
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding='same')
        self.bn6 = nn.BatchNorm2d(16)
        self.relu6 = nn.ReLU()
        
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(in_features=16*2*8, out_features=256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.drop4 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(in_features=256, out_features=numclass)
        
        self.bic = BiCLayer(numclass)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        x = torch.flatten(x, start_dim=1)
        x = self.bn_fc1(F.relu(self.fc1(x)))
        x = self.drop4(x)
        x = self.fc2(x)

        x = self.bic(x)
        return x

class ETC_CNN2(nn.Module):
    def __init__(self, num_classes=3):
        super(ETC_CNN2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Tính toán kích thước sau khi qua các lớp pooling
        self.fc1 = nn.Linear(128 * 2 * 8, 256)  # Đầu vào 20x64 -> sau 3 lần pooling còn 2x8
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.bic = BiCLayer(num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bic(x)
        return x

class ETC_CNN3(nn.Module):
    def __init__(self, num_classes=3):
        super(ETC_CNN3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.drop1 = nn.Dropout(0.3)  # Dropout sau conv2

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.drop2 = nn.Dropout(0.3)  # Dropout sau conv3

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(2048, 256)  # Đầu vào 20x64 -> sau 3 lần pooling còn 2x16
        self.bn_fc = nn.BatchNorm1d(256)
        self.drop_fc = nn.Dropout(0.5)  # Dropout trước FC2
        self.fc2 = nn.Linear(256, num_classes)
        self.bic = BiCLayer(num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.drop1(x)  # Áp dụng Dropout sau MaxPool

        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.drop2(x)  # Áp dụng Dropout sau MaxPool

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.drop_fc(x)  # Áp dụng Dropout trước FC2
        x = self.fc2(x)
        x = self.bic(x)
        return x
    
class ETC_RESNET18(nn.Module):
    def __init__(self, num_classes=3):
        super(ETC_RESNET18, self).__init__()
        self.model = models.resnet18(weights=None)
        
        # Chỉnh sửa lớp đầu vào để phù hợp với dữ liệu 1 kênh
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Thay đổi lớp fully connected cuối cùng để phù hợp với số lượng classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        self.bic = BiCLayer(num_classes)
    def forward(self, x):
        x = self.model(x)
        x = self.bic(x)
        return x
    
class ETC_CNN1D(nn.Module):
    def __init__(self, input_channels=1, output_size=3):
        super(ETC_CNN1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.3)  # Dropout sau conv2

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(0.3)  # Dropout sau conv4

        #final_feature_map_size = 1280 // 4  # Sau 2 lớp MaxPool1d
        self.fc1 = nn.Linear(512 , 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.drop_fc = nn.Dropout(0.5)  # Dropout trước FC2
        self.fc2 = nn.Linear(256, output_size)
        self.bic = BiCLayer(output_size)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.drop1(x)  # Áp dụng Dropout sau MaxPool1d

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.drop2(x)  # Áp dụng Dropout sau MaxPool1d

        x = x.view(x.size(0), 512, -1)  # Tính toán kích thước động
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.drop_fc(x)  # Áp dụng Dropout trước FC2
        x = self.fc2(x)
        x = self.bic(x)
        return x