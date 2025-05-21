
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


cascade = cv2.CascadeClassifier("haarcascade_eye.xml")


# 验证设置是否成功


def normalize(x):
    minn, maxx = x.min(), x.max()
    return (x - minn) / (maxx - minn)


# width, height = 2559, 1439
# root = ""
# filepaths = os.listdir(root)
# X, Y = [], []
# for filepath in filepaths:
#     x, y, _ = filepath.split(' ')
#     x = float(x) / width
#     y = float(y) / height
#     X.append(cv2.imread(root + filepath))
#     Y.append([x, y])
#     X = np.array(X) / 255.0
#     Y = np.array(Y)
#     print (X.shape, Y.shape)

# 定义ResNeXt块
class ResNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, cardinality=4, width=4):
        super(ResNeXtBlock, self).__init__()
        
        # 计算组卷积的通道数
        group_width = cardinality * width
        
        self.conv1 = nn.Conv2d(in_channels, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        
        # 组卷积
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, 
                              padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        
        self.conv3 = nn.Conv2d(group_width, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

# 定义ResNeXt18模型
class ResNeXt18(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNeXt18, self).__init__()
        
        self.in_channels = 64
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNeXt层
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.sigmoid = nn.Sigmoid()  # 用于坐标预测
        
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResNeXtBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x


def train_model(data_dir, model_save_path="eye_tracking_model.pth", epochs=200):
    # 加载数据
    filepaths = os.listdir(data_dir)
    X, Y = [], []
    for filepath in filepaths:
        x, y, _ = filepath.split(' ')
        x = float(x) / width
        y = float(y) / height
        X.append(cv2.imread(os.path.join(data_dir, filepath)))
        Y.append([x, y])
    
    X = np.array(X) / 255.0
    Y = np.array(Y)
    print(f"数据集大小: {X.shape}, {Y.shape}")

    # 准备数据
    X_tensor = torch.FloatTensor(X).permute(0, 3, 1, 2)
    Y_tensor = torch.FloatTensor(Y)
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNeXt18().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    # 训练模型
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

    # 保存模型
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存到: {model_save_path}")

if __name__ == "__main__":
    train_model("dataset")
