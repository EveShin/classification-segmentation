import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. MLP 구조 (network_0927.py 기반 초기 모델)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # x shape: [Batch, Channel * H * W]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# 2. Plain CNN (network_0929 ~ 1004 통합 버전)
class PlainCNN(nn.Module):
    def __init__(self, outputsize=10, version='final'):
        super(PlainCNN, self).__init__()
        self.version = version

        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

        # FC Layers (버전에 따라 구성 변경)
        if version == 'deep_fc':
            self.fc1 = nn.Linear(4 * 4 * 256, 512)
            self.fc2 = nn.Linear(512, outputsize)
        else:
            self.fc1 = nn.Linear(4 * 4 * 256, outputsize)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.maxpool(self.relu(self.bn2(self.conv2(x))))
        x = self.maxpool(self.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)

        if self.version == 'deep_fc':
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
        else:
            x = self.fc1(x)
        return x


# 3. Manual ResNet-20 (network.py / network_1012 기반)
# 레이어를 직접 하나씩 선언하여 구조를 명시적으로 파악하기 좋은 버전
class ResNetManual(nn.Module):
    def __init__(self, outputsize=10):
        super(ResNetManual, self).__init__()
        self.relu = nn.ReLU()

        # Initial Conv
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Stage 1: 16 channels, 32x32
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1, bias=False);
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1, bias=False);
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1, bias=False);
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 16, 3, padding=1, bias=False);
        self.bn5 = nn.BatchNorm2d(16)
        self.conv6 = nn.Conv2d(16, 16, 3, padding=1, bias=False);
        self.bn6 = nn.BatchNorm2d(16)
        self.conv7 = nn.Conv2d(16, 16, 3, padding=1, bias=False);
        self.bn7 = nn.BatchNorm2d(16)

        # Stage 2: 32 channels, 16x16
        self.conv8 = nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False);
        self.bn8 = nn.BatchNorm2d(32)
        self.conv9 = nn.Conv2d(32, 32, 3, padding=1, bias=False);
        self.bn9 = nn.BatchNorm2d(32)
        self.conv10 = nn.Conv2d(32, 32, 3, padding=1, bias=False);
        self.bn10 = nn.BatchNorm2d(32)
        self.conv11 = nn.Conv2d(32, 32, 3, padding=1, bias=False);
        self.bn11 = nn.BatchNorm2d(32)
        self.conv12 = nn.Conv2d(32, 32, 3, padding=1, bias=False);
        self.bn12 = nn.BatchNorm2d(32)
        self.conv13 = nn.Conv2d(32, 32, 3, padding=1, bias=False);
        self.bn13 = nn.BatchNorm2d(32)

        # Stage 3: 64 channels, 8x8
        self.conv14 = nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False);
        self.bn14 = nn.BatchNorm2d(64)
        self.conv15 = nn.Conv2d(64, 64, 3, padding=1, bias=False);
        self.bn15 = nn.BatchNorm2d(64)
        self.conv16 = nn.Conv2d(64, 64, 3, padding=1, bias=False);
        self.bn16 = nn.BatchNorm2d(64)
        self.conv17 = nn.Conv2d(64, 64, 3, padding=1, bias=False);
        self.bn17 = nn.BatchNorm2d(64)
        self.conv18 = nn.Conv2d(64, 64, 3, padding=1, bias=False);
        self.bn18 = nn.BatchNorm2d(64)
        self.conv19 = nn.Conv2d(64, 64, 3, padding=1, bias=False);
        self.bn19 = nn.BatchNorm2d(64)

        # Final Layers
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, outputsize)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        # Stage 1
        identity = x
        out = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(out)) + identity)
        identity = x
        out = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(out)) + identity)
        identity = x
        out = self.relu(self.bn6(self.conv6(x)))
        x = self.relu(self.bn7(self.conv7(out)) + identity)

        # Stage 2 (Downsampling with Padding)
        identity = F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, 8, 8), "constant", 0)
        out = self.relu(self.bn8(self.conv8(x)))
        x = self.relu(self.bn9(self.conv9(out)) + identity)
        identity = x
        out = self.relu(self.bn10(self.conv10(x)))
        x = self.relu(self.bn11(self.conv11(out)) + identity)
        identity = x
        out = self.relu(self.bn12(self.conv12(x)))
        x = self.relu(self.bn13(self.conv13(out)) + identity)

        # Stage 3 (Downsampling with Padding)
        identity = F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, 16, 16), "constant", 0)
        out = self.relu(self.bn14(self.conv14(x)))
        x = self.relu(self.bn15(self.conv15(out)) + identity)
        identity = x
        out = self.relu(self.bn16(self.conv16(x)))
        x = self.relu(self.bn17(self.conv17(out)) + identity)
        identity = x
        out = self.relu(self.bn18(self.conv18(x)))
        x = self.relu(self.bn19(self.conv19(out)) + identity)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# 4. Pre-activation ResNet (network_v2.py 기반)
class PreActResNet(nn.Module):
    def __init__(self, outputsize=10):
        super(PreActResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)

        # Stage 1 Layers
        self.bn1 = nn.BatchNorm2d(16);
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16);
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1, bias=False)

        # Stage 2 Layers (Downsampling)
        self.bn3 = nn.BatchNorm2d(16);
        self.conv4 = nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(32);
        self.conv5 = nn.Conv2d(32, 32, 3, padding=1, bias=False)

        # Stage 3 Layers (Downsampling)
        self.bn5 = nn.BatchNorm2d(32);
        self.conv6 = nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(64);
        self.conv7 = nn.Conv2d(64, 64, 3, padding=1, bias=False)

        self.bn_final = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64, outputsize)

    def forward(self, x):
        x = self.conv1(x)

        # Stage 1
        identity = x
        out = self.conv2(F.relu(self.bn1(x)))
        out = self.conv3(F.relu(self.bn2(out)))
        x = out + identity

        # Stage 2
        out = F.relu(self.bn3(x))
        identity = F.pad(out[:, :, ::2, ::2], (0, 0, 0, 0, 8, 8), "constant", 0)
        out = self.conv4(out)
        out = self.conv5(F.relu(self.bn4(out)))
        x = out + identity

        # Stage 3
        out = F.relu(self.bn5(x))
        identity = F.pad(out[:, :, ::2, ::2], (0, 0, 0, 0, 16, 16), "constant", 0)
        out = self.conv6(out)
        out = self.conv7(F.relu(self.bn6(out)))
        x = out + identity

        x = F.relu(self.bn_final(x))
        x = F.avg_pool2d(x, 8)
        x = x.view(x.size(0), -1)
        return self.fc(x)