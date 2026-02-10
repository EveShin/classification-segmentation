import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAM(nn.Module):
    def __init__(self, c_out, reduction=16):
        super(CBAM, self).init()
        self.c_out = c_out
        self.ch_attn = nn.Sequential(
            nn.Linear(c_out, c_out // reduction),
            nn.ReLU(),
            nn.Linear(c_out // reduction, c_out),
        )
        self.sp_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_ch_avg = torch.mean(x, dim=(2, 3))
        x_ch_max, _ = torch.max(x.view(x.size(0), x.size(1), -1), dim=2)
        x_ch_at = torch.sigmoid(self.ch_attn(x_ch_avg) + self.ch_attn(x_ch_max))
        x_ch_at = x_ch_at.view(-1, self.c_out, 1, 1)
        x_cbam = x * x_ch_at
        x_sp_avg = torch.mean(x_cbam, dim=1, keepdim=True)
        x_sp_max, _ = torch.max(x_cbam, dim=1, keepdim=True)
        x_sp = torch.cat([x_sp_avg, x_sp_max], dim=1)
        return x_cbam * self.sp_attn(x_sp) + x

# --- ResNet (Standard & CBAM 통합) ---
class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out)) if stride != 1 or c_in != c_out else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        return out + identity

class IntegratedResNet(nn.Module):
    def __init__(self, outputsize=200, use_cbam=False):
        super(IntegratedResNet, self).__init__()
        self.use_cbam = use_cbam
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.layer1 = nn.Sequential(ResBlock(64, 64), ResBlock(64, 64), ResBlock(64, 64))
        self.layer2 = nn.Sequential(ResBlock(64, 128, 2), ResBlock(128, 128), ResBlock(128, 128))
        self.layer3 = nn.Sequential(ResBlock(128, 256, 2), ResBlock(256, 256), ResBlock(256, 256))
        self.layer4 = nn.Sequential(ResBlock(256, 512, 2), ResBlock(512, 512), ResBlock(512, 512))
        self.cbams = nn.ModuleDict({'cbam2': CBAM(128), 'cbam3': CBAM(256), 'cbam4': CBAM(512)}) if use_cbam else None
        self.bn_final = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, outputsize)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        if self.use_cbam: x = self.cbams['cbam2'](x)
        x = self.layer3(x)
        if self.use_cbam: x = self.cbams['cbam3'](x)
        x = self.layer4(x)
        if self.use_cbam: x = self.cbams['cbam4'](x)
        x = torch.flatten(F.adaptive_avg_pool2d(F.relu(self.bn_final(x)), (1, 1)), 1)
        return self.fc(x)

# --- DenseNet (Standard & CBAM 통합) ---
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, dropout_rate=0.2):
        super(DenseLayer, self).init()
        self.net = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True), nn.Conv2d(in_channels, 4*growth_rate, 1, bias=False),
                                 nn.BatchNorm2d(4*growth_rate), nn.ReLU(inplace=True), nn.Conv2d(4*growth_rate, growth_rate, 3, padding=1, bias=False))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return torch.cat([x, self.dropout(self.net(x))], 1)

class IntegratedDenseNet(nn.Module):
    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16), num_classes=200, use_cbam=False):
        super(IntegratedDenseNet, self).__init__()
        num_features = 64
        self.features = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        for i, num_layers in enumerate(block_config):
            for j in range(num_layers):
                self.features.add_module(f'db{i+1}_l{j+1}', DenseLayer(num_features, growth_rate))
                num_features += growth_rate
            if use_cbam and i >= 2: self.features.add_module(f'cbam{i+1}', CBAM(num_features))
            if i != len(block_config) - 1:
                trans = nn.Sequential(nn.BatchNorm2d(num_features), nn.ReLU(inplace=True), nn.Conv2d(num_features, num_features//2, 1, bias=False), nn.AvgPool2d(2, 2))
                self.features.add_module(f'trans{i+1}', trans)
                num_features //= 2
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(F.relu(self.features(x)), (1, 1))
        return self.classifier(torch.flatten(out, 1))