import torch
import torch.nn as nn

def conv_bn_relu(in_ch, out_ch, k_size=3, p_size=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=p_size, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1, use_bn_in_shortcut=False):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(c_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False)

        if stride != 1 or c_in != c_out:
            layers = [nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False)]
            if use_bn_in_shortcut:  # network_unet_4444.py 버전 대응
                layers.append(nn.BatchNorm2d(c_out))
            self.shortcut = nn.Sequential(*layers)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        return out


class FCN_8S(nn.Module):
    def __init__(self, class_num=21):
        super(FCN_8S, self).__init__()
        self.vgg_part1 = nn.Sequential(
            conv_bn_relu(3, 64), conv_bn_relu(64, 64), nn.MaxPool2d(2, 2),
            conv_bn_relu(64, 128), conv_bn_relu(128, 128), nn.MaxPool2d(2, 2),
            conv_bn_relu(128, 256), conv_bn_relu(256, 256), conv_bn_relu(256, 256), nn.MaxPool2d(2, 2),
        )
        self.vgg_part2 = nn.Sequential(
            conv_bn_relu(256, 512), conv_bn_relu(512, 512), conv_bn_relu(512, 512), nn.MaxPool2d(2, 2),
        )
        self.vgg_part3 = nn.Sequential(
            conv_bn_relu(512, 512), conv_bn_relu(512, 512), conv_bn_relu(512, 512), nn.MaxPool2d(2, 2),
            conv_bn_relu(512, 4096, k_size=1, p_size=0),
            conv_bn_relu(4096, 4096, k_size=1, p_size=0),
            nn.Conv2d(4096, class_num, kernel_size=1),
        )
        self.predict_part1 = nn.Conv2d(256, class_num, kernel_size=1)
        self.predict_part2 = nn.Conv2d(512, class_num, kernel_size=1)
        self.up_conv_2x_1 = nn.ConvTranspose2d(class_num, class_num, kernel_size=4, stride=2, padding=1)
        self.up_conv_2x_2 = nn.ConvTranspose2d(class_num, class_num, kernel_size=4, stride=2, padding=1)
        self.up_conv_8x = nn.ConvTranspose2d(class_num, class_num, kernel_size=16, stride=8, padding=4)

    def forward(self, x):
        x = self.vgg_part1(x)
        x_p1 = self.predict_part1(x)
        x = self.vgg_part2(x)
        x_p2 = self.predict_part2(x)
        x = self.vgg_part3(x)
        x = self.up_conv_2x_2(x) + x_p2
        x = self.up_conv_2x_1(x) + x_p1
        return self.up_conv_8x(x)


class ResUNet(nn.Module):
    def __init__(self, class_num=21, layers=[3, 3, 3, 3], use_bn_shortcut=False):
        super(ResUNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)

        # Encoder
        self.enc_b1 = self._make_layer(64, 64, layers[0], stride=1, bn=use_bn_shortcut)
        self.enc_b2 = self._make_layer(64, 128, layers[1], stride=2, bn=use_bn_shortcut)
        self.enc_b3 = self._make_layer(128, 256, layers[2], stride=2, bn=use_bn_shortcut)
        self.bridge = self._make_layer(256, 512, layers[3], stride=2, bn=use_bn_shortcut)

        # Decoder
        self.up2x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.tran_1 = nn.Conv2d(512 + 256, 256, kernel_size=1)
        self.dec_b1 = self._make_layer(256, 256, layers[2], bn=use_bn_shortcut)
        self.tran_2 = nn.Conv2d(256 + 128, 128, kernel_size=1)
        self.dec_b2 = self._make_layer(128, 128, layers[1], bn=use_bn_shortcut)
        self.tran_3 = nn.Conv2d(128 + 64, 64, kernel_size=1)
        self.dec_b3 = self._make_layer(64, 64, layers[0], bn=use_bn_shortcut)

        self.predict = nn.Conv2d(64, class_num, kernel_size=1)

    def _make_layer(self, in_ch, out_ch, blocks, stride=1, bn=False):
        layers = [ResBlock(in_ch, out_ch, stride, use_bn_in_shortcut=bn)]
        for _ in range(1, blocks):
            layers.append(ResBlock(out_ch, out_ch, 1, use_bn_in_shortcut=bn))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        s1 = self.enc_b1(x)
        s2 = self.enc_b2(s1)
        s3 = self.enc_b3(s2)
        x = self.bridge(s3)

        x = self.up2x(x)
        x = self.dec_b1(self.tran_1(torch.cat([x, s3], dim=1)))
        x = self.up2x(x)
        x = self.dec_b2(self.tran_2(torch.cat([x, s2], dim=1)))
        x = self.up2x(x)
        x = self.dec_b3(self.tran_3(torch.cat([x, s1], dim=1)))
        return self.predict(x)


class DeepUNet(nn.Module):
    def __init__(self, class_num=21):
        super(DeepUNet, self).__init__()
        self.enc_b1 = nn.Sequential(conv_bn_relu(3, 64), conv_bn_relu(64, 64))
        self.enc_b2 = nn.Sequential(nn.MaxPool2d(2), conv_bn_relu(64, 128), conv_bn_relu(128, 128))
        self.enc_b3 = nn.Sequential(nn.MaxPool2d(2), conv_bn_relu(128, 256), conv_bn_relu(256, 256),
                                    conv_bn_relu(256, 256))
        self.enc_b4 = nn.Sequential(nn.MaxPool2d(2), conv_bn_relu(256, 512), conv_bn_relu(512, 512),
                                    conv_bn_relu(512, 512))
        self.enc_b5 = nn.Sequential(nn.MaxPool2d(2), conv_bn_relu(512, 512), conv_bn_relu(512, 512),
                                    conv_bn_relu(512, 512))

        self.up2x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_b1 = nn.Sequential(conv_bn_relu(1024, 512), conv_bn_relu(512, 512), conv_bn_relu(512, 512))
        self.dec_b2 = nn.Sequential(conv_bn_relu(768, 256), conv_bn_relu(256, 256), conv_bn_relu(256, 256))
        self.dec_b3 = nn.Sequential(conv_bn_relu(384, 128), conv_bn_relu(128, 128), conv_bn_relu(128, 128))
        self.dec_b4 = nn.Sequential(conv_bn_relu(192, 64), conv_bn_relu(64, 64))
        self.predict = nn.Conv2d(64, class_num, kernel_size=1)

    def forward(self, x):
        s1 = self.enc_b1(x)
        s2 = self.enc_b2(s1)
        s3 = self.enc_b3(s2)
        s4 = self.enc_b4(s3)
        x = self.enc_b5(s4)
        x = self.dec_b1(torch.cat([self.up2x(x), s4], 1))
        x = self.dec_b2(torch.cat([self.up2x(x), s3], 1))
        x = self.dec_b3(torch.cat([self.up2x(x), s2], 1))
        x = self.dec_b4(torch.cat([self.up2x(x), s1], 1))
        return self.predict(x)