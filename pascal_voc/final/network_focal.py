import torch
import torch.nn as nn
import cv2
import numpy as np
import os

MEAN = np.array([103.9979349251002, 112.88047937486992, 116.36432290207493])
STD = np.array([71.6988692265365, 67.61791552564353, 68.49109366853313])

def load_semantic_seg_data(img_path, gt_path, size=256):
    VOC_COLORMAP = [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128], [128, 0, 0], [128, 0, 128],
                    [128, 128, 0], [128, 128, 128], [0, 0, 64], [0, 0, 192], [0, 128, 64], [0, 128, 192],
                    [128, 0, 64], [128, 0, 192], [128, 128, 64], [128, 128, 192], [0, 64, 0], [0, 64, 128],
                    [0, 192, 0], [0, 192, 128], [128, 64, 0]]

    img_names = os.listdir(img_path)
    gt_names = os.listdir(gt_path)

    imgs = np.zeros(shape=(len(img_names), size, size, 3), dtype=np.uint8)
    gts = np.zeros((len(gt_names), size, size), dtype=np.uint8)

    for it in range(len(img_names)):
        print("img %d / %d " % (it, len(img_names)))
        img = cv2.imread(img_path + img_names[it])
        img = cv2.resize(img, (size, size))
        imgs[it, :, :, :] = img

    for it in range(len(gt_names)):
        print("gt %d / %d " % (it, len(gt_names)))
        gt = cv2.imread(gt_path + gt_names[it])
        gt_index = np.zeros(shape=(gt.shape[0], gt.shape[1]), dtype=np.uint8)
        for ic in range(len(VOC_COLORMAP)):
            code = VOC_COLORMAP[ic]
            gt_index[np.where(np.all(gt == code, axis=-1))] = ic

        gt_index[np.where(np.all(gt == [192, 224, 224], axis=-1))] = 255

        gt_index = cv2.resize(gt_index, (size, size), interpolation=cv2.INTER_NEAREST)
        gts[it, :, :] = gt_index

    return imgs, gts

def random_horizontal_flip(img, gt, p=0.5):
    if np.random.rand() < p:
        img = np.fliplr(img)
        gt = np.fliplr(gt)

    return img, gt

def random_scale_crop(img, gt, scale_range=(0.8, 1.2), crop_size=256):
    h, w = img.shape[:2]
    scale = np.random.uniform(scale_range[0], scale_range[1])
    new_h, new_w = int(h * scale), int(w * scale)

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    if new_h < crop_size or new_w < crop_size:
        pad_h = max(crop_size - new_h, 0)
        pad_w = max(crop_size - new_w, 0)
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        gt = np.pad(gt, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=255)
        new_h, new_w = img.shape[:2]

    top = np.random.randint(0, new_h - crop_size + 1)
    left = np.random.randint(0, new_w - crop_size + 1)
    img = img[top:top + crop_size, left:left + crop_size, :]
    gt = gt[top:top + crop_size, left:left + crop_size]

    return img, gt

def color_jitter(img, brightness=0.1, contrast=0.1, saturation=0.1):
    img = img.copy().astype(np.float32)

    if np.random.rand() < 0.8:
        alpha = 1.0 + np.random.uniform(-brightness, brightness)
        img = np.clip(img * alpha, 0, 255)

    if np.random.rand() < 0.8:
        alpha = 1.0 + np.random.uniform(-contrast, contrast)
        mean = img.mean()
        img = np.clip((img - mean) * alpha + mean, 0, 255)

    if np.random.rand() < 0.8:
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        alpha = 1.0 + np.random.uniform(-saturation, saturation)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * alpha, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    return img.astype(np.uint8)

def Mini_batch_training_Seg(train_img, train_gt, batch_size, size=256, augmentation=True):
    batch_img = np.zeros((batch_size, size, size, 3))
    batch_gt = np.zeros((batch_size, size, size), dtype=np.int64)

    rand_num = np.random.randint(0, train_img.shape[0], size=batch_size)

    for it in range(batch_size):
        temp = rand_num[it]
        img = train_img[temp, :, :, :].copy()
        gt = train_gt[temp, :, :].copy()

        if augmentation:
            img = color_jitter(img, brightness=0.1, contrast=0.1, saturation=0.1)
            img, gt = random_horizontal_flip(img, gt, p=0.5)
            img, gt = random_scale_crop(img, gt, scale_range=(0.8, 1.2), crop_size=size)

        img = img.astype(np.float32)
        img = (img - MEAN) / STD

        batch_img[it, :, :, :] = img
        batch_gt[it, :, :] = gt

    return batch_img, batch_gt

class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(c_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False)

        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False)
            )
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

class ResUNet(nn.Module):
    def __init__(self, class_num=21):
        super(ResUNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)

        self.enc_b1 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64)
        )

        self.enc_b2 = nn.Sequential(
            ResBlock(64, 128, stride=2),  # Downsample
            ResBlock(128, 128),
            ResBlock(128, 128)
        )

        self.enc_b3 = nn.Sequential(
            ResBlock(128, 256, stride=2),  # Downsample
            ResBlock(256, 256),
            ResBlock(256, 256)
        )

        self.bridge = nn.Sequential(
            ResBlock(256, 512, stride=2),  # Downsample
            ResBlock(512, 512),
            ResBlock(512, 512)
        )

        self.tran_1 = nn.Conv2d(512 + 256, 256, kernel_size=1)
        self.dec_b1 = nn.Sequential(
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256)
        )

        self.tran_2 = nn.Conv2d(256 + 128, 128, kernel_size=1)
        self.dec_b2 = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128)
        )

        self.tran_3 = nn.Conv2d(128 + 64, 64, kernel_size=1)
        self.dec_b3 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64)
        )

        self.up2x = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.predict = nn.Conv2d(64, class_num, kernel_size=(1,1))

    def forward(self, x):
        # E
        x = self.conv1(x)
        x_b1 = self.enc_b1(x)
        x_b2 = self.enc_b2(x_b1)
        x_b3 = self.enc_b3(x_b2)

        # B
        x = self.bridge(x_b3)

        # D
        x = self.up2x(x)
        x = torch.cat([x, x_b3], dim=1)
        x = self.tran_1(x)
        x = self.dec_b1(x)

        x = self.up2x(x)
        x = torch.cat([x, x_b2], dim=1)
        x = self.tran_2(x)
        x = self.dec_b2(x)

        x = self.up2x(x)
        x = torch.cat([x, x_b1], dim=1)
        x = self.tran_3(x)
        x = self.dec_b3(x)

        x = self.predict(x)

        return x

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, ignore_index = 255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, pred, gt_tensor):
        ce_loss = torch.nn.functional.cross_entropy(pred, gt_tensor, reduction='none', ignore_index = self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        valid_mask = (gt_tensor != self.ignore_index).float()
        return focal_loss.sum() / (valid_mask.sum() + 1e-6)