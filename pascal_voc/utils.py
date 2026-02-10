import cv2
import numpy as np
import os
import torch

MEAN = np.array([103.9979, 112.8805, 116.3643])
STD = np.array([71.6989, 67.6179, 68.4911])


def load_semantic_seg_data(img_path, gt_path, size=256):
    VOC_COLORMAP = [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128], [128, 0, 0], [128, 0, 128],
                    [128, 128, 0], [128, 128, 128], [0, 0, 64], [0, 0, 192], [0, 128, 64], [0, 128, 192],
                    [128, 0, 64], [128, 0, 192], [128, 128, 64], [128, 128, 192], [0, 64, 0], [0, 64, 128],
                    [0, 192, 0], [0, 192, 128], [128, 64, 0]]

    img_names = sorted(os.listdir(img_path))
    gt_names = sorted(os.listdir(gt_path))

    imgs = [cv2.resize(cv2.imread(os.path.join(img_path, n)), (size, size)) for n in img_names]

    gts = []
    for n in gt_names:
        gt = cv2.imread(os.path.join(gt_path, n))
        gt_idx = np.zeros(gt.shape[:2], dtype=np.uint8)
        for i, color in enumerate(VOC_COLORMAP):
            gt_idx[np.all(gt == color, axis=-1)] = i
        gt_idx[np.all(gt == [192, 224, 224], axis=-1)] = 255  # Void/Border
        gts.append(cv2.resize(gt_idx, (size, size), interpolation=cv2.INTER_NEAREST))

    return np.array(imgs), np.array(gts)


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

def Mini_batch_training_Seg(train_img, train_gt, batch_size, size=256, augmentation=True, jitter_val=0.1):
    batch_img = np.zeros((batch_size, size, size, 3))
    batch_gt = np.zeros((batch_size, size, size), dtype=np.int64)
    indices = np.random.randint(0, len(train_img), batch_size)

    for i, idx in enumerate(indices):
        img, gt = train_img[idx].copy(), train_gt[idx].copy()
        if augmentation:
            img = color_jitter(img, brightness=jitter_val, contrast=jitter_val, saturation=jitter_val)
            img, gt = random_horizontal_flip(img, gt)
            img, gt = random_scale_crop(img, gt, crop_size=size)

        batch_img[i] = (img.astype(np.float32) - MEAN) / STD
        batch_gt[i] = gt
    return batch_img, batch_gt


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha, self.gamma, self.ignore_index = alpha, gamma, ignore_index

    def forward(self, pred, gt):
        ce_loss = torch.nn.functional.cross_entropy(pred, gt, reduction='none', ignore_index=self.ignore_index)
        focal_loss = self.alpha * (1 - torch.exp(-ce_loss)) ** self.gamma * ce_loss
        mask = (gt != self.ignore_index).float()
        return focal_loss.sum() / (mask.sum() + 1e-6)