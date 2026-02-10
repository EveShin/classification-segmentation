import torch
import numpy as np
import cv2
import os

# 전역 변수: 표준 정규화를 위한 평균과 표준편차
MEAN = None
STD = None

def load_image(path, num_img):
    """폴더명 정렬 및 이미지 로딩 (network_1001 이후 로직 반영)"""
    imgs = np.zeros((num_img, 32, 32, 3))
    cls = np.zeros(num_img)
    cls_names = sorted(os.listdir(path))
    img_count = 0
    for ic, name in enumerate(cls_names):
        path_temp = os.path.join(path, name, '')
        img_names = os.listdir(path_temp)
        for im_name in img_names:
            if img_count >= num_img: break
            img = cv2.imread(path_temp + im_name)
            if img is not None:
                imgs[img_count] = img
                cls[img_count] = ic
                img_count += 1
    return imgs, cls

# --- 증강 기법 개별 정의 ---
def random_horizontal_flip(img):
    if np.random.rand() > 0.5: return np.fliplr(img)
    return img

def random_crop(img, padding=4):
    img_padded = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')
    h, w = img_padded.shape[:2]
    top, left = np.random.randint(0, h - 32 + 1), np.random.randint(0, w - 32 + 1)
    return img_padded[top:top + 32, left:left + 32, :]

def cutout(img, n_holes=1, length=16):
    h, w = img.shape[:2]
    img_cutout = np.copy(img)
    for _ in range(n_holes):
        y, x = np.random.randint(h), np.random.randint(w)
        y1, y2 = np.clip(y - length // 2, 0, h), np.clip(y + length // 2, 0, h)
        x1, x2 = np.clip(x - length // 2, 0, w), np.clip(x + length // 2, 0, w)
        img_cutout[y1:y2, x1:x2, :] = 0
    return img_cutout

def color_jitter(img):
    """Color Jittering (network_arg.py 반영)"""
    img_float = img.astype(np.float32)
    factors = np.random.uniform(0.8, 1.2, 3)
    img_float *= factors[0] # Brightness
    mean_gray = np.mean(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY))
    img_float = (img_float - mean_gray) * factors[1] + mean_gray # Contrast
    gray_3ch = cv2.cvtColor(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR).astype(np.float32)
    img_float = img_float * factors[2] + gray_3ch * (1 - factors[2]) # Saturation
    return np.clip(img_float, 0, 255).astype(img.dtype)

# --- 미니배치 함수 계보 보존 ---
def Mini_batch_basic(train_img, train_cls, batch_size):
    """초기 버전: Scaling만 적용 (network_0927 반영)"""
    indices = np.random.randint(0, train_img.shape[0], size=batch_size)
    batch_img = train_img[indices] / 255.0
    return batch_img, train_cls[indices]

def Mini_batch_augmented(train_img, train_cls, batch_size, augmentation=True):
    """중기 버전: Flip, Crop, Cutout 적용 (network_1011 반영)"""
    batch_img = np.zeros((batch_size, 32, 32, 3))
    batch_cls = np.zeros(batch_size)
    rand_num = np.random.randint(0, train_img.shape[0], size=batch_size)
    for it in range(batch_size):
        temp = rand_num[it]
        img = train_img[temp]
        if augmentation:
            img = random_horizontal_flip(img)
            img = random_crop(img)
            img = cutout(img)
        batch_img[it] = img / 255.0
        batch_cls[it] = train_cls[temp]
    return batch_img, batch_cls

def Mini_batch_full_jitter(train_img, train_cls, batch_size, augmentation=True):
    """최종 버전: Color Jittering 포함 모든 기법 및 정규화 (network_arg 반영)"""
    batch_img = np.zeros((batch_size, 32, 32, 3))
    batch_cls = np.zeros(batch_size)
    rand_num = np.random.randint(0, train_img.shape[0], size=batch_size)
    for it in range(batch_size):
        temp = rand_num[it]
        img = train_img[temp]
        if augmentation:
            img = random_horizontal_flip(img)
            img = random_crop(img)
            if np.random.rand() > 0.5: img = color_jitter(img)
            img = cutout(img)
        img = img / 255.0
        if MEAN is not None and STD is not None: img = (img - MEAN) / STD
        batch_img[it] = img
        batch_cls[it] = train_cls[temp]
    return batch_img, batch_cls