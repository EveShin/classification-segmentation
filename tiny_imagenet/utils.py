import numpy as np
import cv2

MEAN = np.array([0.4000697731636383, 0.44971866014284007, 0.47799389505924583])
STD = np.array([0.2785395075911979, 0.26381946395705097, 0.2719872044484063])

def read_gt(gt_txt, num_img):
    cls = np.zeros(num_img)
    with open(gt_txt, 'r') as f:
        lines = f.readlines()
        for it in range(len(lines)):
            cls[it] = int(lines[it].strip()) - 1
    return cls

def random_horizontal_flip(img, p=0.5):
    if np.random.rand() < p:
        img = np.fliplr(img)
    return img

def random_crop(img, padding=10):
    img_padded = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')
    h, w = img_padded.shape[:2]
    top = np.random.randint(0, h - 128 + 1)
    left = np.random.randint(0, w - 128 + 1)
    return img_padded[top:top + 128, left:left + 128, :]

def random_rotation(img, max_angle=15, p=0.5):
    if np.random.rand() < p:
        angle = np.random.uniform(-max_angle, max_angle)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return img

def color_jitter(img, brightness=0.2, contrast=0.1, saturation=0.3):
    img = img.copy()
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
    return img

def cutout(img, n_holes=1, length=32, normalized=True):
    h, w = img.shape[:2]
    img_cutout = np.copy(img)
    for _ in range(n_holes):
        y, x = np.random.randint(h), np.random.randint(w)
        y1, y2 = np.clip(y - length // 2, 0, h), np.clip(y + length // 2, 0, h)
        x1, x2 = np.clip(x - length // 2, 0, w), np.clip(x + length // 2, 0, w)
        img_cutout[y1:y2, x1:x2, :] = 0.0 if normalized else 0
    return img_cutout


def Mini_batch_training_zip(z_file, z_file_list, train_cls, batch_size, augmentation=True):
    batch_img = np.zeros((batch_size, 128, 128, 3))
    batch_cls = np.zeros(batch_size)
    rand_num = np.random.randint(0, len(z_file_list), size=batch_size)

    for it in range(batch_size):
        temp = rand_num[it]
        img_temp = z_file.read(z_file_list[temp])
        img = cv2.imdecode(np.frombuffer(img_temp, np.uint8), 1).astype(np.float32)

        if augmentation:
            # Color Jitter (HSV 변환 로직 포함)
            if np.random.rand() < 0.8:
                img = np.clip(img * (1.0 + np.random.uniform(-0.2, 0.2)), 0, 255)
            if np.random.rand() < 0.8:
                hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + np.random.uniform(-0.3, 0.3)), 0, 255)
                img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

            # Flip, Rotation, Crop
            if np.random.rand() < 0.5: img = np.fliplr(img)
            if np.random.rand() < 0.5:
                M = cv2.getRotationMatrix2D((64, 64), np.random.uniform(-15, 15), 1.0)
                img = cv2.warpAffine(img, M, (128, 128), borderMode=cv2.BORDER_REFLECT)

            # Cutout (Normalized 전략)
            img = (img / 255.0 - MEAN) / STD
            y, x = np.random.randint(128), np.random.randint(128)
            y1, y2 = np.clip(y - 16, 0, 128), np.clip(y + 16, 0, 128)
            x1, x2 = np.clip(x - 16, 0, 128), np.clip(x + 16, 0, 128)
            img[y1:y2, x1:x2, :] = 0.0
        else:
            img = (img / 255.0 - MEAN) / STD

        batch_img[it] = img
        batch_cls[it] = train_cls[temp]
    return batch_img, batch_cls