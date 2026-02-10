import torch
import numpy as np
import cv2
import os

MEAN = None
STD = None

# 학습에 이용할 사진을 불러와서 RAM에 쌓기(랜덤 샘플 x)
def load_image(path, num_img):
    # color img 일 경우 : img = np.zeros((num_img, height, width, 3))
    imgs = np.zeros((num_img, 32, 32, 3))  # 32x32 컬러이미지
    cls = np.zeros(num_img)

    cls_names = sorted(os.listdir(path))  # 정렬
    img_count = 0

    # 사진 불러오기
    for ic in range(len(cls_names)):  # len(cls_name) : 클래스 개수
        path_temp = path + cls_names[ic] + '/'
        print(path_temp)
        img_names = os.listdir(path_temp)

        for im in range(len(img_names)):  # 파일 개수
            img = cv2.imread(path_temp + img_names[im])
            imgs[img_count, :, :, :] = img  # RGB 고려
            cls[img_count] = ic
            img_count = img_count + 1
    return imgs, cls

# Data Augmentation: Random Horizontal Flip
def random_horizontal_flip(img):
    if np.random.rand() > 0.5:  # 50% 확률로 좌우 반전
        img = np.fliplr(img)
    return img

# Data Augmentation: Random Crop
def random_crop(img, padding=4):
    # 이미지에 패딩 추가 후 원본 크기로 크롭
    img_padded = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')
    h, w = img_padded.shape[:2]
    top = np.random.randint(0, h - 32 + 1)
    left = np.random.randint(0, w - 32 + 1)
    img_cropped = img_padded[top:top + 32, left:left + 32, :]
    return img_cropped

# Data Augmentation: Cutout
def cutout(img, n_holes=1, length=16):
    h = img.shape[0]
    w = img.shape[1]
    img_cutout = np.copy(img)
    for _ in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)
        img_cutout[y1:y2, x1:x2, :] = 0
    return img_cutout

def Mini_batch_training(train_img, train_cls, batch_size, augmentation=True):
    batch_img = np.zeros((batch_size, 32, 32, 3)) # 미니배치 이미지 배열
    batch_cls = np.zeros(batch_size) # 미니배치 레이블 배열

    rand_num = np.random.randint(0, train_img.shape[0], size=batch_size)  # 랜덤으로 batch size 만큼의 인덱스 선택

    for it in range(batch_size):
        temp = rand_num[it]
        img = train_img[temp, :, :, :]

        if augmentation:  # 데이터 증강 적용
            img = random_horizontal_flip(img)
            img = random_crop(img, padding=4)
            img = cutout(img, n_holes=1, length=16)

        img = img / 255.0 # 픽셀 값 0~1 사이로 정규화

        if MEAN is not None and STD is not None:  # 표준 정규화
            img = (img - MEAN) / STD

        batch_img[it, :, :, :] = img
        batch_cls[it] = train_cls[temp]

    return batch_img, batch_cls


# 데이터셋 구분하는 CNN (ResNet 기반) Resnet 32 pre-activation 구조
class CNN(torch.nn.Module):
    def __init__(self, outputsize=10):
        super(CNN, self).__init__()

        # 초기 특징 추출 레이어 (3 -> 16 채널)
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)


        # Residual Block 1 ~ 5 (16채널 유지)
        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.conv3 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(16)

        self.conv4 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.bn4 = torch.nn.BatchNorm2d(16)
        self.conv5 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.bn5 = torch.nn.BatchNorm2d(16)

        self.conv6 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.bn6 = torch.nn.BatchNorm2d(16)
        self.conv7 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.bn7 = torch.nn.BatchNorm2d(16)

        self.conv8 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.bn8 = torch.nn.BatchNorm2d(16)
        self.conv9 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.bn9 = torch.nn.BatchNorm2d(16)

        self.conv10 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.bn10 = torch.nn.BatchNorm2d(16)
        self.conv11 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.bn11 = torch.nn.BatchNorm2d(16)

        # Downsampling Conv (16 -> 32 채널) 및 Residual Block 6 ~ 10 (32채널 유지)
        self.conv12 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn12 = torch.nn.BatchNorm2d(16)
        self.conv13 = torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn13 = torch.nn.BatchNorm2d(32)

        self.conv14 = torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn14 = torch.nn.BatchNorm2d(32)
        self.conv15 = torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn15 = torch.nn.BatchNorm2d(32)

        self.conv16 = torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn16 = torch.nn.BatchNorm2d(32)
        self.conv17 = torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn17 = torch.nn.BatchNorm2d(32)

        self.conv18 = torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn18 = torch.nn.BatchNorm2d(32)
        self.conv19 = torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn19 = torch.nn.BatchNorm2d(32)

        self.conv20 = torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn20 = torch.nn.BatchNorm2d(32)
        self.conv21 = torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn21 = torch.nn.BatchNorm2d(32)

        # Downsampling Conv (32 -> 64 채널) 및 Residual Block 11 ~ 15 (64채널 유지)
        self.conv22 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn22 = torch.nn.BatchNorm2d(32)
        self.conv23 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn23 = torch.nn.BatchNorm2d(64)

        self.conv24 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn24 = torch.nn.BatchNorm2d(64)
        self.conv25 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn25 = torch.nn.BatchNorm2d(64)

        self.conv26 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn26 = torch.nn.BatchNorm2d(64)
        self.conv27 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn27 = torch.nn.BatchNorm2d(64)

        self.conv28 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn28 = torch.nn.BatchNorm2d(64)
        self.conv29 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn29 = torch.nn.BatchNorm2d(64)

        self.conv30 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn30 = torch.nn.BatchNorm2d(64)
        self.conv31 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn31 = torch.nn.BatchNorm2d(64)

        # 크기와 채널이 변하는 identity mapping을 위한 shortcut connection
        self.shortcut1 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
            torch.nn.BatchNorm2d(32)
        )
        self.shortcut2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
            torch.nn.BatchNorm2d(64)
        )

        # 최종 분류를 위한 Fully Connected Layer
        self.bn32 = torch.nn.BatchNorm2d(64)
        self.fc1 = torch.nn.Linear(64, outputsize)

        # 활성화 함수 및 풀링
        self.ReLU = torch.nn.ReLU()
        self.AvgPool = torch.nn.AvgPool2d(kernel_size=8)

    def forward(self, x):
        # Initial Convolution
        x = self.conv1(x)

        # [128, 3, 32, 32] -> [128, 16, 32, 32]

        # Residual Block 1
        identity = x
        out = self.bn2(x)
        out = self.ReLU(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.ReLU(out)
        out = self.conv3(out)
        x = out + identity
        # [128, 16, 32, 32] -> [128, 16, 32, 32]

        # Residual Block 2
        identity = x
        out = self.bn4(x)
        out = self.ReLU(out)
        out = self.conv4(out)
        out = self.bn5(out)
        out = self.ReLU(out)
        out = self.conv5(out)
        x = out + identity
        # [128, 16, 32, 32] -> [128, 16, 32, 32]

        # Residual Block 3
        identity = x
        out = self.bn6(x)
        out = self.ReLU(out)
        out = self.conv6(out)
        out = self.bn7(out)
        out = self.ReLU(out)
        out = self.conv7(out)
        x = out + identity
        # [128, 16, 32, 32] -> [128, 16, 32, 32]

        # Residual Block 4
        identity = x
        out = self.bn8(x)
        out = self.ReLU(out)
        out = self.conv8(out)
        out = self.bn9(out)
        out = self.ReLU(out)
        out = self.conv9(out)
        x = out + identity

        # [128, 16, 32, 32] -> [128, 16, 32, 32]

        # Residual Block 5
        identity = x
        out = self.bn10(x)
        out = self.ReLU(out)
        out = self.conv10(out)
        out = self.bn11(out)
        out = self.ReLU(out)
        out = self.conv11(out)
        x = out + identity
        # [128, 16, 32, 32] -> [128, 16, 32, 32]

        # Downsampling Block 1
        identity = self.shortcut1(x)
        out = self.bn12(x)
        out = self.ReLU(out)
        out = self.conv12(out)
        out = self.bn13(out)
        out = self.ReLU(out)
        out = self.conv13(out)
        x = out + identity

        # [128, 16, 32, 32] -> [128, 32, 16, 16]

        # Residual Block 7
        identity = x
        out = self.bn14(x)
        out = self.ReLU(out)
        out = self.conv14(out)
        out = self.bn15(out)
        out = self.ReLU(out)
        out = self.conv15(out)
        x = out + identity
        # [128, 32, 16, 16] -> [128, 32, 16, 16]

        # Residual Block 8
        identity = x
        out = self.bn16(x)
        out = self.ReLU(out)
        out = self.conv16(out)
        out = self.bn17(out)
        out = self.ReLU(out)
        out = self.conv17(out)
        x = out + identity
        # [128, 32, 16, 16] -> [128, 32, 16, 16]

        # Residual Block 9
        identity = x
        out = self.bn18(x)
        out = self.ReLU(out)
        out = self.conv18(out)
        out = self.bn19(out)
        out = self.ReLU(out)
        out = self.conv19(out)
        x = out + identity
        # [128, 32, 16, 16] -> [128, 32, 16, 16]

        # Residual Block 10
        identity = x
        out = self.bn20(x)
        out = self.ReLU(out)
        out = self.conv20(out)
        out = self.bn21(out)
        out = self.ReLU(out)
        out = self.conv21(out)
        x = out + identity
        # [128, 32, 16, 16] -> [128, 32, 16, 16]

        # Downsampling Block 2
        identity = self.shortcut2(x)
        out = self.bn22(x)
        out = self.ReLU(out)
        out = self.conv22(out)
        out = self.bn23(out)
        out = self.ReLU(out)
        out = self.conv23(out)
        x = out + identity
        # [128, 32, 16, 16] -> [128, 64, 8, 8]

        # Residual Block 12
        identity = x
        out = self.bn24(x)
        out = self.ReLU(out)
        out = self.conv24(out)
        out = self.bn25(out)
        out = self.ReLU(out)
        out = self.conv25(out)
        x = out + identity
        # [128, 64, 8, 8] -> [128, 64, 8, 8]

        # Residual Block 13
        identity = x
        out = self.bn26(x)
        out = self.ReLU(out)
        out = self.conv26(out)
        out = self.bn27(out)
        out = self.ReLU(out)
        out = self.conv27(out)
        x = out + identity
        # [128, 64, 8, 8] -> [128, 64, 8, 8]

        # Residual Block 14
        identity = x
        out = self.bn28(x)
        out = self.ReLU(out)
        out = self.conv28(out)
        out = self.bn29(out)
        out = self.ReLU(out)
        out = self.conv29(out)
        x = out + identity
        # [128, 64, 8, 8] -> [128, 64, 8, 8]

        # Residual Block 15
        identity = x
        out = self.bn30(x)
        out = self.ReLU(out)
        out = self.conv30(out)
        out = self.bn31(out)
        out = self.ReLU(out)
        out = self.conv31(out)
        x = out + identity
        # [128, 64, 8, 8] -> [128, 64, 8, 8]

        # Final Layers
        x = self.bn32(x)
        x = self.ReLU(x)
        x = self.AvgPool(x)
        # [128, 64, 8, 8] -> [128, 64, 1, 1]
        x = torch.reshape(x, [-1, 64])
        # [128, 64, 1, 1] -> [128, 64]
        x = self.fc1(x)
        # [128, 64] -> [128, 10]

        return x