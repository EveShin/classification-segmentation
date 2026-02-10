# Deep Learning Image Classification & Segmentation Projects

본 레포지토리는 이미지 분류(Classification)와 의미론적 분할(Semantic Segmentation) 태스크에 대한 딥러닝 모델의 성능을 단계적으로 최적화한 실험들을 담고 있습니다. 각 프로젝트는 기본 아키텍처에서 시작하여 고도화된 전략을 적용해 성능을 개선하는 과정을 기록하였습니다.

## 프로젝트 구성

- **프로젝트 1**: CIFAR-10 Image Classification
- **프로젝트 2**: Tiny-ImageNet Classification Challenge
- **프로젝트 3**: PASCAL VOC Semantic Segmentation

---

## 프로젝트 1: CIFAR-10 Image Classification Performance Optimization

CIFAR-10 데이터셋을 활용하여 이미지 분류 모델의 성능을 단계적으로 개선한 프로젝트입니다. 단순한 다층 퍼셉트론(MLP)에서 시작하여 적층형 CNN을 거쳐, 최종적으로 ResNet 아키텍처와 다양한 학습 전략을 도입하여 정확도를 극대화하였습니다.

### 개요

- **데이터셋**: CIFAR-10 (32x32 컬러 이미지, 10개 클래스)
- **주요 목표**: 모델 구조 고도화 및 학습 전략 최적화를 통한 분류 정확도 향상
- **환경**: PyTorch, OpenCV, NumPy

### 모델 발전 계보

#### 1단계: 다층 퍼셉트론 (MLP)
- 입력 이미지를 1차원으로 펼쳐 3개의 완전 연결 계층(Fully Connected Layer)을 통과시키는 기초 모델
- 한계: 공간 정보 유실로 인해 복잡한 이미지 패턴 학습에 제약

#### 2단계: Plain CNN
- **초기**: 2개의 Convolutional Layer와 3개의 FC Layer 사용
- **깊이 확장**: 레이어를 3~4개로 늘리고 채널 수를 최대 512까지 증설
- **안정화**: 과적합 방지를 위해 Dropout(0.3~0.5) 도입 및 FC 레이어 단순화

#### 3단계: ResNet (Residual Network)
- **ResNet-8/14/20**: Shortcut Connection을 도입하여 기울기 소실 문제 해결
- **Pre-activation (v2)**: BN → ReLU → Conv 순서로 배치 변경하여 학습 안정성 향상

### 적용된 주요 학습 전략

#### 데이터 전처리 및 증강
- **표준 정규화**: 각 채널별 MEAN과 STD를 계산하여 데이터 분포 표준화
- **증강 기법**:
  - Random Horizontal Flip & Crop: 좌우 반전 및 랜덤 크롭으로 데이터 다양성 확보
  - Cutout: 이미지 일부를 사각형으로 가려 일반화 성능 향상
  - Color Jitter: 밝기, 대비, 채도를 랜덤 조절하여 조명 변화에 강인하게 설계

#### 최적화 및 스케줄링
- **Optimizer**: SGD(Momentum 0.9, Weight Decay 5e-4) 주력 사용, Adam 비교 실험 진행
- **LR Scheduler**: 고정 학습률에서 StepLR을 거쳐 CosineAnnealingLR 적용으로 부드러운 수렴 유도

#### Early Stopping
- 검증 정확도가 일정 기간(Patience) 동안 개선되지 않을 경우 학습을 조기 종료하여 최적 가중치 보존

### 파일 구성

- `model.py`: 통합된 모든 모델 아키텍처 클래스 정의
- `utils.py`: 데이터 로딩, 전처리 및 각종 증강 함수 정의
- `main.py`: 하이퍼파라미터 설정 및 전체 학습/평가 루프 실행

---

## 프로젝트 2: Tiny-ImageNet Classification Challenge

200개의 클래스를 가진 Tiny-ImageNet 데이터셋에 대해 딥러닝 모델의 성능을 최적화하고 분류 정확도를 개선하는 프로젝트입니다. ResNet과 DenseNet을 기반으로 어텐션 메커니즘(CBAM)과 고도화된 데이터 증강 전략을 적용했습니다.

### 주요 특징

- **다양한 아키텍처 지원**: 기본 ResNet부터 DenseNet, CBAM 어텐션 모듈이 결합된 하이브리드 모델 실험 가능
- **고도화된 데이터 증강**: Color Jitter(HSV 변환 포함), Random Horizontal Flip, Random Rotation, Cutout 전략 통합
- **학습 안정화 전략**: Label Smoothing Cross Entropy 손실 함수 사용 및 Gradient Accumulation을 통한 메모리 효율적 대규모 배치 학습 구현
- **정밀한 성능 지표**: Top-1 Accuracy와 Top-5 Accuracy를 실시간으로 측정하여 모델의 예측 분포 분석

### 프로젝트 구조

프로젝트는 유지보수와 실험 효율을 위해 세 가지 핵심 모듈로 분리되어 있습니다.

- `model.py`: CNN 기반의 다양한 모델 아키텍처 정의 (ResNet, DenseNet, CBAM 모듈 등)
- `utils.py`: 데이터 로딩, Ground Truth 파싱 및 데이터 증강 유틸리티
- `main.py`: 하이퍼파라미터 설정, 학습 및 테스트 루프, Early Stopping 및 체크포인트 복구 로직

### 적용 기술 및 전략

#### 모델 아키텍처
- **ResNet + CBAM**: Residual Block 사이사이에 Channel & Spatial Attention 모듈을 배치하여 중요한 특징(Feature)에 집중
- **DenseNet**: 레이어 간의 조밀한 연결(Dense Connection)을 통해 기울기 소실 문제 완화 및 특징 재사용성 극대화

#### 학습 전략
- **Gradient Accumulation**: GPU 메모리 한계를 극복하기 위해 accumulation_steps 적용, 물리적 배치 사이즈보다 큰 유효 배치 사이즈 구현
- **Scheduler Restoration**: 학습 중단 시 체크포인트에서 모델 가중치뿐만 아니라 CosineAnnealingLR 스케줄러의 시점까지 완벽하게 복구하여 학습 연속성 유지
- **Early Stopping**: 검증 정확도가 일정 기간(patience) 동안 개선되지 않을 경우 학습을 조기 종료하여 과적합 방지

### 실행 방법

```bash
# CBAM이 적용된 ResNet 학습
python main.py --model resnet_cbam

# DenseNet 학습
python main.py --model densenet
```

### 실험 결과

학습 진행 상황 및 성능 지표는 각 모델 저장 경로의 `accuracy.txt`와 `output.txt`에 기록됩니다.

- **Top-1 Accuracy**: 모델이 가장 높은 확률로 예측한 클래스가 실제 정답인 비율
- **Top-5 Accuracy**: 모델이 예측한 상위 5개 클래스 중 실제 정답이 포함된 비율

---

## 프로젝트 3: PASCAL VOC Semantic Segmentation Performance Optimization

PASCAL VOC 2012 데이터셋을 사용하여 Semantic Segmentation 모델을 바닥부터(From Scratch) 구현하고, 다양한 아키텍처와 학습 전략을 통해 성능을 점진적으로 개선한 프로젝트입니다.

### 개요

- **목표**: 파이토치(PyTorch)를 이용해 Segmentation 모델을 직접 구현하고, mIoU 및 Pixel Accuracy 최적화
- **데이터셋**: PASCAL VOC 2012 (21개 클래스: 20개 객체 + 1개 배경)
- **주요 실험**:
  - FCN-8S vs ResNet-FCN vs ResUNet 구조 비교
  - Loss Function (CrossEntropy vs Focal Loss) 비교
  - 입력 해상도(256 vs 384) 및 데이터 증강(Augmentation) 전략 수립

### 주요 모델 아키텍처

| 모델명 | 특징 | 관련 파일 |
|--------|------|-----------|
| FCN-8S | VGG-16 기반의 Skip Connection을 활용한 고전적 FCN 구조 | `network_fcn.py` |
| DeepUNet | Conv-BN-ReLU 블록을 깊게 쌓은 수업 시간 기본 모델 | `network_school.py` |
| ResUNet | ResBlock을 사용하여 Gradient Vanishing을 방지한 고성능 모델 | `network_resunet.py` |
| ResUNet-4444 | 각 레벨당 4개의 ResBlock을 배치하여 수용 영역(Receptive Field) 확장 | `network_unet_4444.py` |

### 성능 개선 전략

#### 1. Loss Function 최적화
- **Focal Loss**: 클래스 불균형 문제 해결을 위해 배경(Background)보다 탐지하기 어려운 객체에 더 높은 가중치 부여

#### 2. 학습 파이프라인 고도화
- **2-Stage Training**: 초기 50,000회 반복까지는 전체 학습 진행, 이후 Encoder 동결(Freeze)하여 Decoder의 세부 복원 능력 극대화
- **Gradient Accumulation**: 작은 배치 사이즈의 한계를 극복하기 위해 그래디언트 누적으로 학습 안정성 확보
- **Mixed Precision**: FP16 계산을 통해 학습 속도 향상 및 메모리 효율 개선

#### 3. 데이터 증강 (Augmentation)
- Color Jitter (밝기, 대비, 채도 변화), Random Horizontal Flip, Random Scale & Crop을 적용하여 모델의 일반화 성능 개선

### 사용 방법

모든 실험 코드는 `main.py`에 통합되어 있으며, 주석 처리를 통해 전략을 선택할 수 있습니다.

1. 모델 선택: `model = ResUNet(...)` 또는 `FCN_8S()` 중 선택
2. 전략 선택: `FocalLoss`, `StepLR` 등의 주석 해제
3. 학습 실행:

```bash
python main.py
```

---

## 요구 사항

- Python 3.x
- PyTorch
- OpenCV
- NumPy
- Tensorboard (프로젝트 3)
