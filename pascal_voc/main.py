import os
import torch
from torch.utils.tensorboard import SummaryWriter
from model import FCN_8S, ResUNet, DeepUNet
from utils import load_semantic_seg_data, Mini_batch_training_Seg, FocalLoss, MEAN, STD

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
ACCUMULATION_STEPS = 2
NUM_TRAINING = 100000
LEARNING_RATE = 0.001
IMAGE_SIZE = 256

# 경로 설정
BASE_PATH = '/content/drive/MyDrive/Colab Notebooks/project3/'
IMG_PATH = BASE_PATH + 'VOC2012/JPEGImages/'
GT_PATH = BASE_PATH + 'VOC2012/SegmentationClass/'
MODEL_SAVE_PATH = BASE_PATH + 'checkpoints/'
LOG_PATH = BASE_PATH + 'logs/'

if not os.path.exists(MODEL_SAVE_PATH): os.makedirs(MODEL_SAVE_PATH)

# --- 1. 데이터 로드 ---
print("Loading data...")
train_img, train_gt = load_semantic_seg_data(IMG_PATH, GT_PATH, size=IMAGE_SIZE)
# 테스트 데이터는 별도로 분리되어 있다고 가정하거나 전체 중 일부 사용
test_img, test_gt = train_img[-10:], train_gt[-10:]

# --- 2. 모델 및 최적화 설정 ---
# 실험하고자 하는 모델의 주석을 해제하세요.
model = ResUNet(class_num=21).to(DEVICE)
# model = FCN_8S(class_num=21).to(DEVICE)
# model = ResUNet(class_num=21, layers=[4,4,4,4], use_bn_shortcut=True).to(DEVICE) # 4444 버전

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.5)

# scaler = torch.cuda.amp.GradScaler()

criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
# criterion = FocalLoss(alpha=0.25, gamma=2) # Focal Loss 실험 시 사용

# --- 3. 학습 루프 ---
writer = SummaryWriter(LOG_PATH)
best_pixel_accuracy = 0.0
patience = 10000
patience_counter = 0

model.train()
for it in range(NUM_TRAINING):
    # if it == 50000:
    #     for param in model.enc_b1.parameters(): param.requires_grad = False
    #     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE*0.1)

    # 데이터 가져오기 (Augmentation 포함)
    batch_img, batch_gt = Mini_batch_training_Seg(train_img, train_gt, BATCH_SIZE, size=IMAGE_SIZE)

    # [전략: Mixed Precision 적용 여부]
    # with torch.cuda.amp.autocast(enabled=True):
    inputs = torch.from_numpy(batch_img.transpose(0, 3, 1, 2)).float().to(DEVICE)
    targets = torch.from_numpy(batch_gt).long().to(DEVICE)

    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss = loss / ACCUMULATION_STEPS  # Gradient Accumulation 반영

    loss.backward()

    if (it + 1) % ACCUMULATION_STEPS == 0:
        optimizer.step()
        optimizer.zero_grad()
        # if 'scheduler' in locals(): scheduler.step()

    if it % 1000 == 0:
        model.eval()
        with torch.no_grad():
            pred = torch.argmax(outputs, dim=1)
            correct = (pred == targets).sum().item()
            total = (targets != 255).sum().item()
            acc = (correct / total) * 100

            print(f"Iter: {it}, Loss: {loss.item():.4f}, Acc: {acc:.2f}%")
            writer.add_scalar('Loss/train', loss.item(), it)
            writer.add_scalar('Accuracy/train', acc, it)

            if acc > best_pixel_accuracy:
                best_pixel_accuracy = acc
                torch.save(model.state_dict(), MODEL_SAVE_PATH + 'best_model.pt')
                patience_counter = 0
            else:
                patience_counter += 1000

        model.train()

    if patience_counter >= patience:
        print("Early stopping triggered.")
        break

writer.close()