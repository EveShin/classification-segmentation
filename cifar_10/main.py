import torch
import numpy as np
import os
import model as models
import utils as fn

num_training = 64000
batch_size = 128
learning_rate = 0.1
model_save_path = './model_final/'
train_path = './train/'
test_path = './test/'

# 복구 및 Early Stopping 설정 (main_1005_resnet 이후 표준)
brestore = False
restore_iter = 0
patience = 7000
min_delta = 0.1
best_accuracy = 0.0
patience_counter = 0

if not os.path.isdir(model_save_path):
    os.makedirs(model_save_path)

# --- [데이터 로드 및 정규화] ---
train_images, train_cls = fn.load_image(train_path, 50000)
test_images, test_cls = fn.load_image(test_path, 1000)

train_normalized = train_images / 255.0
MEAN = np.mean(train_normalized, axis=(0, 1, 2))
STD = np.std(train_normalized, axis=(0, 1, 2))
fn.MEAN, fn.STD = MEAN, STD

# model = models.NeuralNet(3072, 512, 256, 10) # 초기 MLP 전략 (0927)
# model = models.PlainCNN(outputsize=10, version='deep_fc') # 깊은 FC 전략 (1001_fc)
model = models.ResNetManual(outputsize=10)  # 표준 ResNet 전략 (1012)
# model = models.PreActResNet(outputsize=10) # Pre-activation 전략 (v2)

if brestore:
    model.load_state_dict(torch.load(os.path.join(model_save_path, f'model_{restore_iter}.pt')))

# A. SGD (ResNet 계열 표준 - Momentum + Weight Decay)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# B. Adam (0929 버전에서 시도됨)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# C. 스케줄러 선택
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1) # Step 하락 전략
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training)  # Cosine 곡선 전략

for it in range(restore_iter, num_training + 1):

    # 미니배치 증강 선택
    # batch_img, batch_cls = fn.Mini_batch_basic(train_images, train_cls, batch_size) # 단순 Scaling (0927)
    # batch_img, batch_cls = fn.Mini_batch_augmented(train_images, train_cls, batch_size) # Flip+Crop+Cutout (1011)
    batch_img, batch_cls = fn.Mini_batch_full_jitter(train_images, train_cls, batch_size)  # +Color Jitter (arg)

    batch_img = np.transpose(batch_img, (0, 3, 1, 2))
    model.train()
    optimizer.zero_grad()

    pred = model(torch.from_numpy(batch_img.astype(np.float32)))
    loss = criterion(pred, torch.tensor(batch_cls, dtype=torch.long))

    loss.backward()
    optimizer.step()
    if 'scheduler' in locals(): scheduler.step()

    if it % 1000 == 0:
        model.eval()
        correct = 0
        with torch.no_grad():
            for itest in range(len(test_images)):
                t_img = np.transpose((test_images[itest:itest + 1] / 255.0 - fn.MEAN) / fn.STD, (0, 3, 1, 2))
                output = model(torch.from_numpy(t_img.astype(np.float32)))
                if torch.argmax(output).item() == int(test_cls[itest]): correct += 1

        accuracy = (correct / len(test_images)) * 100
        print(f"it: {it} | Accuracy: {accuracy:.2f}%")

        # Early Stopping 전략 (main_1010 이후 도입)
        if accuracy > best_accuracy + min_delta:
            best_accuracy = accuracy
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pt'))
        else:
            patience_counter += 1000
            if patience_counter >= patience:
                print(f"Early Stop at {it}");
                break

torch.save(model.state_dict(), os.path.join(model_save_path, 'final_model.pt'))