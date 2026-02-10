# 스케줄러 변경

import network as fn
import torch
import numpy as np
import os

# user parameter
num_training = 40000
learning_rate = 0.1
model_save_path = './model/'
brestore = False
restore_iter = 0

# Early stopping parameters
patience = 15000  # 성능 개선이 없을 때 기다릴 iteration 수
min_delta = 0.1  # 개선으로 인정할 최소 정확도 차이
best_accuracy = 0.0  # 최고 정확도 저장
patience_counter = 0  # patience 카운터

if not brestore:
    restore_iter = 0

# data load
train_path = './train/'
test_path = './test/'
train_images, train_cls = fn.load_image(train_path, 50000)
test_images, test_cls = fn.load_image(test_path, 1000)

# 표준 정규화 계산
train_normalized = train_images / 255.0
MEAN = np.mean(train_normalized, axis=(0, 1, 2))
STD = np.std(train_normalized, axis=(0, 1, 2))

print(f"MEAN: {MEAN}")
print(f"STD: {STD}")

fn.MEAN = MEAN
fn.STD = STD

print("훈련 데이터 shape:", train_images.shape)
print("테스트 데이터 shape:", test_images.shape)
print("훈련 라벨 분포:", np.bincount(train_cls.astype(int)))
print("테스트 라벨 분포:", np.bincount(test_cls.astype(int)))

# 실제 이미지 값 범위 확인
print("이미지 값 범위:", train_images.min(), train_images.max())

model = fn.CNN()

if brestore:
    print('Model restored from file')
    model.load_state_dict(torch.load(model_save_path + 'model_%d.pt' % restore_iter))
    model.eval()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4) # 1e-4 -> 5e-4
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training) # CosineAnnealingLR 추가

loss = torch.nn.CrossEntropyLoss()

if not os.path.isdir(model_save_path):
    os.makedirs(model_save_path)

for it in range(restore_iter, num_training + 1):
    # 스케줄러를 사용하므로 수동 학습률 조절 부분은 삭제
    batch_img, batch_cls = fn.Mini_batch_training(train_images, train_cls, 128, augmentation=True)
    batch_img = np.transpose(batch_img, (0, 3, 1, 2))

    model.train()
    optimizer.zero_grad()
    pred = model(torch.from_numpy(batch_img.astype(np.float32)))
    cls_tensor = torch.tensor(batch_cls, dtype=torch.long)

    train_loss = loss(pred, cls_tensor)
    train_loss.backward()
    optimizer.step()

    # --- [추가된 부분]: 매 스텝마다 스케줄러를 업데이트하여 학습률 조절
    scheduler.step()
    # --- [여기까지 추가] ---

    if it % 100 == 0:
        print("it: %d   train loss: %f" % (it, train_loss.item()))

    if it % 1000 == 0:
        print('Test')
        model.eval()
        count = 0
        for itest in range(len(test_images)):
            test_img = test_images[itest:itest + 1, :, :, :] / 255.0
            test_img = (test_img - fn.MEAN) / fn.STD
            test_img = np.transpose(test_img, (0, 3, 1, 2))
            with torch.no_grad():
                pred = model(torch.from_numpy(test_img.astype(np.float32)))
            pred = pred.numpy()
            pred = np.reshape(pred, 10)
            pred = np.argmax(pred)
            gt = test_cls[itest]
            if int(gt) == int(pred):
                count = count + 1
        current_accuracy = (count / len(test_images) * 100)
        print('Accuracy : %f ' % current_accuracy)

        # Early stopping 로직
        if current_accuracy > best_accuracy + min_delta:
            best_accuracy = current_accuracy
            patience_counter = 0  # 초기화
            print('Best model saved (acc: %.2f%%)' % best_accuracy)
            torch.save(model.state_dict(), model_save_path + 'best_model.pt')
        else:
            patience_counter += 1000  # 성능 개선이 없으면 카운터 증가
            print('No improvement (%d/%d)' % (patience_counter, patience))

        if patience_counter >= patience:
            print('Early stop / best: %.2f%%' % best_accuracy)
            break

torch.save(model.state_dict(), model_save_path + 'final_model.pt')
print('Training done / best: %.2f%%' % best_accuracy)