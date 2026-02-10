import os
import sys
import time
import datetime
import zipfile
import cv2
import numpy as np
import torch
import argparse

import utils as utils
import model as models

# main_final.py와 main_re.py에서 사용한 GPU 4번을 기본값으로 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Integrated Training Script for all experiments")
    parser.add_argument("--model", type=str, default="resnet_cbam",
                        choices=["resnet", "resnet_cbam", "densenet", "densenet_cbam"],
                        help="실험할 모델 아키텍처 선택")
    return parser.parse_args()


class PrintLog:
    def __init__(self, filepath, mode="a"):
        self.file = open(filepath, mode)
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self.file.write(data)

    def flush(self):
        self._stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self._stdout
        self.file.close()


def main():
    args = parse_args()

    num_training = 150000
    learning_rate = 0.1
    batch_size = 64  # ResNet 계열 기본 배치 사이즈
    accumulation_steps = 2  # main_final.py의 기울기 누적 전략

    # if "densenet" in args.model:
    #     batch_size = 32
    #     accumulation_steps = 4

    # Early Stopping 전략
    patience = 20000
    min_delta = 0.1
    best_accuracy = 0.0
    patience_counter = 0

    brestore = True
    restore_iter = 137500
    model_save_path = f'./model/{args.model}_integrated'

    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    logger = PrintLog(os.path.join(model_save_path, "output.txt"))
    sys.stdout = logger

    path = r'/home/kms0712w900/Desktop/project2/'
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Starting experiment: {args.model}")

    z_train = zipfile.ZipFile(os.path.join(path, 'train.zip'), 'r')
    z_train_list = z_train.namelist()
    train_cls = utils.read_gt(os.path.join(path, 'train_gt.txt'), len(z_train_list))

    z_test = zipfile.ZipFile(os.path.join(path, 'test.zip'), 'r')
    z_test_list = z_test.namelist()
    test_cls = utils.read_gt(os.path.join(path, 'test_gt.txt'), len(z_test_list))

    if "resnet" in args.model:
        use_cbam = "cbam" in args.model
        model = models.IntegratedResNet(outputsize=200, use_cbam=use_cbam).to(DEVICE)
    else:  # densenet
        use_cbam = "cbam" in args.model
        model = models.IntegratedDenseNet(num_classes=200, use_cbam=use_cbam).to(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training, eta_min=1e-6)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    if brestore:
        checkpoint_path = os.path.join(model_save_path, f'model_{restore_iter}.pt')
        if os.path.exists(checkpoint_path):
            print(f'Model restored from {checkpoint_path}')
            model.load_state_dict(torch.load(checkpoint_path))
            print(f'Restoring scheduler to iteration {restore_iter}...')
            for _ in range(restore_iter):
                scheduler.step()
            print('Scheduler restored successfully')
        else:
            print(f'Warning: Checkpoint not found at {checkpoint_path}. Starting from 0.')
            restore_iter = 0

    start_time = time.time()
    optimizer.zero_grad()

    for it in range(restore_iter, num_training + 1):
        batch_img, batch_label = utils.Mini_batch_training_zip(z_train, z_train_list, train_cls, batch_size,
                                                               augmentation=True)
        batch_img = np.transpose(batch_img, (0, 3, 1, 2))  # [B, H, W, C] -> [B, C, H, W]

        model.train()
        pred = model(torch.from_numpy(batch_img.astype(np.float32)).to(DEVICE))
        label_tensor = torch.tensor(batch_label, dtype=torch.long).to(DEVICE)

        train_loss = criterion(pred, label_tensor) / accumulation_steps
        train_loss.backward()

        if (it + 1) % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if it % 100 == 0:
            print(f"it: {it}   train loss: {train_loss.item() * accumulation_steps:.4f}")

        if it % 500 == 0 and it != 0:
            torch.save(model.state_dict(), os.path.join(model_save_path, f'model_{it}.pt'))

        if it % 1000 == 0 and it != 0:
            model.eval()
            t1_count, t5_count = 0, 0

            sec = (time.time() - start_time)
            result = str(datetime.timedelta(seconds=sec)).split(".")[0]
            print(f"iter {it} | Time: {result} | Testing...")

            for itest in range(len(z_test_list)):
                img_temp = z_test.read(z_test_list[itest])
                img = cv2.imdecode(np.frombuffer(img_temp, np.uint8), 1).astype(np.float32)

                test_img = (img / 255.0 - utils.MEAN) / utils.STD
                test_img = np.transpose(test_img, (2, 0, 1))[np.newaxis, ...]  # [1, C, H, W]

                with torch.no_grad():
                    pred_test = model(torch.from_numpy(test_img).to(DEVICE))

                probs = pred_test.cpu().numpy().flatten()
                gt = test_cls[itest]

                top5_indices = np.argsort(probs)[-5:][::-1]
                if int(gt) == int(top5_indices[0]): t1_count += 1
                if int(gt) in top5_indices.astype(int): t5_count += 1

            t1_acc = (t1_count / float(len(z_test_list))) * 100
            t5_acc = (t5_count / float(len(z_test_list))) * 100
            print(f"top-1 : {t1_acc:.4f}%     top-5: {t5_acc:.4f}%")

            # Early Stopping 및 Best Accuracy 업데이트
            if t1_acc > best_accuracy + min_delta:
                best_accuracy = t1_acc
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pt'))
                print(f'Best model saved (acc: {best_accuracy:.2f}%)')
            else:
                patience_counter += 1000
                print(f'No improvement ({patience_counter}/{patience})')

            with open(os.path.join(model_save_path, 'accuracy.txt'), 'a+') as f:
                f.write(f"iter: {it}   top-1 : {t1_acc:.4f}     top-5: {t5_acc:.4f}\n")

            if patience_counter >= patience:
                print(f'Early stop / best: {best_accuracy:.2f}%')
                break

            start_time = time.time()

    torch.save(model.state_dict(), os.path.join(model_save_path, 'final_model.pt'))
    logger.close()


if __name__ == "__main__":
    main()