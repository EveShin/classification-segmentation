import network_final as fn
import torch
import numpy as np
import os
import zipfile
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"CUDA available: {torch.cuda.is_available()}")

num_training = 150000
learning_rate = 0.1
accumulation_steps = 2
batch_size = 64
model_save_path = './model/resnet_attention'
brestore = True
restore_iter = 137000

patience = 20000
min_delta = 0.1
best_accuracy = 0.0
patience_counter = 0

if not brestore:
    restore_iter = 0

path = r'/home/kms0712w900/Desktop/project2/'
print('load zip...')
z_train = zipfile.ZipFile(path + 'train.zip', 'r')
z_train_list = z_train.namelist()
train_cls = fn.read_gt(path + 'train_gt.txt', len(z_train_list))

z_test = zipfile.ZipFile(path + 'test.zip', 'r')
z_test_list = z_test.namelist()
test_cls = fn.read_gt(path + 'test_gt.txt', len(z_test_list))

print(f"Training samples: {len(z_train_list)}")
print(f"Test samples: {len(z_test_list)}")
print(f"MEAN: {fn.MEAN}")
print(f"STD: {fn.STD}")

model = fn.ResNet().to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4, nesterov=True)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training, eta_min=1e-6)

if brestore:
    print('Model restored from file')
    model.load_state_dict(torch.load(model_save_path + '/model_%d.pt' % restore_iter))
    print(f'Restoring scheduler to iteration {restore_iter}...')
    for _ in range(restore_iter):
        scheduler.step()
    print('Scheduler restored successfully')

loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

if not os.path.isdir(model_save_path):
    os.makedirs(model_save_path)

optimizer.zero_grad()

for it in range(restore_iter, num_training + 1):
    batch_img, batch_cls = fn.Mini_batch_training_zip(z_train, z_train_list, train_cls, batch_size, augmentation=True)
    batch_img = np.transpose(batch_img, (0, 3, 1, 2))

    model.train()
    pred = model(torch.from_numpy(batch_img.astype(np.float32)).to(DEVICE))
    cls_tensor = torch.tensor(batch_cls, dtype=torch.long).to(DEVICE)

    train_loss = loss(pred, cls_tensor) / accumulation_steps
    train_loss.backward()

    if (it + 1) % accumulation_steps == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    if it % 100 == 0:
        print("it: %d   train loss: %.4f" % (it, train_loss.item()))

    if it % 500 == 0 and it != 0:
        print('Saving checkpoint...')
        torch.save(model.state_dict(), model_save_path + '/model_%d.pt' % it)

    if it % 1000 == 0 and it != 0:
        print('Test')
        model.eval()
        t1_count = 0
        t5_count = 0

        for itest in range(len(z_test_list)):
            img_temp = z_test.read(z_test_list[itest])
            img = cv2.imdecode(np.frombuffer(img_temp, np.uint8), 1)
            img = img.astype(np.float32)

            test_img = img / 255.0
            test_img = (test_img - fn.MEAN) / fn.STD
            test_img = np.reshape(test_img, [1, 128, 128, 3])
            test_img = np.transpose(test_img, (0, 3, 1, 2))

            with torch.no_grad():
                pred = model(torch.from_numpy(test_img.astype(np.float32)).to(DEVICE))

            pred = pred.cpu().numpy()
            pred = np.reshape(pred, 200)

            gt = test_cls[itest]

            for ik in range(5):
                max_index = np.argmax(pred)

                if int(gt) == int(max_index):
                    if ik == 0:
                        t1_count += 1
                    t5_count += 1
                pred[max_index] = -9999

        t1_accuracy = t1_count / float(len(z_test_list)) * 100
        t5_accuracy = t5_count / float(len(z_test_list)) * 100

        print("top-1 : %.4f%%     top-5: %.4f%%\n" % (t1_accuracy, t5_accuracy))

        if t1_accuracy > best_accuracy + min_delta:
            best_accuracy = t1_accuracy
            patience_counter = 0
            print('Best model saved (acc: %.2f%%)' % best_accuracy)
            torch.save(model.state_dict(), model_save_path + '/best_model.pt')
        else:
            patience_counter += 1000
            print('No improvement (%d/%d)' % (patience_counter, patience))

        f = open(model_save_path + '/accuracy.txt', 'a+')
        f.write("iter: %d   top-1 : %.4f     top-5: %.4f\n" % (it, t1_accuracy, t5_accuracy))
        f.close()

        if patience_counter >= patience:
            print('Early stop / best: %.2f%%' % best_accuracy)
            break

torch.save(model.state_dict(), model_save_path + '/final_model.pt')
print('Training done / best: %.2f%%' % best_accuracy)