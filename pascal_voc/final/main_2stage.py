import os
import cv2
import network as fn
import torch
import numpy as np

VOC_COLORMAP = [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128], [128, 0, 0], [128, 0, 128],
                [128, 128, 0], [128, 128, 128], [0, 0, 64], [0, 0, 192], [0, 128, 64], [0, 128, 192],
                [128, 0, 64], [128, 0, 192], [128, 128, 64], [128, 128, 192], [0, 64, 0], [0, 64, 128],
                [0, 192, 0], [0, 192, 128], [128, 64, 0]]

MEAN = np.array([103.9979349251002, 112.88047937486992, 116.36432290207493])
STD = np.array([71.6988692265365, 67.61791552564353, 68.49109366853313])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"CUDA available: {torch.cuda.is_available()}")

batch_size = 8
accumulation_steps = 2
num_training = 50000
learning_rate = 0.01
path = '/content/drive/MyDrive/Colab Notebooks/project3/'
model_save_path = path + 'model/'
image_save_path = path + 'seg_img/'
brestore = False
restore_iter = 0

patience = 99999
min_delta = 0.0
best_pixel_accuracy = 0.0
patience_counter = 0

class_weights = torch.ones(21).to(DEVICE)
class_weights[0] = 0.1

if not brestore:
    restore_iter = 0

print('load data...')
train_img, train_gt = fn.load_semantic_seg_data(path + 'train/train_img/', path + 'train/train_gt/', size=256)
test_img, test_gt = fn.load_semantic_seg_data(path + '/test/test_img/', path + '/test/test_gt/', size=256)
print('load image finish')
print(f'Train: {train_img.shape}, Test: {test_img.shape}')

model = fn.ResUNet(class_num=21).to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4, momentum=0.9)

if brestore:
    print('Model restored from file')
    checkpoint = torch.load(model_save_path + 'checkpoint_%d.pt' % restore_iter)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_pixel_accuracy = checkpoint['best_pixel_accuracy']
    patience_counter = checkpoint['patience_counter']
    print('Restore complete')

if not os.path.isdir(model_save_path):
    os.makedirs(model_save_path)

log_file = open(model_save_path + 'training_log.txt', 'a+')

optimizer.zero_grad()

for it in range(restore_iter, num_training + 1):
    current_lr = learning_rate * ((1 - it / num_training) ** 0.9)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    batch_img, batch_gt = fn.Mini_batch_training_Seg(train_img, train_gt, batch_size, size=256, augmentation=True)
    batch_img = np.transpose(batch_img, (0, 3, 1, 2))

    model.train()
    pred = model(torch.from_numpy(batch_img.astype(np.float32)).to(DEVICE))
    gt_tensor = torch.tensor(batch_gt, dtype=torch.long).to(DEVICE)

    train_loss = torch.nn.functional.cross_entropy(
        pred,
        gt_tensor,
        weight=class_weights,
        ignore_index=255
    ) / accumulation_steps

    train_loss.backward()

    if (it + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    if it % 100 == 0:
        loss_val = train_loss.item() * accumulation_steps
        print("it: %d   train loss: %.4f   lr: %.6f" % (it, loss_val, current_lr))
        log_file.write("it: %d   train loss: %.4f   lr: %.6f\n" % (it, loss_val, current_lr))
        log_file.flush()

    if it % 1000 == 0 and it != 0:
        print('Saving checkpoint...')
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_pixel_accuracy': best_pixel_accuracy,
            'patience_counter': patience_counter
        }
        torch.save(checkpoint, model_save_path + 'checkpoint_%d.pt' % it)
        print('Checkpoint saved.')

    if it % 1000 == 0 and it != 0:
        print('Testing...')
        model.eval()

        correct_pixels = 0
        total_pixels = 0

        for itest in range(len(test_img)):
            img_temp = test_img[itest:itest + 1, :, :, :].astype(np.float32)
            img_temp = (img_temp - MEAN) / STD
            img_temp = np.transpose(img_temp, (0, 3, 1, 2))

            with torch.no_grad():
                pred = model(torch.from_numpy(img_temp.astype(np.float32)).to(DEVICE))

            pred = pred.cpu().numpy()
            pred = np.argmax(pred[0, :, :, :], axis=0)

            gt = test_gt[itest, :, :]
            valid = (gt != 255)

            correct_pixels = correct_pixels + np.sum((pred == gt) & valid)
            total_pixels = total_pixels + np.sum(valid)

        pixel_accuracy = correct_pixels / float(total_pixels) * 100

        print("Pixel Accuracy: %.2f%%" % pixel_accuracy)
        log_file.write("it: %d   pixel_acc: %.2f%%\n" % (it, pixel_accuracy))
        log_file.flush()

        if pixel_accuracy > best_pixel_accuracy + min_delta:
            best_pixel_accuracy = pixel_accuracy
            patience_counter = 0
            print('Best model saved (acc: %.2f%%)' % best_pixel_accuracy)
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_pixel_accuracy': best_pixel_accuracy,
                'patience_counter': patience_counter
            }
            torch.save(checkpoint, model_save_path + 'best_model.pt')
        else:
            patience_counter += 1000
            print('No improvement (patience: %d/%d)' % (patience_counter, patience))

        if patience_counter >= patience:
            print('Early stopping / best accuracy: %.2f%%' % best_pixel_accuracy)
            break

    if it % 5000 == 0 and it != 0:
        model.eval()

        for itest in range(min(10, len(test_img))):
            img_temp = test_img[itest:itest + 1, :, :, :].astype(np.float32)
            img_temp = (img_temp - MEAN) / STD
            img_temp = np.transpose(img_temp, (0, 3, 1, 2))

            with torch.no_grad():
                pred = model(torch.from_numpy(img_temp.astype(np.float32)).to(DEVICE))

            pred = pred.cpu().numpy()
            pred = np.argmax(pred[0, :, :, :], axis=0)

            test_save = np.zeros(shape=(256, 256, 3), dtype=np.uint8)
            for ic in range(len(VOC_COLORMAP)):
                code = VOC_COLORMAP[ic]
                test_save[np.where(pred == ic)] = code

            temp = image_save_path + '%d/' % it
            if not os.path.isdir(temp):
                os.makedirs(temp)

            cv2.imwrite(temp + '%d.png' % itest, test_save)

checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'best_pixel_accuracy': best_pixel_accuracy,
    'patience_counter': patience_counter
}
torch.save(checkpoint, model_save_path + 'final_model.pt')

log_file.close()
print('Training done / best accuracy: %.2f%%' % best_pixel_accuracy)