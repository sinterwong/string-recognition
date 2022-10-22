import time
import torch
from torch.utils.data import DataLoader
from data.transform import data_transform
from dataset import resizeNormalize, TextImageSet, randomSequentialSampler, alignCollate
import config as cfg
import shutil
import os


def is_equal(label_gt, label_p):
    compare = [1 if int(label_gt[i]) == int(
        label_p[i]) else 0 for i in range(cfg.text_length)]
    return sum(compare)


def eval_dpnet(model, use_gpu, device, save_error=False, save_error_dir='work_dirs/valError_AM'):
    model.to(device)
    count, error, correct = 0, 0, 0
    val_transform = data_transform(False)
    val_data = TextImageSet(cfg.val_root,
                            transform=val_transform,
                            is_train=False)
    val_loader = DataLoader(val_data, batch_size=1,
                            shuffle=False, num_workers=6)
    start = time.time()
    with torch.no_grad():
        for i, (data, labels_pro, img_p) in enumerate(val_loader):
            count += 1
            if use_gpu:
                data = data.to(device)

            # Forward pass: Compute predicted y by passing x to the model
            label_predict = model(data)

            label_predict = label_predict.split(1, 0)  # ([1, batch, 35], ....)

            output = [el.squeeze(0).data.cpu().numpy().tolist()
                      for el in label_predict]
            # print(output[0])
            # print(len(output[0]))
            # print(len(output[0][0]))

            predict_label = [t[0].index(max(t[0])) for t in output]

            label = [int(el.numpy()) for el in labels_pro]
            #   compare YI, outputY
            if is_equal(predict_label, label) == cfg.text_length:
                correct += 1
            else:
                error += 1
                if save_error:
                    pre_result = ""
                    if not os.path.exists(save_error_dir):
                        os.makedirs(save_error_dir)

                    for p in range(len(predict_label)):
                        pre_result += cfg.chars[predict_label[p]]

                    img_name = pre_result + "_" + os.path.basename(img_p[0])

                    shutil.copy(img_p[0], os.path.join(
                        save_error_dir, img_name))

    return count, correct, error, float(correct) / count, (time.time() - start) / count
