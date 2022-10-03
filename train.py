from __future__ import print_function
from loss import AMSoftmax
import logging
from models.dpnet_v3 import DpNet
from models.resnet import ResDpnet
from verification import eval_dpnet
import time
from data.transform import data_transform
from dataset import resizeNormalize, TextImageSet, randomSequentialSampler, alignCollate
from torch.utils.data import DataLoader
import utils
import numpy as np
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import torch
import collections
import random
import config as cfg
import os

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device_id
# from data.collate_batch import train_collate_fn, val_collate_fn
logging.basicConfig(level=logging.INFO)


# custom weights initialization called on DpNet
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        num = 1
        for s in list(p.size()):
            num = num * s
        pp += num
    return pp


def train_model(cfg, lrScheduler, train_loader, store_name, model, criterion, optimizer, use_gpu):
    epochs = cfg.nepoch
    batch_size = cfg.batch_size
    device = cfg.device
    best_acc = 0

    for epoch in range(epochs):

        # model train
        loss_aver = []
        model.train(True)
        start = time.time()

        for i, (data, label_pro) in enumerate(train_loader):
            if len(data) != batch_size:
                continue
            data = data.to(device) if use_gpu else data
            # predict -- > [7, batch, 35]
            predicts = model(data)
            predict_lp = predicts.split(1, 0)  # ([1, batch, 35], ....)

            loss = 0.0
            for j in range(len(predict_lp)):
                l = Variable(torch.LongTensor(
                    [el[j] for el in label_pro]).cuda(0))

                loss += criterion(predict_lp[j].squeeze(0), l)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_aver.append(loss.data)

            loss_val = sum(loss_aver) / len(loss_aver)

            if i % cfg.displayInterval == 0:
                logging.info('[%d/%d][%d/%d] Loss: %f' %
                             (epoch, cfg.nepoch, i, len(train_loader), loss_val))
        if len(loss_aver) < 1:
            continue

        logging.info('%s %s %s\n' % (epoch, sum(loss_aver) /
                                     len(loss_aver), time.time() - start))
        model.eval()
        count, correct, error, precision, avg_time = eval_dpnet(
            model, use_gpu, device, save_error=True)
        logging.info(
            '****************************** Val ********************************')
        logging.info('epoch: %s, loss: %3.3f, cost time: %s' % (
            epoch, float(sum(loss_aver) / len(loss_aver)), time.time() - start))
        logging.info('*** total %s correct %s error %s precision %s avgTime %s' %
                     (count, correct, error, precision, avg_time))
        logging.info(
            '*******************************************************************\n')

        if precision > best_acc:
            state = {
                'net': model.state_dict(),
                'acc': precision
            }
            save_path = os.path.join(
                cfg.output, "best_%.3f.pth" % (precision))
            torch.save(state, save_path)

            best_acc = precision

        lrScheduler.step()

    return model


def main():
    use_gpu = torch.cuda.is_available()
    if not os.path.isdir(cfg.output):
        os.makedirs(cfg.output)

    store_name = os.path.join(cfg.output, 'best.pth')
    # model = DpNet(cfg.input_size[0], length=cfg.text_length)
    model = ResDpnet(False, pretrained=cfg.pretrained_path,
                     length=cfg.text_length)
    if cfg.resume_file:
        if not os.path.isfile(cfg.resume_file):
            print("fail to load existed model! Existing ...")
            exit(0)
        model.load_state_dict(torch.load(cfg.resume_file)['net'])
    else:
        model.apply(weights_init)  # 初始化参数
    # if use_gpu:
    #     # 设置多卡训练
    #     model = torch.nn.DataParallel(
    #         model, device_ids=list(range(torch.cuda.device_count())))
    model = model.to(cfg.device)
    print('model params: ', get_n_params(model))
    if cfg.loss_name == "amsoftmax":
        criterion = AMSoftmax()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)

    train_transform = data_transform(True)
    train_data = TextImageSet(cfg.train_root,
                              is_train=True,
                              transform=train_transform)
    assert train_data
    train_loader = DataLoader(train_data,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=cfg.workers,
                              collate_fn=alignCollate())

    _ = train_model(cfg, lr_scheduler, train_loader, store_name,
                    model, criterion, optimizer, use_gpu)


if __name__ == '__main__':
    main()
