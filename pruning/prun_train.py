from __future__ import print_function
from loss import AMSoftmax
import logging
from models.dpnet_v3 import DpNet
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
import prun_config as pcfg
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


def train_model(cfg, lrScheduler, train_loader, model, criterion, optimizer, use_gpu):

    def updateBN():
        for name, m in model.named_modules():
            if name in ['bn1', 'module.bn1'] or ('downsample' in name):
                continue
            if isinstance(m, nn.BatchNorm2d):
                m.weight.grad.data.add_(pcfg.SCALE_SPARSE_RATE*torch.sign(m.weight.data))

    epochs = cfg.nepoch
    batch_size = cfg.batch_size
    device = cfg.device
    loss_lambda = 1.2

    best_acc = 0

    for epoch in range(epochs):
        loss_aver = []
        model.train(True)
        lrScheduler.step()
        start = time.time()

        # r'''
        for i, (data, label_pro) in enumerate(train_loader):
            if len(data) != batch_size:
                # print('num data: ', len(data))
                continue
            data = data.to(device) if use_gpu else data
            predicts = model(data)
            # print(predicts.size())
            predict_lp = predicts.split(1, 0)  # ([1, batch, 35], ....)

            loss = 0.0

            for j in range(len(predict_lp)):
                l = Variable(torch.LongTensor(
                    [el[j] for el in label_pro]).cuda(0))

                """ 特别关注第一位
                if j == 0:
                    loss += (loss_lambda * criterion(predict_lp[j].squeeze(0), l))
                else:
                    loss += criterion(predict_lp[j].squeeze(0), l)
                """

                loss += criterion(predict_lp[j].squeeze(0), l)

            optimizer.zero_grad()
            loss.backward()
            if pcfg.SPARSE:
                updateBN()
            optimizer.step()

            loss_aver.append(loss.data)

            loss_val = sum(loss_aver) / len(loss_aver)

            if i % 50 == 0:
                logging.info('[%d/%d][%d/%d] Loss: %f' %
                             (epoch, cfg.nepoch, i, len(train_loader), loss_val))
        if len(loss_aver) < 1:
            continue

        logging.info('%s %s %s\n' % (epoch, sum(loss_aver) /
                                     len(loss_aver), time.time() - start))
        model.eval()
        count, correct, error, precision, avg_time = eval_dpnet(
            model, cfg.val_root, use_gpu, device, save_error=False)
        logging.info(
            '****************************** Val ********************************')
        logging.info('epoch: %s, loss: %3.3f, cost time: %s' % (
            epoch, float(sum(loss_aver) / len(loss_aver)), time.time() - start))
        logging.info('*** total %s error %s precision %s avgTime %s' %
                     (count, error, precision, avg_time))
        logging.info(
            '*******************************************************************\n')

        if precision > best_acc:
            state = {
                'net': model.state_dict(),
                'acc': precision
            }
            save_path = os.path.join(pcfg.pruned_output, "DpNetV3_SR_Acc%.5f.pth" % (precision))
            torch.save(state, save_path)

            best_acc = precision
        # '''

        """
        model.eval()
        count, correct, error, precision, avg_time = eval_dpnet(
            model, cfg.val_root, use_gpu, device, save_error=True)
        logging.info(
            '****************************** Val ********************************')
        logging.info('*** total %s error %s precision %s avgTime %s' %
                     (count, error, precision, avg_time))
        logging.info(
            '*******************************************************************\n')
        exit()
        """

    return model


def main():
    if not os.path.isdir(pcfg.pruned_output):
        os.makedirs(pcfg.pruned_output)
    if pcfg.FINETUNE:
        with open(pcfg.pruned_cfg) as rf:
            nm = rf.readlines()[0]
        checkpoint = torch.load(pcfg.pruned_model)
        model = DpNet(cfg.input_size[0], nm=list(map(int, nm.split(","))), length=cfg.text_length)
        model.load_state_dict(checkpoint["net"])
    else:
        model = DpNet(cfg.input_size[0], length=cfg.text_length)

    model = model.to(cfg.device)
    """
    org_dict = torch.load(pcfg.pruned_model)
    temp = collections.OrderedDict()
    for k, v in org_dict['state_dict'].items():
        temp['.'.join(k.split('.')[1:])] = v
    model.load_state_dict(temp)
    """
    
    print('model params: ', get_n_params(model))
    # criterion = nn.CrossEntropyLoss()
    criterion = AMSoftmax()
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

    _ = train_model(cfg, lr_scheduler, train_loader, model, criterion, optimizer, True)


if __name__ == '__main__':
    main()
