from __future__ import print_function
from loss import AMSoftmax
import logging
from models.dpnet import DpNet
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


def train_model(cfg, lr_scheduler, train_loader, model, criterion, optimizer, use_gpu):
    best_acc = 0

    for epoch in range(cfg.nepoch):

        # model train
        loss_aver = []
        model.train(True)
        start = time.time()
        scaler = torch.cuda.amp.GradScaler()
        for i, (data, label_pro) in enumerate(train_loader):
            with torch.cuda.amp.autocast():  # 自动混合精度 (pytorch1.6之后)
                data = data.to(cfg.device) if use_gpu else data
                # predict -- > [7, batch, 35]
                predicts = model(data)
                predict_lp = predicts.split(1, 0)  # ([1, batch, 35], ....)

                loss = torch.cuda.FloatTensor(
                    [0]) if data.is_cuda else torch.Tensor([0])
                for j in range(len(predict_lp)):
                    l = Variable(torch.LongTensor(
                        [el[j] for el in label_pro]).cuda(0))

                    loss += criterion(predict_lp[j].squeeze(0), l)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                loss_aver.append(loss.data)

                loss_val = sum(loss_aver) / len(loss_aver)

                if i % cfg.displayInterval == 0:
                    logging.info('[%d/%d][%d/%d] LR: %f - Loss: %f' %
                                 (epoch, cfg.nepoch, i, len(train_loader), optimizer.state_dict()['param_groups'][0]['lr'], loss_val))
        
        logging.info('%s %s %s\n' % (epoch, sum(loss_aver) /
                                     len(loss_aver), time.time() - start))
        model.eval()
        count, correct, error, precision, avg_time = eval_dpnet(
            model, use_gpu, cfg.device, save_error=False)
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
                'acc': precision,
                'epoch': epoch,
                # 'lr_scheduler': lr_scheduler
            }
            save_path = os.path.join(
                cfg.output, "dpnet_%dx%d_acc%.3f.pth" % (cfg.input_size[0], cfg.input_size[1], precision))
            torch.save(state, save_path)
            best_acc = precision
        lr_scheduler.step()


def main():
    use_gpu = torch.cuda.is_available()
    if not os.path.isdir(cfg.output):
        os.makedirs(cfg.output)

    model = DpNet(cfg.input_size[0], length=cfg.text_length)
    # model = ResDpnet(True, pretrained=cfg.pretrained_path,
                    #  length=cfg.text_length)
    model = model.to(cfg.device)
    print('model params: ', get_n_params(model))
    
    if cfg.resume_file:
        if not os.path.isfile(cfg.resume_file):
            print("fail to load existed model! Existing ...")
            exit(0)

        print("loading resume model...")
        checkpoint = torch.load(cfg.resume_file)
        resume_dict = checkpoint['net']
        model_dict = model.state_dict()

        # 将与 model_dict 对应的参数提取出来保存
        temp_dict = {k: v for k, v in resume_dict.items() if k in model_dict}
        # 根据 det_model_dict 的 key 更新现有的 model_dict 的值(预训练的参数值替换初始化的参数)
        model_dict.update(temp_dict)
        # 加载模型需要的参数
        model.load_state_dict(model_dict)
        # lr_scheduler = checkpoint['lr_scheduler']
    else:
        model.apply(weights_init)  # 初始化参数
    
    if cfg.loss_name == "amsoftmax":
        criterion = AMSoftmax()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters(
    )), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

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

    train_model(cfg, lr_scheduler, train_loader,
                    model, criterion, optimizer, use_gpu)


if __name__ == '__main__':
    main()
