import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models.dpnet import DpNet
import pruning.prun_config as pcfg
import config
import collections
from verification import eval_dpnet

if not os.path.exists(pcfg.pruned_output):
    os.makedirs(pcfg.pruned_output)

model = DpNet(config.input_size[0])
model.eval()
# model.to(pcfg.device)

if pcfg.sparse_train_model:
    if os.path.isfile(pcfg.sparse_train_model):
        print("=> loading checkpoint '{}'".format(pcfg.sparse_train_model))

        org_dict = torch.load(pcfg.sparse_train_model)
        temp = collections.OrderedDict()
        for k, v in org_dict['net'].items():
            temp['.'.join(k.split('.')[1:])] = v
        model.load_state_dict(temp)

        best_prec1 = org_dict['acc']
        print("=> loaded checkpoint '{}' Prec1: {:f}"
              .format(pcfg.sparse_train_model, best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(pcfg.sparse_train_model))

print(model)

total = 0
for k, m in enumerate(model.named_modules()):
    if k < 2:
        continue
    if isinstance(m[1], nn.BatchNorm2d):
        total += m[1].weight.data.shape[0]

bn = torch.zeros(total)
index = 0
for k, m in enumerate(model.named_modules()):
    if k < 2:
        continue
    if isinstance(m[1], nn.BatchNorm2d):
        size = m[1].weight.data.shape[0]
        bn[index:(index+size)] = m[1].weight.data.abs().clone()
        index += size

y, i = torch.sort(bn)
thre_index = int(total * pcfg.PRUE_RATIO)
thre = y[thre_index]

pruned = 0
cfg = []  # BN 之前的一些 channel
cfg_mask = []

for k, m in enumerate(model.named_modules()):
    if k < 2:
        continue
    if isinstance(m[1], nn.BatchNorm2d):
        weight_copy = m[1].weight.data.abs().clone()
        mask = weight_copy.gt(thre).float()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m[1].weight.data.mul_(mask)
        m[1].bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))

pruned_ratio = pruned/total

print('Pre-processing Successful!')

# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model):
    count, correct, error, precision, avg_time = eval_dpnet(model, config.val_root, True, pcfg.device, save_error=False)

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, count, 100. * correct / count))
    return correct / float(count)

# acc = test(model)
# Make real prune
print(cfg)
newmodel = DpNet(config.input_size[0], nm=cfg)
newmodel.to(pcfg.device)

num_parameters = sum([param.nelement() for param in newmodel.parameters()])

savepath = pcfg.pruned_cfg
with open(savepath, "w") as fp:
    fp.write(",".join(map(str, cfg)) + "\n")
    fp.write(str(num_parameters) + "\n")

layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
for [m0, m1] in zip(model.named_modules(), newmodel.named_modules()):
    if isinstance(m0[1], nn.BatchNorm2d):
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))
        m1[1].weight.data = m0[1].weight.data[idx1.tolist()].clone()
        m1[1].bias.data = m0[1].bias.data[idx1.tolist()].clone()
        m1[1].running_mean = m0[1].running_mean[idx1.tolist()].clone()
        m1[1].running_var = m0[1].running_var[idx1.tolist()].clone()
        layer_id_in_cfg += 1
        start_mask = end_mask.clone()

        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0[1], nn.Conv2d):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        # 调整 weight shape
        w1 = m0[1].weight.data[:, idx0.tolist(), :, :].clone()
        w1 = w1[idx1.tolist(), :, :, :].clone()
        m1[1].weight.data = w1.clone()

    elif isinstance(m0[1], nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        m1[1].weight.data = m0[1].weight.data[:, idx0].clone()
        m1[1].bias.data = m0[1].bias.data.clone()
        break

torch.save({'cfg': cfg, 'net': newmodel.state_dict()}, pcfg.pruned_model)

print(newmodel)
model = newmodel

test(model)