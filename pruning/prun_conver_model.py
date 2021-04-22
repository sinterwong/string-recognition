from __future__ import print_function
from __future__ import division

import torch
import collections
from models.dpnet_v3 import DpNet
import prun_config as pcfg
import config as cfg


if __name__ == '__main__':
    model_path = 'models/pruned/DpNetV3_SR_Pruned_Acc0.98018.pth'
    # imgH, nc, nclass, rnn_node
    with open(pcfg.pruned_cfg) as rf:
        nm = rf.readlines()[0]
    model = DpNet(cfg.input_size[0], nm=list(map(int, nm.split(","))))
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["net"])
    with torch.no_grad():
        print('.....................')
        traced_script_module = torch.jit.trace(model, torch.randn(1, 3, 64, 192).to('cpu'))
        print('---------------------')
        # traced_script_module.save("models/jit/motor-licence-plate-recognition.pt")
        traced_script_module.save("models/pruned/DpNetV3_SR_Pruned_Acc0.98018.pt")
