from __future__ import print_function
from __future__ import division

import torch
import collections
from models.dpnet_v3 import DpNet
import config as cfg


if __name__ == '__main__':
    # model_path = 'models/trained/DpNet_AM_SW_F128_acc0.98694.pth'
    model_path = 'models/logs/DpNetV3_AM_Double_acc0.93691.pth'
    # imgH, nc, nclass, rnn_node
    model = DpNet(cfg.input_size[0], length=cfg.text_length)
    model.to('cpu')
    model.eval()
    '''
    org_dict = torch.load(model_path)
    # print(org_dict)
    temp = collections.OrderedDict()
    for k, v in org_dict['net'].items():
        temp['.'.join(k.split('.')[1:])] = v
    model.load_state_dict(temp)
    # '''
    model.load_state_dict(torch.load(cfg.resume_file)['net'])

    with torch.no_grad():
        print('.....................')

        traced_script_module = torch.jit.trace(model, torch.randn(1, 3, 80, 192).to('cpu'))
        
        print('---------------------')
        # traced_script_module.save("models/jit/motor-licence-plate-recognition.pt")
        traced_script_module.save("models/jit/DpNetV3_AM_Double_acc0.93691.pt")
