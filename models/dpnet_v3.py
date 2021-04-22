import torch.nn as nn
import torch
import config as cfg


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class DpNet(nn.Module):

    def __init__(self, imgH, nc=3, nm=None, leakyRelu=False, length=None):
        super(DpNet, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0, 0]
        ss = [1, 1, 1, 1, 1, 1, 1, 1]

        if nm is None:
            nm = [64, 128, 256, 256, 512, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0, True)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x32x96
        convRelu(1, True)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x16x48
        convRelu(2, True)
        convRelu(3, True)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x8x24
        convRelu(4, True)
        convRelu(5, True)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x4x24
        convRelu(6, True)  # 512x2x24
        convRelu(7, True)  # 512x1x24

        self.cnn = cnn

        # self.gap = nn.AdaptiveAvgPool2d((1, length))

        self.gap1 = nn.AdaptiveAvgPool2d((1, 2))
        self.gap2 = nn.AdaptiveAvgPool2d((1, 5))


        self.classifier1 = nn.Sequential(
            nn.Linear(nm[-1], 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, len(cfg.provinces)),
        )

    def forward(self, input):
        # conv features
        out = self.cnn(input)

        # 双层车牌, 上下拼接
        outUp, outDown = out.split(1, dim=2)
        outUp = self.gap1(outUp)
        outDown = self.gap2(outDown)
        out = torch.cat([outUp, outDown], dim=3)

        # out = self.gap(out)

        out = [c.view(c.size(0), -1) for c in out.split(1, dim=3)]
        out = torch.cat([self.classifier1(o).unsqueeze(0) for o in out], dim=0)
        return out

