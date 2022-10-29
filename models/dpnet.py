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

    def __init__(self, imgH, nm=None, leakyRelu=False, length=None):
        super(DpNet, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        self.length = length

        ks = [3, 3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0, 0]
        ss = [1, 1, 1, 1, 1, 1, 1, 1]

        if nm is None:
            nm = [64, 128, 256, 256, 512, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = 3 if i == 0 else nm[i - 1]
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
        
        # self.requires_grad_(False)
        
        # self.gap = nn.AdaptiveAvgPool2d((1, length))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(nm[-1], 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, len(cfg.chars)),
        )
        self.classifier1 = nn.Sequential(
            nn.Linear(nm[-1], 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, len(cfg.chars)),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(nm[-1], 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, len(cfg.chars)),
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(nm[-1], 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, len(cfg.chars)),
        )

    def forward(self, input):
        # conv features
        out = self.cnn(input)
        out = self.gap(out)

        # out = [c.view(c.size(0), -1) for c in out.split(1, dim=3)]
        # out = torch.cat([self.classifier(o).unsqueeze(0) for o in out], dim=0)
        
        out[0] = self.classifier(out[0])
        out[1] = self.classifier1(out[1])
        out[2] = self.classifier2(out[2])
        out[3] = self.classifier3(out[3])
        out = torch.cat([o.unsqueeze(0) for o in out], dim=0)

        # out = torch.cat([self.classifier(out.view(out.size(0), -1)).unsqueeze(0) for _ in range(self.length)], dim=0)

        # out = out.view(out.size(0), -1)
        # out_1 = self.classifier(out).unsqueeze(0)
        # out_2 = self.classifier1(out).unsqueeze(0)
        # out_3 = self.classifier2(out).unsqueeze(0)
        # out_4 = self.classifier3(out).unsqueeze(0)
        # out = torch.cat([out_1, out_2, out_3, out_4], dim=0)

        return out
