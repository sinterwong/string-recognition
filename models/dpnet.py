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

    def __init__(self, imgH, nc=3, leakyRelu=False):
        super(DpNet, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0, 0]
        ss = [1, 1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512, 512]

        self.gap = nn.AdaptiveAvgPool2d((1, 7))

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

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x32x96
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x16x48
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x8x24
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x4x24
        convRelu(6, True)  # 512x2x24
        convRelu(7, True)  # 512x1x24

        self.cnn = cnn


        self.classifier1 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(3584, 64),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(64, len(cfg.chars)),
        )

        self.classifier2 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(3584, 64),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(64, len(cfg.alphabets)),
        )

        self.classifier3 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(3584, 64),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(64, len(cfg.alphabets)),
        )

        self.classifier4 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(3584, 64),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(64, len(cfg.alphabets)),
        )

        self.classifier5 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(3584, 64),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(64, len(cfg.alphabets)),
        )

        self.classifier6 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(3584, 64),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(64, len(cfg.alphabets)),
        )

        self.classifier7 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(3584, 64),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(64, len(cfg.alphabets)),
        )

    def forward(self, input):
        # conv features
        out = self.cnn(input)
        out = self.gap(out)

        b, c, h, w = out.size()
        assert h == 1, "the height of conv must be 1"

        out = out.view(out.size(0), -1)

        out1 = self.classifier1(out)
        out2 = self.classifier2(out)
        out3 = self.classifier3(out)
        out4 = self.classifier4(out)
        out5 = self.classifier5(out)
        out6 = self.classifier6(out)
        out7 = self.classifier7(out)

        # print(out1.size())
        # print(out2.size())
        # print(out3.size())
        # print(out4.size())
        # print(out5.size())
        # print(out6.size())
        # print(out7.size())

        out = torch.cat([i.unsqueeze(0) for i in [out1, out2, out3, out4, out5, out6, out7]], dim=0)
        return out


if __name__ == '__main__':
    import time
    dpnet = DpNet(64)

    data = torch.rand((32, 3, 64, 96))

    torch.save(dpnet.state_dict(), 'Test.pth')

    # start = time.time()
    dpnet(data)
    # print(time.time() - start)