from PIL import Image
import numpy as np
import torch
from data.transform import data_transform
from misc_functions import save_class_activation_images
import glob
from dataset import read_image, TextImageSet
from torch.utils.data import DataLoader
from models.dpnet_v2 import DpNet
import collections
import config as cfg
from torch.autograd import Variable


class CamExtractor():
    """
        Extracts cam features from the model
    """

    def __init__(self, model):
        self.model = model
        self.gradient = None
        self.gradients = []

    def save_gradient(self, grad):
        self.gradient = grad

    def save_gradients(self, grad):
        self.gradients.append(grad)

    def forward_pass_on_convolutions(self, x, nl=17):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None

        """
        3  ->  ('cnn.conv0', Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        4  ->  ('cnn.relu0', ReLU(inplace))
        5  ->  ('cnn.pooling0', MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
        6  ->  ('cnn.conv1', Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        7  ->  ('cnn.relu1', ReLU(inplace))
        8  ->  ('cnn.pooling1', MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
        9  ->  ('cnn.conv2', Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        10  ->  ('cnn.batchnorm2', BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        11  ->  ('cnn.relu2', ReLU(inplace))
        12  ->  ('cnn.conv3', Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        13  ->  ('cnn.relu3', ReLU(inplace))
        14  ->  ('cnn.pooling2', MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1), dilation=1, ceil_mode=False))
        15  ->  ('cnn.conv4', Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        16  ->  ('cnn.batchnorm4', BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        17  ->  ('cnn.relu4', ReLU(inplace))
        18  ->  ('cnn.conv5', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        19  ->  ('cnn.relu5', ReLU(inplace))
        20  ->  ('cnn.pooling3', MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1), dilation=1, ceil_mode=False))
        21  ->  ('cnn.conv6', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1)))
        22  ->  ('cnn.batchnorm6', BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        23  ->  ('cnn.relu6', ReLU(inplace))
        24  ->  ('cnn.conv7', Conv2d(512, 512, kernel_size=(2, 2), stride=(1, 1)))
        25  ->  ('cnn.batchnorm7', BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        26  ->  ('cnn.relu7', ReLU(inplace))
        """

        for idx, m in enumerate(self.model.named_modules()):
            if idx < 3:
                continue
            if idx > 26:
                continue
            # print(idx, " -> ", m)
            x = m[1](x)
            # print(x.size())

            if idx == nl:
                x.register_hook(self.save_gradient)
                conv_output = x

        x = self.model.gap(x)

        # 拆分 out 为 [Bx512, Bx512....]
        c1, c2, c3, c4, c5, c6, c7 = x.split(1, dim=3)

        c1 = c1.view(c1.size(0), -1)
        c2 = c2.view(c2.size(0), -1)
        c3 = c3.view(c3.size(0), -1)
        c4 = c4.view(c4.size(0), -1)
        c5 = c5.view(c5.size(0), -1)
        c6 = c6.view(c6.size(0), -1)
        c7 = c7.view(c7.size(0), -1)

        out1 = self.model.classifier1(c1)
        out2 = self.model.classifier2(c2)
        out3 = self.model.classifier3(c3)
        out4 = self.model.classifier4(c4)
        out5 = self.model.classifier5(c5)
        out6 = self.model.classifier6(c6)
        out7 = self.model.classifier7(c7)

        out = torch.cat([i.unsqueeze(0)
                         for i in [out1, out2, out3, out4, out5, out6, out7]], dim=0)

        return conv_output, out

    def forward_pass(self, x, nl):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x, nl)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model)

    def generate_cam(self, input_image, target_class, num_classes, nl):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(
            input_image, nl)
        # Target for backprop

        # Zero grads
        self.model.zero_grad()

        # 分离出每一位车牌号, 分别计算各自的结果梯度
        outputs = model_output.split(1, 0)  # ([1, batch, 35], ....)
        for j in range(len(outputs)):
            l = int(target_class[j])
            one_hot_output = torch.FloatTensor(1, num_classes).zero_()
            one_hot_output[0][l] = 1
            outputs[j].squeeze(0).backward(
                gradient=one_hot_output, retain_graph=True)

        # Backward pass with specified target
        # Get hooked gradients
        guided_gradients = self.extractor.gradient.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        # Take averages for each gradient
        weights = np.mean(guided_gradients, axis=(1, 2))
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) -
                                     np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[3],
                                                    input_image.shape[2]), Image.ANTIALIAS))/255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.

        # You can also use the code below instead of the code line above, suggested by @ ptschandl
        # from scipy.ndimage.interpolation import zoom
        # cam = zoom(cam, np.array(input_image[0].shape[1:])/np.array(cam.shape))
        return cam


if __name__ == '__main__':
    data_root = "visualizations/test"
    # model_path = 'models/trained/DpNet_CE_acc0.985.pth'
    model_path = 'models/trained/DpNet_AM_acc0.98361.pth'

    # Define model
    pretrained_model = DpNet(64)
    org_dict = torch.load(model_path)
    temp = collections.OrderedDict()
    for k, v in org_dict['net'].items():
        temp['.'.join(k.split('.')[1:])] = v
    pretrained_model.load_state_dict(temp)

    val_transform = data_transform(False)
    val_data = TextImageSet(data_root,
                            transform=val_transform,
                            is_train=False)
    val_loader = DataLoader(val_data, batch_size=1,
                            shuffle=False, num_workers=6)

    for i, (data, labels_pro, img_path) in enumerate(val_loader):
        if i > 10:
            break
        img_path = img_path[0]
        original_image = read_image(img_path)
        original_image = original_image.resize((192, 64))
        file_name_to_export = img_path[img_path.rfind(
            '/')+1:img_path.rfind('.')]

        # Grad cam
        grad_cam = GradCam(pretrained_model)
        # Generate cam mask single
        # cam = grad_cam.generate_cam(
        #     data, labels_pro, num_classes=len(cfg.alphabets))
        for l in range(3, 26):
            cam = grad_cam.generate_cam(data, labels_pro, num_classes=len(cfg.alphabets), nl=l)
            # Save mask
            save_class_activation_images(
                original_image, cam, "AM_"+str(l) + "_" + file_name_to_export, file_name_to_export.split('_')[0])
        print('Grad cam completed')
