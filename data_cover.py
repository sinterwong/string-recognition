import cv2
import torch
import time
from PIL import Image
from models.morn import MORN
from collections import OrderedDict
import torchvision.transforms as transforms
from torch.autograd import Variable


def load_morn(model_path, nc=1, targetH=32, targetW=100, inputDataType='torch.cuda.FloatTensor', 
                maxBatch=256, cuda_flag=False, driver='cuda'):

    if cuda_flag:
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location='cpu')

    morn = MORN(nc, targetH, targetW, inputDataType, maxBatch, cuda_flag)
    MORN_state_dict_rename = OrderedDict()   

    for k, v in state_dict.items():
        name = k.replace("module.", "") # remove `module.`
        MORN_state_dict_rename[name] = v

    morn.load_state_dict(MORN_state_dict_rename)
    return morn.to(driver)


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


def img_processing(image, trans, cuda_flag=True):
    image = trans(image)
    if cuda_flag:
        image = image.to('cuda')
    image = image.view(1, *image.size())
    image = Variable(image)

    return image

def main():
    image = Image.open(im_path).convert('L')
    im_transforms = resizeNormalize((32, 100))

    data = img_processing(image, im_transforms, cuda_flag=False)

    # print(img.data.type())
    morn = load_morn(morn_model_path, inputDataType='torch.FloatTensor', cuda_flag=False, driver='cpu')
    # print(morn)
    # exit()
    start = time.time()
    output = morn(data, True)
    print('morn cost time: ', time.time() - start)

    result = output.squeeze(0).squeeze(0).mul_(0.5).add_(0.5).detach().numpy() * 255

    cv2.imwrite('data/morned.jpg', result)


if __name__ == "__main__":
    morn_model_path = 'data/morn.pth'
    im_path = 'data/demo02.jpg'

    main()

