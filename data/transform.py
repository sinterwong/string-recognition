import torchvision.transforms as T
import config as cfg
import numpy as np
import random
from PIL import Image


# 自定义添加椒盐噪声的 transform
class AddPepperNoise(object):
    """增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) or (isinstance(p, float))
        self.snr = snr
        self.p = p

    # transform 会调用该方法
    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        # 如果随机概率小于 seld.p，则执行 transform
        if random.uniform(0, 1) < self.p:
            # 把 image 转为 array
            img_ = np.array(img).copy()
            # 获得 shape
            h, w, c = img_.shape
            # 信噪比
            signal_pct = self.snr
            # 椒盐噪声的比例 = 1 -信噪比
            noise_pct = (1 - self.snr)
            # 选择的值为 (0, 1, 2)，每个取值的概率分别为 [signal_pct, noise_pct/2., noise_pct/2.]
            # 椒噪声和盐噪声分别占 noise_pct 的一半
            # 1 为盐噪声，2 为 椒噪声
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[
                                    signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = np.random.randint(200, 255)   # 盐噪声
            img_[mask == 2] = np.random.randint(0, 60)     # 椒噪声
            # 再转换为 image
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        # 如果随机概率大于 seld.p，则直接返回原图
        else:
            return img


def data_transform(is_train=True):
    # normalize_transform = T.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)
    if is_train:
        transform = T.Compose([
            T.ColorJitter(brightness=cfg.bright_prob, saturation=cfg.saturation_prob,
                          contrast=cfg.contrast_prob, hue=cfg.hue_prob),
            T.RandomGrayscale(cfg.grayScale_prob),
            AddPepperNoise(0.9, p=0.2),
            # T.RandomAffine(15),
            # T.GaussianBlur((3, 3), sigma=(0.1, 2.0)),
            T.Resize(cfg.input_size),
            T.Pad(cfg.pad),
            # T.RandomCrop(cfg.input_size),
            T.ToTensor(),
            # normalize_transform,
        ])

        return transform

    else:
        transform = T.Compose([
            T.Resize(cfg.input_size),
            T.ToTensor(),
            # normalize_transform
        ])

    return transform
