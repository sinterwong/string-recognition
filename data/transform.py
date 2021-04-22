import torchvision.transforms as T
import config as cfg


def data_transform(is_train=True):
    normalize_transform = T.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)
    if is_train:
        transform = T.Compose([
            T.Resize(cfg.input_size),
            T.ColorJitter(brightness=cfg.bright_prob, saturation=cfg.saturation_prob,
                          contrast=cfg.contrast_prob, hue=cfg.hue_prob),
            T.RandomGrayscale(cfg.grayScale_prob),
            T.Pad(cfg.pad),
            T.RandomCrop(cfg.input_size),
            T.ToTensor(),
            normalize_transform,
        ])

        return transform

    else:
        transform = T.Compose([
            T.Resize(cfg.input_size),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
