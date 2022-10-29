import os.path as osp

''' data iter setting '''
# train_root = '/home/wangjq/license_plate_dataset/licence-plate/train'
# val_root = '/home/wangjq/license_plate_dataset/licence-plate/val'
train_root = 'datasets/train'
val_root = 'datasets/val'

workers = 8
batch_size = 512
input_size = (64, 128)
pixel_mean = [0.485, 0.456, 0.406]
pixel_std = [0.229, 0.224, 0.225]
bright_prob = 0.2
saturation_prob = 0.2
contrast_prob = 0.2
hue_prob = 0.2
grayScale_prob = 0.2
pad = 3
model_name = "resnet"  # [resnet, dpnet]

text_length = 4

# chars = "皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学挂警港澳武ABCDEFGHJKLMNPQRSTUVWXYZ0123456789-"
# chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
chars = "0123456789"

chars2idx = {v: k for k, v in dict(enumerate(chars)).items()}


''' solver setting '''
nepoch = 80
device = 'cuda'
device_id = "0"

output = 'work_dirs/'
displayInterval = 20
lr = 0.01
beta1 = 0.9
lr_step_size = 18
lr_gamma = 0.1
loss_name = "ce"  # amsoftmax, ce
weight_decay = 1e-6

pretrained_path = None
resume_file = "work_dirs/resnet_64x128_acc0.984.pth"
# resume_file = "work_dirs/dpnet_64x128_acc0.684.pth"
