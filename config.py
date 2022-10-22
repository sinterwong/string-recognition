import os.path as osp

''' data iter setting '''
# train_root = '/home/wangjq/license_plate_dataset/licence-plate/train'
# val_root = '/home/wangjq/license_plate_dataset/licence-plate/val'
train_root = 'datasets/train'
val_root = 'datasets/val'

workers = 8
batch_size = 128
input_size = (32, 128)
# pixel_mean = [0.485, 0.456, 0.406]
# pixel_std = [0.229, 0.224, 0.225]
bright_prob = 0.2
saturation_prob = 0.2
contrast_prob = 0.2
hue_prob = 0.2
grayScale_prob = 0.2
pad = 3

text_length = 4

# chars = "皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学挂警港澳武ABCDEFGHJKLMNPQRSTUVWXYZ0123456789-"
chars = "abcdefghijklmnopqrstuvwxyz0123456789"

chars2idx = {v: k for k, v in dict(enumerate(chars)).items()}


''' solver setting '''
nepoch = 300
device = 'cuda'
device_id = "0"

output = 'work_dirs/'
displayInterval = 20
lr = 0.001
beta1 = 0.9
lr_step_size = 18
lr_gamma = 0.5
loss_name = "ce"  # amsoftmax, ce
weight_decay = 1e-6

pretrained_path = None
resume_file = "models/resnet_pretrained.pth"
