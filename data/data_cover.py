import os
import glob
import tqdm
import shutil

data_root = "/home/wangxt/workspace/datasets/CaptchaDataset-master/OCR_Dataset"
out_root = "data/train"
if not os.path.exists(out_root):
    os.makedirs(out_root)

image_paths = glob.glob(data_root + "/*.jp*")
json_file = os.path.join(data_root, "label_dict.txt")

with open(json_file, mode='r') as rf:
    data_mapping = eval(rf.read())

for p in image_paths:
    name = os.path.basename(p)
    code = data_mapping[name]
    
    out_path = os.path.join(out_root, code + "_" + name)
    shutil.copy(p, out_path)
