import os.path as osp

# sparse train
SPARSE = True
SCALE_SPARSE_RATE = 0.01
PRUE_RATIO = 0.85
LR = 0.00001

pruned_output = "models/pruned"

sparse_train_model = "models/sparse/DpNetV3_SR_Acc0.97694.pth"
pruned_model = osp.join(pruned_output, "DpNetV3_Purned.pth")
pruned_cfg = osp.join(pruned_output, "prune.txt")

device = "cuda"
