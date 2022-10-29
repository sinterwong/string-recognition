import os
import torch
from models.resnet import ResDpnet
import config as cfg


def main():
    # 初始化模型

    model = ResDpnet(True, length=cfg.text_length)
    model_path = "work_dirs/resnet_64x128_acc0.998.pth"
    model_info = torch.load(model_path)
    model.load_state_dict(model_info["net"])
    model.to("cpu")
    model.eval()

    output = model_path.replace(".pth", ".onnx")

    dummy_input1 = torch.randn(1, 3, cfg.input_size[0], cfg.input_size[1])
    # convert to onnx
    print("convert to onnx......")
    input_names = ["input"]
    output_names = ["output"]
    # torch.onnx.export(model, (dummy_input1, dummy_input2, dummy_input3), "C3AE.onnx", verbose=True, input_names=input_names, output_names=output_names)
    torch.onnx.export(model, dummy_input1, output, verbose=False,
                      input_names=input_names, output_names=output_names)
    print("convert done!")


if __name__ == "__main__":
    main()
