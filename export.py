import os
import torch
from models.resnet import ResDpnet
import config as cfg


def main():
    # 初始化模型

    model = ResDpnet(False, length=cfg.text_length)
    model_path = "work_dirs/best_0.970.pth"
    model_info = torch.load(model_path)
    model.load_state_dict(model_info["net"])
    model.to("cpu")
    model.eval()

    output = os.path.join("work_dirs", "best_%.3f.onnx" % model_info['acc'])

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
