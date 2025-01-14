import torch
from torchvision.models import resnet18
import torch.nn as nn

import RESNET


def export():
    # 有一个网络
    net = RESNET.resnet18Layer(weights=None)

    # 下面是对网络的小改变
    in_channels = net.conv1.in_channels
    out_channels = net.conv1.out_channels
    net.conv1 = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=7,
        stride=1,
        padding=1,
        bias=False,
    )
    net.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features=in_features, out_features=53, bias=True)
    # 权重参数
    state_dict = torch.load("runs/weights/resnet_cnn_epoch_20.pth", weights_only=True)
    net.load_state_dict(state_dict)

    # 导出为onnx格式
    imgdata = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        net,
        imgdata,
        "runs/weights/resnet18.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
    )
    print("ONNX Export Success!")


if __name__ == "__main__":
    export()
