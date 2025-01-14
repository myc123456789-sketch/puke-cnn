import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
import cv2

import RESNET

transformdata = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
])
def imgread(img_path):
    imgdata = cv2.imread(img_path)
    # 转换成RGB
    imgdata = cv2.cvtColor(imgdata, cv2.COLOR_BGR2RGB)
    # imgdata=cv2.cvtColor(imgdata,cv2.COLOR_BGR2GRAY)
    imgdata = transformdata(imgdata)
    # tensor ---> CHW  --->NCHW
    imgdata = imgdata.unsqueeze(0)
    return imgdata

def inference():
    # 网络准备
    # net = resnet18(weights=None)
    net = RESNET.resnet18Layer(weights=None)
    in_channels = net.conv1.in_channels
    out_channels = net.conv1.out_channels
    net.conv1 = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=7,
        stride=1,
        padding=0,
        bias=False
    )
    net.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features=in_features, out_features=53, bias=True)
    state_dict = torch.load("runs/weights/resnet_cnn_epoch_30.pth", weights_only=True)
    net.load_state_dict(state_dict)
    # 切换到推理模式
    net.eval()

    # 获取要推理的图片
    img_path = "./images/test/king of diamonds/4.jpg"
    imgdata = imgread(img_path)
    
    # 使用模型进行推理
    out = net(imgdata)
    print(out)
    out = nn.Softmax(dim=1)(out)
    classlabels = ['ace of clubs', 'ace of diamonds','ace of hearts', 'ace of spades', 'eight of clubs',
                    'eight of diamonds','eight of hearts','eight of spades','five of clubs','five of diamonds',
                    'five of hearts','five of spades','four of clubs',
                    'four of diamonds',
                    'four of hearts',
                    'four of spades',
                    'jack of clubs',
                    'jack of diamonds',
                    'jack of hearts',
                    'jack of spades',
                    'joker',
                    'king of clubs',
                    'king of diamonds',
                    'king of hearts',
                    'king of spades',
                    'nine of clubs', 
                    'nine of diamonds',
                    'nine of hearts',
                    'nine of spades',
                    'queen of clubs',
                    'queen of diamonds',
                    'queen of hearts',
                    'queen of spades',
                    'seven of clubs',
                    'seven of diamonds',
                    'seven of hearts',
                    'seven of spades',
                    'six of clubs',
                    'six of diamonds',
                    'six of hearts',
                    'six of spades',
                    'ten of clubs',
                    'ten of diamonds',
                    'ten of hearts',
                    'ten of spades',
                    'three of clubs',
                    'three of diamonds',
                    'three of hearts',
                    'three of spades',
                    'two of clubs',
                    'two of diamonds',
                    'two of hearts',
                    'two of spades']

    print(classlabels[torch.argmax(out, dim=1).item()])


if __name__ == "__main__":
    inference()
