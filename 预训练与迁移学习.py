import time
import torch
import torchvision
from torchvision.models import resnet18,ResNet18_Weights
import torchvision.transforms as transforms
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

# 路径
current_path = os.path.dirname(__file__)
data_path = os.path.relpath(os.path.join(current_path, "datasets"))
weight_path = os.path.relpath(os.path.join(current_path, "runs/weights"))


def pretrained():
    # 1，【预训练】我要先拿到预训练的权重参数
    pretrained_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # 保存预训练权重文件
    torch.save(
        pretrained_model.state_dict(),
        os.path.join(weight_path, "resnet18_pretrained.pth"),
    )

    # 【迁移学习】
    # 2，构建自己的网络模型
    net = resnet18(weights=None)
    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features=in_features, out_features=53, bias=True)
    
    # 修改conv1
    in_channels = net.conv1.in_channels
    out_channels = net.conv1.out_channels
    net.conv1 = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=False,
    )
    # 池化的操作不是和我这个小图片
    net.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
    # 分类器
    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features=in_features, out_features=10, bias=True)
    print(net)

    # 3， 加载预训练权重参数并处理好
    state_dict = torch.load(os.path.join(weight_path, "resnet18_pretrained.pth"))
    # 分类器：删除全连接层的权重参数
    state_dict.pop("fc.weight")
    state_dict.pop("fc.bias")
    del state_dict['conv1.weight']
    
    my_state_dict = net.state_dict()
    my_state_dict.update(state_dict)
    
    # 4，更新参数到我们的网络模型参数中
    net.load_state_dict(my_state_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # 5，训练自己的网络模型
    ## 5.1， 冻结不需要更新的权重参数，假设只更新全连接层参数
   

    grade_true_state_dict = filter(lambda p: p.requires_grad, net.parameters())
  
    # 数据集
    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_path, "train"),
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomCrop(32, padding=4),  # 随机裁剪并填充
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
            transforms.RandomGrayscale(p=0.1),# 随机灰度
            transforms.RandomRotation(10),  # 随机旋转 ±10 度
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
        ])
    )
    # 模型训练
    epochs = 10
    lr = 0.001
    batch_size = 32
    # 损失函数
    cred = nn.CrossEntropyLoss(reduction="sum")
    optimizer = optim.Adam(grade_true_state_dict, lr=lr)

    for epoch in range(epochs):
        start_time = time.time()  # 开始时间
        accuracy = 0
        total_loss = 0
        # count = 0
        for x, y in DataLoader(train_dataset, batch_size=batch_size, shuffle=True):
            x, y = x.to(device), y.to(device)
            # print(x.shape, y)
            # 使用网络进行预测
            yhat = net(x)
            # 预测正确的梳理
            accuracy += torch.sum(torch.argmax(yhat, dim=1) == y)
            # 计算损失
            loss = cred(yhat, y)
            total_loss += loss
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 梯度更新
            optimizer.step()
        torch.save(net.state_dict(), os.path.join(weight_path, f"last.pth"))

        # 完成模型的保存
        print(
            f"{epoch}/{epochs} time：{time.time() - start_time} accuracy:{accuracy / len(train_dataset)} loss:{total_loss / len(train_dataset)}"
        )


if __name__ == "__main__":
    pretrained()
