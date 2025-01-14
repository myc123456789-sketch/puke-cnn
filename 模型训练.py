import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
import time
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

import RESNET

# 定义模型
current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, "datasets")
weight_path = os.path.join(current_path, "runs/weights")

# 创建保存路径
os.makedirs(weight_path, exist_ok=True)

# 初始化 TensorBoard
writer = SummaryWriter()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():
    # 数据获取
    train_datasets = torchvision.datasets.ImageFolder(
        root=os.path.join(data_path, "train"),
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomCrop(224, padding=4),  # 随机裁剪并填充
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
            transforms.RandomGrayscale(p=0.9),# 随机灰度
            transforms.RandomRotation(10),  # 随机旋转 ±10 度
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
        ])
    )
    

     # 初始化带有CBAM的ResNet18模型
    net = RESNET.resnet18Layer(weights=None)
    # net = resnet18(weights=None)
    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features=in_features, out_features=53, bias=True)
    
    if os.path.exists(os.path.join(weight_path, "resnet_cnn_epoch_30.pth")):
        state_dict = torch.load(os.path.join(weight_path, "resnet_cnn_epoch_30.pth"))
        net.load_state_dict(state_dict, strict=False)  # 使用strict=False以允许部分加载
    

    # 分类器：删除全连接层的权重参数
    # state_dict.pop("fc.weight")
    # state_dict.pop("fc.bias")
    
    # #4，更新参数到我们的网络模型参数中
    # my_state_dict = net.state_dict()
    # my_state_dict.update(state_dict)
    # net.load_state_dict(my_state_dict)  
    net.to(device)
    # 保存网络结构到 TensorBoard
    input_to_model = torch.randn(64, 3, 32, 32).to(device)  # 使用正确的输入尺寸
    writer.add_graph(net, input_to_model)

    # 设置训练参数
    epochs = 5
    batch_size = 32
    criterion = nn.CrossEntropyLoss(reduction="mean")  # 使用平均损失
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    num_workers = 2 if os.name != 'nt' else 0  # 根据操作系统配置 num_workers

    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0 
        correct = 0
        total = 0

        train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            #每10个批次记录一次图像
            if i % 10 == 9:
                grid = make_grid(inputs[:8], nrow=4, normalize=True)
                writer.add_image(f'images_epoch_{epoch}_batch_{i}', grid, global_step=epoch * len(train_loader) + i)

        # 计算平均损失和准确率
        avg_loss = running_loss / len(train_loader)
        accuracy = correct / total

        # 打印训练信息
        print(f"[{epoch + 1}/{epochs}] Time: {time.time() - start_time:.2f}s, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        # 将训练信息记录到 TensorBoard
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/train', accuracy, epoch)

        
        # 定期保存模型
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model_filename = f"resnet_cnn_epoch_{epoch + 31}.pth"
            torch.save(net.state_dict(), os.path.join(weight_path, model_filename))

    # 关闭 TensorBoard writer
    writer.close()

if __name__ == "__main__":
    train()