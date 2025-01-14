import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from torchvision.models import resnet18
import RESNET
# 生成验证数据用到的包
import numpy as np
import pandas as pd

# 关闭科学计数法打印
torch.set_printoptions(sci_mode=False)

# 路径兼容处理
current_path = os.path.dirname(__file__)
print(current_path)
data_path = os.path.relpath(os.path.join(current_path, "datasets"))
pth_path = os.path.relpath(
    os.path.join(current_path, r"./runs/weights", "resnet_cnn_epoch_30.pth")
)

# 验证过程数据记录表格
excel_path = os.path.relpath(
    os.path.join(current_path, r"./metrics", "validation_metrics.xlsx")
)


def modelVal():
    # 1， 我们应该有验证数据
    vaildation_data =torchvision.datasets.ImageFolder(
        root=os.path.join("./datasets/valid"),
        transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        ),
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 2，模型：训练好的模型
   
   
   
    # model = resnet18(weights=None)
    model =RESNET.resnet18Layer(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features=in_features, out_features=53, bias=True)
    state_dict = torch.load(pth_path)
    # 初始化到模型
    model.load_state_dict(state_dict)
    # 移到GPU上
    model.to(device)

    # 3，使用训练好的模型对验证数据进行推理，看一下推理效果怎么样
    accuracy = 0
    total_excel_data = np.empty(shape=(0, 55))
    for input, label in DataLoader(vaildation_data, batch_size=16):
        input, label = input.to(device), label.to(device)
        # 使用模型对数据进行推理
        y_pred = model(input)

        # 统计推理正确的记录数
        accuracy += torch.sum(torch.argmax(y_pred, dim=1) == label)
        # 转换为numpy
        pred_excel_data = torch.argmax(y_pred, dim=1).cpu().detach().numpy()
        # numpy升维
        pred_excel_data = np.expand_dims(pred_excel_data, axis=1)
        # 真实值
        label = label.cpu().unsqueeze(dim=1).detach().numpy()
        # 不需要numpy升维
        excel_data = y_pred.cpu().detach().numpy()
        # 把2个numpy数组合并为一个numpy数组
        excel_data = np.concatenate((excel_data, pred_excel_data, label), axis=1)
        # 拼接而不是append
        total_excel_data = np.concatenate((total_excel_data, excel_data), axis=0)

    # 写入Excel
    columns = [*vaildation_data.classes, "predict", "label"]
    df = pd.DataFrame(total_excel_data, columns=columns)
    df.to_excel(excel_path, index=False)
    print("写入Excel成功......")

    print("验证数据集的准确率：%.4f" % (accuracy / len(vaildation_data)))
    # 打印为4为小数的准确率


if __name__ == "__main__":
    modelVal()
