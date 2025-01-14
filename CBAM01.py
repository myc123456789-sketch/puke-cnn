import torch
import torch.nn as nn


# 通道注意力
class ChannelAttentionModule(nn.Module):
    def __init__(self, c, r=16):
        super(ChannelAttentionModule, self).__init__()
        # 两个池化
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 共享感知机：全连接层
        self.sharedMLP = nn.Sequential(
            nn.Linear(in_features=c, out_features=c // r),
            nn.ReLU(),
            nn.Linear(in_features=c // r, out_features=c),
        )

    def forward(self, x):
        maxpool = self.maxpool(x)  # 自适应最大池化
        avgpool = self.avgpool(x)  # 自适应平均池化
        # 共享感知机
        maxpool = self.sharedMLP(maxpool.view(maxpool.size(0), -1))
        avgpool = self.sharedMLP(avgpool.view(avgpool.size(0), -1))
        # 激活函数
        pool = nn.Sigmoid()(maxpool + avgpool).view(x.size(0), -1, 1, 1)

        return x * pool


# 空间注意力
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=2,  # 输入通道
                out_channels=1,  # 输出通道
                kernel_size=7,  # 卷积核大小
                stride=1,
                padding=3,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        maxpool = torch.max(x, 1, keepdim=True)[0]
        avgpool = torch.mean(x, 1, keepdim=True)
        pool = torch.cat([maxpool, avgpool], dim=1)
        samap = self.conv(pool)
        return x * samap


# 混合注意力
class CBAM(nn.Module):
    def __init__(self, c, r=16):
        super(CBAM, self).__init__()
        self.cam = ChannelAttentionModule(c, r)
        self.sam = SpatialAttentionModule()

    def forward(self, x):
        x = self.cam(x)
        x = self.sam(x)
        return x


# if __name__ == "__main__":
#     # 模拟数据
#     x = torch.randn(1, 512, 224, 224)

#     model = CBAM(x.size(1), 16)

#     out = model(x)
#     print(out.shape)
 