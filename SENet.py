import torch
import torch.nn as nn


class SENet(nn.Module):
    def __init__(self, inplanes, r=16):
        super(SENet, self).__init__()
        # Squeeze：压缩、降维、挤压
        self.sq = nn.AdaptiveAvgPool2d(1)
        # Excitation:激活
        self.ex = nn.Sequential(
            nn.Linear(inplanes, inplanes // r),
            nn.ReLU(),
            nn.Linear(inplanes // r, inplanes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 缓存x
        intifi = x
        x = self.sq(x)
        x = x.view(x.size(0), -1)
        x = self.ex(x).unsqueeze(2).unsqueeze(3)
        return intifi * x


if __name__ == "__main__":
    torch.manual_seed(1)
    input = torch.randn(1, 512, 224, 224)
    print("输入特征：", input.shape)
    model = SENet(input.size(1), 16)

    out = model(input)
    print("加入了SE之后的特征值：", out.shape)
    pass
