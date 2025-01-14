import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock, _resnet, ResNet18_Weights, ResNet
from SENet import SENet
from torchvision.models.resnet import _ovewrite_named_param


class ResNetLayer(ResNet):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group: int = 64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNetLayer, self).__init__(
            block,
            layers,
            num_classes,
            zero_init_residual,
            groups,
            width_per_group,
            replace_stride_with_dilation,
            norm_layer=norm_layer,
        )
        # 更新Layer
        self.layer3 = self._modify_layer(self.layer3)
        self.layer4 = self._modify_layer(self.layer4)

    def _modify_layer(self, layer):
        myLayer = []
        for block in layer:
            myLayer.append(block)
            myLayer.append(SENet(block.conv2.out_channels, 16))
            
        return nn.Sequential(*myLayer)


def resnet18Layer(*, weights=None, progress=True, **kwargs):

    weights = ResNet18_Weights.verify(weights)
    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)


def _resnet(
    block,
    layers,
    weights,
    progress: bool,
    **kwargs,
):
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNetLayer(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(
            weights.get_state_dict(progress=progress, check_hash=True)
        )

    return model


if __name__ == "__main__":
    model = resnet18Layer()
    print(model)
