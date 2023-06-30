from torch.nn import Module, BatchNorm2d, Sequential, Conv2d, ReLU, Linear, AvgPool2d

from src.configs.model_config import RESNET_CONFIGS
from src.constants import resnet_type
from src.modules.pre_activation_block import PreActivationBlock
from src.modules.pre_activation_bottleneck import PreActivationBottleneck


class ResNet(Module):
    def __init__(
        self,
        enc_dim: int,
        model_type: resnet_type = resnet_type.TYPE_18_LAYERS,
        num_classes: int = 2
    ):
        super().__init__()
        self.in_channels = 16

        num_blocks, block = RESNET_CONFIGS[model_type]

        self.conv1 = Conv2d(1, 16, kernel_size=3, stride=1, bias=False)
        self.bn1 = BatchNorm2d(16)
        self.activation = ReLU()

        self.layer1 = self._make_layer(num_blocks[0], block, out_channels=64, stride=1)
        self.layer2 = self._make_layer(num_blocks[1], block, out_channels=128, stride=2)
        self.layer3 = self._make_layer(num_blocks[2], block, out_channels=256, stride=2)
        self.layer4 = self._make_layer(num_blocks[3], block, out_channels=512, stride=2)
        self.bn5 = BatchNorm2d(512)
        self.avg_pool = AvgPool2d(kernel_size=(1, 12), stride=1)
        self.fc = Linear(512, enc_dim)
        self.fc_mu = Linear(enc_dim, num_classes) if num_classes >= 2 else Linear(enc_dim, 1)

    def _make_layer(
        self,
        num_blocks: int,
        block: PreActivationBlock | PreActivationBottleneck,
        out_channels: int,
        stride: int = 1
    ):
        layers = [block(
            in_channels=self.in_channels,
            out_channels=out_channels,
            stride=stride
        )]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=stride
            ))
        return Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.bn5(out)
        out = self.activation(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        feat = self.fc(out)
        mu = self.fc_mu(feat)
        return feat, mu
