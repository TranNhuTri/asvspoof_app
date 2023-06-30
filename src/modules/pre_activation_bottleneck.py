from torch import nn
from torch.nn import functional


class PreActivationBottleneck:
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1, bias=False)

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.bn1(x)
        out = functional.relu(out)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)

        out = self.bn2(out)
        out = functional.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = functional.relu(out)
        out = self.conv3(out)

        out += shortcut
        return out
