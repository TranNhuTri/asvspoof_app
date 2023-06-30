from torch import nn
from torch.nn import functional


class PreActivationBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expansion = 1
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.bn1(x)
        out = functional.relu(out)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)

        out = self.bn2(out)
        out = functional.relu(out)
        out = self.conv2(out)

        out += shortcut
        return out
