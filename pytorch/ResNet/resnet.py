import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, prev_layer, out_channel, stride=1):
        super(BasicBlock, self).__init__()

        self.resmap = nn.Sequential(
            nn.Conv2d(prev_layer, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or prev_layer != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(prev_layer, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = self.resmap(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    def __init__(self, prev_layer, out_channel, stride=1):
        super(Bottleneck, self).__init__()

        neck_channel = out_channel // 4

        self.resmap = nn.Sequential(
            nn.Conv2d(prev_layer, neck_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(neck_channel),
            nn.ReLU(True),
            nn.Conv2d(neck_channel, neck_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(neck_channel),
            nn.ReLU(True),
            nn.Conv2d(neck_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or prev_layer != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(prev_layer, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = self.resmap(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=200):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.conv2 = self._make_layer(block, 64, 64, num_blocks[0], stride=1)
        self.conv3 = self._make_layer(block, 64, 128, num_blocks[1], stride=2)
        self.conv4 = self._make_layer(block, 128, 256, num_blocks[2], stride=2)
        self.conv5 = self._make_layer(block, 256, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.linear = nn.Linear(512, 200)

    def _make_layer(self, block, prev_layer, out_channel, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(prev_layer, out_channel, stride))
            prev_layer = out_channel
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])
