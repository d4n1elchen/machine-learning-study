import torch
import torch.nn as nn
import torch.functional as f

class Inception(nn.Module):
    def __init__(self, prev_layer, n11, n33, n55, np):
        super(Inception, self).__init__()
        # 1x1 branch
        self.b1 = nn.Sequential(
            nn.Conv2d(prev_layer, n11, kernel_size=1),
            nn.BatchNorm2d(n11),
            nn.ReLU(True),
        )
        # 1x1 -> 3x3 branch
        self.b2 = nn.Sequential(
            nn.Conv2d(prev_layer, n33[0], kernel_size=1),
            nn.BatchNorm2d(n33[0]),
            nn.ReLU(True),
            nn.Conv2d(n33[0], n33[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(n33[1]),
            nn.ReLU(True),
        )
        # 1x1 -> 5x5 branch
        self.b3 = nn.Sequential(
            nn.Conv2d(prev_layer, n55[0], kernel_size=1),
            nn.BatchNorm2d(n55[0]),
            nn.ReLU(True),
            nn.Conv2d(n55[0], n55[1], kernel_size=5, padding=2),
            nn.BatchNorm2d(n55[1]),
            nn.ReLU(True),
        )
        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(prev_layer, np, kernel_size=1),
            nn.BatchNorm2d(np),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)

class GoogLeNet(nn.Module):
    def __init__(self, num_class=200):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.a3 = Inception(192,  64,  [96, 128], [16, 32], 32)
        self.b3 = Inception(256, 128, [128, 192], [32, 96], 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  [96, 208], [16,  48],  64)
        self.b4 = Inception(512, 160, [112, 224], [24,  64],  64)
        self.c4 = Inception(512, 128, [128, 256], [24,  64],  64)
        self.d4 = Inception(512, 112, [144, 288], [32,  64],  64)
        self.e4 = Inception(528, 256, [160, 320], [32, 128], 128)

        self.a5 = Inception(832, 256, [160, 320], [32, 128], 128)
        self.b5 = Inception(832, 384, [192, 384], [48, 128], 128)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.linear = nn.Linear(1024, num_class)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
