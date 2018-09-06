# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 22:34:08 2018

@author: daniel
"""

import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

#%% Define network
class AlexNet(nn.Module):

    def __init__(self, num_classes=200):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x

#%% Load the model
model = AlexNet()
checkpoint = torch.load('model_best.pth')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

val_transform = transforms.Compose([
                     transforms.Resize(256),
                     transforms.CenterCrop(224),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[1, 1, 1])
                ])

#%% Load labels
from collections import defaultdict
wnids = []
words = defaultdict(str)
with open('wnids.txt', 'r') as f:
    for line in f:
        wnids.append(line.strip('\n'))
wnids = sorted(wnids)
with open('words.txt', 'r') as f:
    for line in f:
        sp = line.strip('\n').split('\t', 1)
        wnid = sp[0]
        word = sp[1]
        if wnid in wnids:
            words[wnid] = word

#%% Load image and make prediction
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
rand_id = np.random.randint(1000)
image = Image.open(os.path.join('tiny-imagenet-200', 'test', 'images', 
                               'test_{}.JPEG'.format(rand_id)))
plt.imshow(image)

with torch.no_grad():
    output = model(val_transform(image).unsqueeze(0)).numpy()[0]
    top5_cate = output.argsort()[::-1][:5]
    
    for i, cate in enumerate(top5_cate):
        print('PRED'+str(i+1)+': '+words[wnids[cate]])