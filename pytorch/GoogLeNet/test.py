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

from googlenet import GoogLeNet

#%% Load the model
model = GoogLeNet()
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
