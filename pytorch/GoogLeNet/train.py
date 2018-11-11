# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 13:52:55 2018

@author: daniel
"""

import os
import time

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchsummary import summary

from googlenet import GoogLeNet

#%% Main function
def main():
    ## Load data
    data_dir = '.\\tiny-imagenet-200'

    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')
    augmentation = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[1, 1, 1])
                   ])
    val_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[1, 1, 1])
                   ])

    train_dataset = datasets.ImageFolder(traindir, augmentation)
    val_dataset = datasets.ImageFolder(valdir, val_transform)

    ## Prepare dataloader
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True)

    ## Init network
    model = GoogLeNet()

    # Hyper parameters
    lr = 0.01
    momentum = 0.9
    weight_decay = 0.0005

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    # Get GPU
    is_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if is_gpu else "cpu")
    if is_gpu:
        print("Move to GPU: ", device)
        model.to(device)
        criterion.to(device)

    summary(model, (3, 224, 224))

    ## Training
    check_epoch = 10
    impv_prec1 = 0
    prev_prec1 = 0
    best_prec1 = 0
    epochs = 200
    for epoch in range(epochs):
        train(train_loader, model, criterion, optimizer, is_gpu, device, epoch)
        prec1 = validate(val_loader, model, criterion, is_gpu, device)

        # Save model if the precision is improved
        if prec1 > best_prec1:
            best_prec1 = prec1
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, 'model_best.pth')

        impv_prec1 += prec1 - prev_prec1
        prev_prec1 = prec1

        # Check acc improvement every 10 epoch
        if (epoch+1) % check_epoch == 0:
            if impv_prec1 < 1.0:
                lr /= 10
                check_epoch *= 2
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            print('CHECK_IMPROVE Epoch: {}, Prec1 imporve: {:.3f}, lr: {:.8f}'.format(epoch, impv_prec1, lr))
            impv_prec1 = 0

#%% Training
def train(train_loader, model, criterion, optimizer, is_gpu, device, epoch):
    # Profiler
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # Metrics
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # Move input and target to GPU if abailable
        if is_gpu:
            input, target = input.to(device), target.to(device)

        # Feed forward
        output = model(input)
        loss = criterion(output, target)

        # Metrics
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # Backprop and SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Profiling
        batch_time.update(time.time() - end)
        end = time.time()

        # Print every 100 batch
        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

#%% Validation
def validate(val_loader, model, criterion, is_gpu, device):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if is_gpu:
                input, target = input.to(device), target.to(device)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

        print('VALIDATE Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

#%% Some helping function
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
