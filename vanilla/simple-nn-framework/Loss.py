import numpy as np

eps = 1e-8

class Loss:

    def __call__(self, out, y):
        return self.forward(out, y)

class CrossEntropyLoss(Loss):

    def forward(self, out, y):
        self.y = y
        self.o = out
        return -np.sum(y * np.log(out), axis=1)

    def backward(self):
        return -self.y / (self.o + eps)