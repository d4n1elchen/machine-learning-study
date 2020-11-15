import numpy as np
from Layers import Linear, Softmax, Sigmoid
from Loss import CrossEntropyLoss
from Optimizer import Adam, SGD, Momentum, AdaGrad

W1 = np.array([[ 0.08454747,  0.37271892,  0.17799139,  0.07773995, -0.13223377],
       [ 0.25269602, -0.10810212,  0.67857074,  0.80308746, -0.20188521],
       [ 0.50528259,  0.05004747,  0.11785664,  0.737155  , -0.74298734]])
W2 = np.array([[-0.71511303, -0.83100611,  0.57611447],
       [ 0.48178163,  0.64087984,  0.82899129],
       [ 0.51815783, -0.0667197 ,  0.48589079],
       [-0.66116809,  0.24235032, -0.61773023],
       [ 0.77018916,  0.0378424 , -0.14780986]])

class MLP:

    def __init__(self, input_size=784, layers=[512, 256], num_class=10, lr=0.01):
        self.input_size = input_size
        self.n_layers = len(layers)
        self.num_class = num_class

        self.optimizer = Adam(lr=lr)

        self.linear = []
        self.sigmoid = []

        self.linear.append(Linear(input_size, layers[0]))
        self.optimizer.init(self.linear[-1])
        self.sigmoid.append(Sigmoid())

        for i in range(1, self.n_layers):
            self.linear.append(Linear(layers[i-1], layers[i]))
            self.optimizer.init(self.linear[-1])
            self.sigmoid.append(Sigmoid())
        
        self.linear.append(Linear(layers[-1], num_class))
        self.optimizer.init(self.linear[-1])
        self.softmax = Softmax()

        self.criteria = CrossEntropyLoss()
    
    def forward(self, x):
        for i in range(self.n_layers):
            x = self.linear[i](x)
            x = self.sigmoid[i](x)
        
        x = self.linear[self.n_layers](x)
        self.out = self.softmax(x)
        return self.out
    
    def loss(self, y):
        return self.criteria(self.out, y)
    
    def update(self):
        dloss = self.criteria.backward()
        dsm = self.softmax.backward(dloss)
        di = self.linear[self.n_layers].backward(dsm)
        dw_update, db_update = self.optimizer.update_amount(self.linear[self.n_layers])
        self.linear[self.n_layers].update(dw_update, db_update)

        for i in reversed(range(self.n_layers)):
            ds = self.sigmoid[i].backward(di)
            di = self.linear[i].backward(ds)
            dw_update, db_update = self.optimizer.update_amount(self.linear[i])
            self.linear[i].update(dw_update, db_update)