import numpy as np
from Layers import Linear, Softmax, Sigmoid
from Loss import CrossEntropyLoss
from Optimizer import Adam, SGD, Momentum, AdaGrad

class MLP:

    def __init__(self, input_size=784, layers=[512, 256], num_class=10, lr=0.01):
        self.input_size = input_size
        self.n_layers = len(layers)
        self.num_class = num_class

        # Optimizer
        self.optimizer = Adam(lr=lr)

        # Network topology
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

        # Loss function
        self.criteria = CrossEntropyLoss()
    
    # Feedforward
    def forward(self, x):
        for i in range(self.n_layers):
            x = self.linear[i](x)
            x = self.sigmoid[i](x)
        
        x = self.linear[self.n_layers](x)
        self.out = self.softmax(x)
        return self.out
    
    # Loss function
    def loss(self, y):
        return self.criteria(self.out, y)

    # Calculate grad by backpropagation
    def backward(self):
        dloss = self.criteria.backward()
        dsm = self.softmax.backward(dloss)
        di = self.linear[self.n_layers].backward(dsm)

        for i in reversed(range(self.n_layers)):
            ds = self.sigmoid[i].backward(di)
            di = self.linear[i].backward(ds)
    
    # Update parameters
    def update(self):
        for layer in self.linear:
            self.optimizer.update(layer)