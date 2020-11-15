import numpy as np
np.random.seed(0)

eps = 1e-8

class Parameter(np.ndarray):

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        obj._grad = np.zeros(obj.shape)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self._grad = getattr(obj, '_grad', None)

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, grad):
        self._grad = grad

class Layer:

    def __init__(self):
        self.param = []

    def __call__(self, x):
        return self.forward(x)

class Linear(Layer):

    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        scale = 1 / max(1., (input_size + output_size) / 2.)
        limit = np.sqrt(3.0 * scale)

        self.weight = Parameter(np.random.uniform(-limit, limit, size=(input_size, output_size)))
        self.bias = Parameter(np.zeros(shape=(output_size,)))

        self.param.append(self.weight)
        self.param.append(self.bias)
    
    def load_weight(self, w):
        self.weight = w
    
    def forward(self, x):
        self.input = x
        return np.matmul(x, self.weight) + self.bias
    
    def backward(self, grad_out):
        m = self.input.shape[0]
        grad_input = np.matmul(grad_out, self.weight.T)
        grad_weight = np.matmul(self.input.T, grad_out) / m
        grad_bias = np.sum(grad_out, axis=0) / m

        self.weight.grad = grad_weight.view(np.ndarray)
        self.bias.grad = grad_bias.view(np.ndarray)
        
        return grad_input
    
    def update(self, dw, db):
        self.weight += dw
        self.bias += db

class Sigmoid(Layer):

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad_out):
        return grad_out * (self.output * (1 - self.output))

class Softmax(Layer):

    def forward(self, x):
        self.n_input = x.shape[1]
        self.o = np.exp(x) / np.sum(np.exp(x), axis=1)[:, None]
        return self.o

    def backward(self, grad_out):
        m = grad_out.shape[0]
        grad_input = np.zeros((m, self.n_input))
        for k in range(self.n_input):
            for i in range(self.n_input):
                if i == k:
                    grad_input[:, k] += self.o[:, i] * (1 - self.o[:, i]) * grad_out[:, i]
                else:
                    grad_input[:, k] += -self.o[:, i] * self.o[:, k] * grad_out[:, i]
        return grad_input