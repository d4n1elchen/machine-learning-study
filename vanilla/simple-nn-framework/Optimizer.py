import numpy as np

eps = 1e-8

class SGD:

    def __init__(self, lr = 1, clipnorm = 1.0):
        self.lr = lr
        self.clipnorm = clipnorm
    
    def init(self, layer):
        pass

    def update_amount(self, layer):
        d = []

        for p in layer.param:
            g = p.grad
            
            g_norm = np.linalg.norm(g)

            if g_norm > self.clipnorm:
                g = self.clipnorm * g / g_norm

            d.append(-g * self.lr)
        return tuple(d)

class Adam:

    def __init__(self, lr = 0.1, clipnorm = 1.0):
        self.clipnorm = clipnorm

        self.lr = lr
        self.beta1 = 0.9
        self.beta2 = 0.999

        self.t = 0

    def init(self, layer):
        for p in layer.param:
            p.t = 0
            p.m = np.zeros(p.shape)
            p.v = np.zeros(p.shape)

    def update_amount(self, layer):
        d = []

        for p in layer.param:
            g = p.grad
            
            p.t += 1
            p.m = p.m * self.beta1 + (1 - self.beta1) * g
            p.v = p.v * self.beta2 + (1 - self.beta2) * (g ** 2)

            m_corrected = p.m / (1 - (self.beta1 ** p.t))
            v_corrected = p.v / (1 - (self.beta2 ** p.t))

            d.append(-self.lr * m_corrected / (np.sqrt(v_corrected) + eps))
        
        return tuple(d)

class Momentum:

    def __init__(self, lr = 0.1, clipnorm = 1.0):
        self.clipnorm = clipnorm

        self.lr = lr
        self.beta = 0.9

        self.t = 0

    def init(self, layer):
        for p in layer.param:
            p.v = np.zeros(p.shape)

    def update_amount(self, layer):
        d = []

        for p in layer.param:
            g = p.grad
            
            p.v = p.v * self.beta - self.lr * g

            d.append(p.v)
        
        return tuple(d)

class AdaGrad:

    def __init__(self, lr = 0.1, clipnorm = 1.0):
        self.clipnorm = clipnorm

        self.lr = lr

        self.t = 0

    def init(self, layer):
        for p in layer.param:
            p.n = np.zeros(p.shape)

    def update_amount(self, layer):
        d = []

        for p in layer.param:
            g = p.grad
            
            p.n += g ** 2

            d.append(-self.lr / np.sqrt(p.n + eps) * g)
        
        return tuple(d)