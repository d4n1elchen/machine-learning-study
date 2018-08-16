import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

X = np.array([-100, -1, 0, 1, 100])
print(sigmoid(X)) # [ 3.72007598e-44 2.68941421e-01 5.00000000e-01 7.31058579e-01 1.00000000e+00 ]
