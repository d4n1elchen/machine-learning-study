import numpy as np

def sigmoid(x):
    """ Sigmoid function """
    return 1 / (1 + np.exp(-x))

def softmax(x):
    """ Softmax function, xxx[:, None] means add new axis """
    return np.exp(x) / np.sum(np.exp(x), axis=1)[:, None]

## Set random seed for deterministic result
np.random.seed(1)

## Network configuration
input_n = 3
l1_n_node = 4
l2_n_node = 2 # output_n

## Data
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

## layer1
# weights and bias
W1 = np.random.random(size=(input_n, l1_n_node))
b1 = np.zeros(shape=(l1_n_node,))
l1 = sigmoid(np.matmul(X, W1) + b1) # X*W1 + b1

## layer2
# weights and bias
W2 = np.random.random(size=(l1_n_node, l2_n_node))
b2 = np.zeros(shape=(l2_n_node,))
l2 = sigmoid(np.matmul(l1, W2) + b2) # l1*W2 + b2
print(l2) # [[ 0.61884298  0.80490403]
          #  [ 0.62729991  0.81766482]
          #  [ 0.6264586   0.83369992]
          #  [ 0.63429445  0.84368203]]

## Softmax
out = softmax(l2)
print(out) # [[ 0.45361847  0.54638153]
           #  [ 0.45255197  0.54744803]
           #  [ 0.44837431  0.55162569]
           #  [ 0.44784352  0.55215648]]
