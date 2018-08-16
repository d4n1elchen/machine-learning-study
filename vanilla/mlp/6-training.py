import numpy as np

def sigmoid(x):
    """ Sigmoid function """
    return 1 / (1 + np.exp(-x))

def softmax(x):
    """ Softmax function, xxx[:, None] means add new axis """
    return np.exp(x) / np.sum(np.exp(x), axis=1)[:, None]

def cross_entropy(pred_y, real_y):
    """ Compute cross entropy between pred_y and real_y"""
    return -np.sum(real_y * np.log(pred_y), axis=1)

## Set random seed for deterministic result
np.random.seed(87)

## Network configuration
input_n = 3
l1_n_node = 5
l2_n_node = 3 # output_n

learning_rate = 1
epoch = 5000

## Data
X = np.array([[0,0,1],
              [0,1,0],
              [1,0,0],
              [0,1,1],
              [1,0,1],
              [1,1,0],
              [1,1,1]])
y = np.array([[1,0,0],
              [1,0,0],
              [1,0,0],
              [0,1,0],
              [0,1,0],
              [0,1,0],
              [0,0,1]])
batch_size = len(y)

## Weight init

## layer1
# weights and bias
W1 = np.random.random(size=(input_n, l1_n_node))
b1 = np.zeros(shape=(l1_n_node,))

## layer2
# weights and bias
W2 = np.random.random(size=(l1_n_node, l2_n_node))
b2 = np.zeros(shape=(l2_n_node,))

for i in range(epoch):
    ## feed forward
    l1 = sigmoid(np.matmul(X, W1) + b1)
    l2 = np.matmul(l1, W2) + b2

    ## Softmax
    out = softmax(l2)

    ## loss function
    loss = cross_entropy(out, y)
    print(loss.mean())

    ## back propagation
    dl2 = (out - y)                                        # dJ/dl2 = dJ/do  * do/dl2 (z2 = l1 * W2 + b2)
    dW2 = np.matmul(l1.T, dl2) / batch_size                # dJ/dW2 = dJ/dz2 * dz2/dW2
    db2 = np.matmul(np.ones(batch_size), da2) / batch_size # dJ/db2 = dJ/da2 * da2/db2

    dl1 = np.matmul(da2, W2.T) * (l1 * (1 - l1))           # dJ/dl1 = dJ/da2 * da2/dl1 * dl1/da1 (a1 = X * W1 + b1)
    dW1 = np.matmul(X.T, da1) / batch_size                 # dJ/dW1 = dJ/da1 * da1/dW1
    db1 = np.matmul(np.ones(batch_size), da1) / batch_size # dJ/db1 = dJ/da1 * da1/db1

    ## gradient descent
    W2 = W2 - learning_rate * dW2 # w = w - r * dw
    b2 = b2 - learning_rate * db2
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
