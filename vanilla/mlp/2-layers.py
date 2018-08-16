import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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
print(X)

## layer1
# weights and bias
W1 = np.random.random(size=(input_n, l1_n_node))
b1 = np.zeros(shape=(l1_n_node,))
l1 = sigmoid(np.matmul(X, W1) + b1) # X*W1 + b1

print(W1) # [[  4.17022005e-01   7.20324493e-01   1.14374817e-04   3.02332573e-01]
          #  [  1.46755891e-01   9.23385948e-02   1.86260211e-01   3.45560727e-01]
          #  [  3.96767474e-01   5.38816734e-01   4.19194514e-01   6.85219500e-01]]

print(l1) # [[ 0.59791076  0.63153712  0.60329049  0.66490264]
          #  [ 0.63263166  0.65275138  0.64690327  0.73706713]
          #  [ 0.69291643  0.77887824  0.60331786  0.72860414]
          #  [ 0.72323098  0.79437146  0.64692939  0.79135506]]

## layer2
# weights and bias
W2 = np.random.random(size=(l1_n_node, l2_n_node))
b2 = np.zeros(shape=(l2_n_node,))
l2 = sigmoid(np.matmul(l1, W2) + b2) # l1*W2 + b2

print(W2) # [[ 0.20445225  0.87811744]
          #  [ 0.02738759  0.67046751]
          #  [ 0.4173048   0.55868983]
          #  [ 0.14038694  0.19810149]]


print(l2) # [[ 0.61884298  0.80490403]
          #  [ 0.62729991  0.81766482]
          #  [ 0.6264586   0.83369992]
          #  [ 0.63429445  0.84368203]]
