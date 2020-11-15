from data_loader import load_data
from MLP import MLP
from preprocess import preprocess

import numpy as np
import sys
import time

train_image_path = sys.argv[1]
train_label_path = sys.argv[2]
test_image_path = sys.argv[3]

print("Loading data ... ", end="")
n_data_train, image_train, label_train = load_data(image_path=train_image_path, label_path=train_label_path)
n_data_test, image_test = load_data(image_path=test_image_path, label_path=None)
print("Success! n_data_train =", n_data_train, ", n_data_test =", n_data_test)

# Preprocess
image_train, label_onehot_train = preprocess(image_train, label_train)
image_test = preprocess(image_test)

# Shuffle
p = np.random.permutation(n_data_train)

image_train = image_train[p]
label_train = label_train[p]
label_onehot_train = label_onehot_train[p]

# Create model
mlp = MLP()

# Hyper params
epoch = 10
batch_size = 32

def accuracy(out, label):
    out_arg = np.argmax(out, axis=1)
    acc = (out_arg == label).sum() / label.shape[0]
    return acc

for e in range(epoch):
    tic = time.time()
    print("Start epoch", e)
    for i in range(0, n_data_train, batch_size):
        batch_image = image_train[i:min(i + batch_size, n_data_train), :]
        batch_label_onehot = label_onehot_train[i:min(i + batch_size, n_data_train), :]

        out = mlp.forward(batch_image)
        loss = mlp.loss(batch_label_onehot)
        mlp.update()

        if i >= n_data_train - batch_size:
            batch_label = label_train[i:min(i + batch_size, n_data_train)]
            out = mlp.forward(batch_image)
            acc = accuracy(out, batch_label)
            print("Accuracy =", acc)

    toc = time.time()
    print("Time elapsed", toc - tic)

out = mlp.forward(image_test)
out_arg = np.argmax(out, axis=1)
np.savetxt("test_predictions.csv", out_arg, fmt='%i', delimiter=",")