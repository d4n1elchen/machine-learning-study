from data_loader import load_data
from MLP import MLP
from preprocess import preprocess

import numpy as np
np.random.seed(0)

print("Loading data ... ", end="")
n_data, image, label = load_data(image_path="train_image.csv", label_path="train_label.csv")
n_data_test, image_test, label_test = load_data(image_path="test_image.csv", label_path="test_label.csv")
print("Success! n_data =", n_data, ", ndata_test = ", n_data_test)

# Preprocess
image, label_onehot = preprocess(image, label)
image_test, label_onehot_test = preprocess(image_test, label_test)

# Shuffle
p = np.random.permutation(n_data)

image = image[p]
label = label[p]
label_onehot = label_onehot[p]

# Create model
mlp = MLP()

# Hyper params
epoch = 5
batch_size = 32

# image = image[0:1]
# label_one_hot = label_one_hot[0:1]

def accuracy(out, label):
    out_arg = np.argmax(out, axis=1)
    acc = (out_arg == label).sum() / label.shape[0]
    return acc

trial_size = 10000

# Iterate through training dataset every `trial_size` data
for t in range(0, n_data, trial_size):
    print("Start trial", int(t / n_data))
    trial_image = image[t:min(t + trial_size, n_data), :]
    trial_label_onehot = label_onehot[t:min(t + trial_size, n_data), :]

    for e in range(epoch):
        print("Start epoch", e)
        for i in range(0, trial_size, batch_size):
            batch_image = trial_image[i:min(i + batch_size, trial_size), :]
            batch_label_onehot = trial_label_onehot[i:min(i + batch_size, trial_size), :]

            out = mlp.forward(batch_image)
            loss = mlp.loss(batch_label_onehot)
            mlp.update()

        out = mlp.forward(image_test)
        acc = accuracy(out, label_test)
        print("Testing accuracy = ", acc)