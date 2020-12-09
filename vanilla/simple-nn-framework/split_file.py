import pandas as pd
import numpy as np

image = pd.read_csv("train_image.csv", header=None).values
label = pd.read_csv("train_label.csv", header=None).values
n_data = image.shape[0]

batch_size = 10000
for i in range(0, n_data, batch_size):
    print(f"Saving {i} batch ...")
    image_b = image[i:min(i + batch_size, n_data), :]
    label_b = label[i:min(i + batch_size, n_data), :]
    np.savetxt(f"train_image_{i}.csv", image_b, fmt='%i', delimiter=",")
    np.savetxt(f"train_label_{i}.csv", label_b, fmt='%i', delimiter=",")