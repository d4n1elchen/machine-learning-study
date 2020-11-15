import numpy as np

def preprocess(image, label=None):
    image = image / 255

    if label is not None:
        label_one_hot = np.zeros((label.size, 10))
        label_one_hot[np.arange(label.size), label] = 1

        return image, label_one_hot
    else:
        return image