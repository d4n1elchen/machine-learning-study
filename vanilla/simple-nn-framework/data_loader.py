import pandas as pd

def load_data(image_path="train_image.csv", label_path="train_label.csv"):
    image = load_image(image_path)

    if label_path is not None:
        label = load_label(label_path)
        return image.shape[0], image, label
    else:
        return image.shape[0], image

def load_image(path):
    return pd.read_csv(path, header=None).values

def load_label(path):
    return pd.read_csv(path, header=None).values.squeeze(axis=1)