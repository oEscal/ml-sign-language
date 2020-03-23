import math
from numbers import Number
from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ShuffleSplit
from skimage.transform import rescale


def read_file(path_file: str) -> (np.ndarray, np.ndarray):
    """Function to read datafile and returns a tuple with the following format: (X, y). X represents all the features
        and y represents all the outputs for each data example
    """
    data = pd.read_csv(path_file, header=None, skiprows=1).values
    return np.divide(data[:, 1:], np.max(data)), data[:, 0].reshape(data.shape[0], 1)


def represent_data_graphically(data: np.ndarray, file_save: str, rows: int = 10, cols: int = 10):
    data_image_size = int(math.sqrt(len(data[0, 1:])))
    data_len = len(data)

    fig, axis = plt.subplots(rows, cols, figsize=(data_image_size, data_image_size))
    for row in range(rows):
        for col in range(cols):
            example_id = np.random.randint(data_len)
            axis[row, col].imshow(data[example_id, 1:].reshape(data_image_size, data_image_size, order="F"))
    plt.savefig(file_save)


def sigmoid(z: Union[Number, np.ndarray]) -> np.ndarray:
    """return the sigmoid of z
    """
    return .5 * (1 + np.tanh(.5 * z))


# Computes the gradient of sigmoid function
def sigmoid_gradient(z):
    """computes the gradient of the sigmoid function
    """
    sigmoid_val = sigmoid(z)
    return sigmoid_val * (1 - sigmoid_val)


def rescale_image(data, factor):
    data_size = int(data.shape[0] ** 0.5)
    img = rescale(data.reshape(data_size, data_size), factor, mode='reflect')
    x = img.shape[0] ** 2
    return img.reshape(x, 1).ravel()


def rescale_dataset(dataset, factor=0.75):
    rescaled_data = []
    for img in dataset:
        rescaled_data.append(rescale_image(img, factor))
    return np.asarray(rescaled_data)


def plot_image(data):
    data_size = int(data.shape[0] ** 0.5)
    img = data.reshape(data_size, data_size)
    plt.imshow(img)
    plt.show()









