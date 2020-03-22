import math
from numbers import Number
from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_file(path_file: str) -> (np.ndarray, np.ndarray):
	"""Function to read datafile and returns a tuple with the following format: (X, y). X represents all the features
		and y represents all the outputs for each data example
	"""
	data = pd.read_csv(path_file, header=None, skiprows=1).values
	return data[:, 1:], data[:, 0].reshape(data.shape[0], 1)


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
	return 1 / (1 + np.exp(-z))


# Computes the gradient of sigmoid function
def sigmoid_gradient(z):
	"""computes the gradient of the sigmoid function
	"""
	sigmoid_val = sigmoid(z)
	return sigmoid_val * (1 - sigmoid_val)
