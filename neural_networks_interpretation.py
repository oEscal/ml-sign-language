import json

import numpy as np
from utils import sigmoid, read_file

RESULTS_DIR = "results/neural_networks/"


def predict(theta1, theta2, X):
	"""
	Predict the label of an input given a trained neural network
	"""

	# number of training examples
	m = X.shape[0]

	# add an extra column of 1´s corresponding to xo=1
	X = np.append(np.ones((m, 1)), X, axis=1)

	# Compute the output of the hidden layer (with sigmoid activation functions)
	z1 = np.dot(X, theta1.T)  # Inputs to the hidden layer neurons
	a1 = sigmoid(z1)  # Outputs  of the hidden layer neurons

	# Add a column of ones
	a1 = np.append(np.ones((m, 1)), a1, axis=1)

	# Compute the output of the output layer (with sigmoid activation functions)
	z2 = np.dot(a1, theta2.T)  # Inputs to the output layer neurons
	a2 = sigmoid(z2)  # Outputs  of the output layer neurons

	return np.argmax(a2, axis=1) + 1


def main():
	study_for = "alpha"
	time_file_name = f"time.json"   # _{study_for}

	file_info = open(f"{RESULTS_DIR}{time_file_name}")
	all_data = file_info.readlines()
	file_info.close()

	all_data = [json.loads(line.replace('\n', '')) for line in all_data]

	X, y = read_file("dataset/sign_mnist_train.csv")
	m = X.shape[0]

	for data in all_data:
		id = data['theta_file_id']

		theta1 = np.load(f"{RESULTS_DIR}theta1_id{id}.npy")
		theta2 = np.load(f"{RESULTS_DIR}theta2_id{id}.npy")

		pred = predict(theta1, theta2, X)
		print(f"Training Set Accuracy for {study_for}={data[study_for]}:", sum(pred[:, np.newaxis] == y)[0] / m * 100,
		      "%")


if __name__ == "__main__":
	main()
