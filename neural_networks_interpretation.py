import json
import pickle

import numpy as np
import matplotlib.pyplot as plt

from classifiers import Classifier
from utils import sigmoid, read_file

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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

	X, y = read_file("dataset/merged_cv_set.csv")

	precision = []
	cost_plot = plt.figure(1)
	for data in all_data:
		id = data['file_id']

		with open(f"{RESULTS_DIR}classifier_id{id}", 'rb') as file:
			classifier: Classifier = pickle.load(file)
		precision.append((data[study_for], classifier.accuracy(X, y)))
		print(f"Accuracy for {study_for}={data[study_for]}: {classifier.accuracy(X, y)}")

		matrix = classifier.confusion_matrix(X, y)
		print(matrix.diagonal() / matrix.sum(axis=1))

		plt.plot(classifier.history.loss_curve_)
	
	plt.legend([f"{study_for}={data[study_for]}" for data in all_data])
	cost_plot.savefig(f"{study_for}_cost.png")

	precision_plot = plt.figure(2)
	precision.sort(key=lambda x: x[0])
	plt.plot([p[0] for p in precision], [p[1] for p in precision], marker='o')
	plt.title("Accuracy for a range of alpha")
	plt.xlabel('alpha')
	plt.ylabel('accuracy')
	for p in precision[:3]:
		plt.annotate(f"{(p[0], p[1])}", xy=(p[0] + 0.005, p[1] + 0.005))
	precision_plot.savefig(f"{study_for}_accuracy.png")


if __name__ == "__main__":
	main()
