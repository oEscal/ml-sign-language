import json
import pickle

import matplotlib.pyplot as plt

from classifiers import Classifier
from utils import read_file

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

RESULTS_DIR = "results/neural_networks/"


def main():
	study_for = "alpha"
	time_file_name = f"time.json"

	with open(f"{RESULTS_DIR}{time_file_name}") as file:
		all_data = file.readlines()
		all_data = sorted([json.loads(line.replace('\n', '')) for line in all_data], key=lambda x: x[study_for])

	X_train, y_train = read_file("dataset/merged_cv_set.csv")
	X_cv, y_cv = read_file("dataset/merged_test_set.csv")

	accuracy = []
	error = []
	error_train = []
	for data in all_data:
		id = data['file_id']

		with open(f"{RESULTS_DIR}classifier_id{id}", 'rb') as file:
			classifier: Classifier = pickle.load(file)

		current_accuracy = classifier.accuracy(X_cv, y_cv)
		current_error = classifier.log_error(X_cv, y_cv)
		current_error_train = classifier.log_error(X_train, y_train)

		error.append(current_error)
		error_train.append(current_error_train)
		accuracy.append(current_accuracy)

		confusion_matrix = classifier.confusion_matrix(X_cv, y_cv)

		# print info
		for i in range(10):
			print()

		title = f"{study_for}={data[study_for]}"
		print(f"{title:^70}")
		print(f" -> Accuracy: {current_accuracy}\n")
		print(f" -> Error: {current_error}")
		print(f" -> Confusion matrix:\n{confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)}\n")

		plt.plot(classifier.history.loss_curve_)

	# make plots
	study_for_data = [data[study_for] for data in all_data]

	cost_plot = plt.figure(1)
	plt.legend([f"{study_for}={data[study_for]}" for data in all_data])
	cost_plot.savefig(f"{RESULTS_DIR}{study_for}_cost.png")

	error_plot = plt.figure(2)
	plt.plot(study_for_data, error_train, color="blue", marker='o')
	plt.plot(study_for_data, error, color="orange", marker='^')
	plt.legend(["Training score", "Cross Validation score"])

	# get the minimum error and add a text on that point
	min_error = min(error)
	min_error_index = error.index(min_error)
	min_error_data = study_for_data[min_error_index]
	plt.annotate(f"({min_error_data}, {min_error:.3})",
	             xy=(min_error_data + max(study_for_data)/50, min_error - max(error)/50))

	plt.grid()
	plt.title(f"Error for a variation of {study_for}")
	plt.xlabel(study_for)
	plt.ylabel("Error")
	error_plot.savefig(f"{RESULTS_DIR}{study_for}_error.png")

	accuracy_plot = plt.figure(3)
	plt.plot(study_for_data, accuracy, marker='o')
	plt.grid()
	plt.title("Accuracy")
	plt.xlabel(study_for)
	plt.ylabel("Accuracy")
	accuracy_plot.savefig(f"{RESULTS_DIR}{study_for}_accuracy.png")


if __name__ == "__main__":
	main()
