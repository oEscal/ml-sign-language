import argparse
import json
import pickle
import sys

import matplotlib.pyplot as plt

from classifiers import Classifier
from utils import read_file

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

RESULTS_DIR = "results/neural_networks/"


def print_latex_list_to_table_line(orig_list, end_line=False):
	print(*orig_list, sep='&', end='\\\\\n' + (r'\hline' if end_line else ''))


def main(args):
	study_for = args.study
	time_file_name = f"time.json"

	with open(f"{RESULTS_DIR}{time_file_name}") as file:
		all_data = file.readlines()
		all_data = sorted([json.loads(line.replace('\n', '')) for line in all_data], key=lambda x: x[study_for])

	X_train, y_train = read_file("dataset/merged_train_set.csv")
	X_cv, y_cv = read_file("dataset/merged_cv_set.csv")
	X_test, y_test = read_file("dataset/merged_test_set.csv")

	x_label = last_part_title = study_for.replace('_', ' ')
	if study_for == "hidden_layer1":
		x_label = "number of nodes on the hidden layer"
		last_part_title = f"the {x_label}"
	elif study_for == "num_iterations":
		x_label = "number of iterations"
		last_part_title = f"the {x_label}"

	accuracy = []
	error = []
	error_train = []
	times = []
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
		times.append(data['time'])

		# print info
		for i in range(10):
			print()

		title = f"{study_for}={data[study_for]}"
		print(f"{title:^70}")
		print(f" -> Accuracy: {current_accuracy}\n")
		print(f" -> Error: {current_error}")

		plt.plot(classifier.history.loss_curve_)

		# generate latex tables
		if args.generate_tables_latex:
			confusion_matrix = classifier.confusion_matrix(X_test, y_test)
			accuracy_per_class = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)
			recall = classifier.recall(X_test, y_test)
			precision = classifier.precision(X_test, y_test)
			f1_score = classifier.f1_score(X_test, y_test)

			total_accuracy = classifier.accuracy(X_test, y_test)
			total_recall = classifier.recall(X_test, y_test, average='macro')
			total_precision = classifier.precision(X_test, y_test, average='macro')
			total_f1_score = classifier.f1_score(X_test, y_test, average='macro')

			classes = classifier.history.classes_
			number_classes = len(classes)

			original_stdout = sys.stdout
			with open(f"{RESULTS_DIR}{study_for}_per_class_metrics{id}.tex", 'w') as file:
				sys.stdout = file
				print(r"\begin{tabular}{l c c c c}")
				print(r"Class & Accuracy & Recall & Precision & F1 Score\\ \hline")
				for index in range(number_classes):
					print(f"{classes[index]} & {accuracy_per_class[index]:.3} & {recall[index]:.3} & "
					      f"{precision[index]:.3} & {f1_score[index]:.3}\\\\")
				print(r"\hline" + f"\nMacro Average & {total_accuracy:.3} & {total_recall:.3} & "
					      f"{total_precision:.3} & {total_f1_score:.3}\\\\")
				print(r"\end{tabular}")

			with open(f"{RESULTS_DIR}{study_for}_confusion_matrix{id}.tex", 'w') as file:
				sys.stdout = file
				print(r"\begin{tabular}{l|" + "c "*number_classes + "}")
				print("Class&", end='')
				print_latex_list_to_table_line(classes, end_line=True)
				for index in range(number_classes):
					print_latex_list_to_table_line([f"{classes[index]}"] + list(confusion_matrix[index]))
				print(r"\end{tabular}")

			sys.stdout = original_stdout

	# make plots
	study_for_data = [data[study_for] for data in all_data]

	cost_plot = plt.figure(1)
	if args.retrained:
		plt.axvline(x=1001, ymin=0, ymax=max(classifier.history.loss_curve_), color='k', ls='dotted')
		plt.text(x=1050, y=3, s='Beginning of re-training')
	plt.legend([f"{study_for if study_for != 'hidden_layer1' else 'size'}={data[study_for]}" for data in all_data])
	plt.title(f"Cost function for {last_part_title}")
	plt.xlabel("Number of iterations")
	plt.ylabel(r"$J(\theta)$")
	cost_plot.savefig(f"{RESULTS_DIR}{study_for}_cost.png")

	time_plot = plt.figure(2)
	plt.plot(study_for_data, times, marker='o', color="blue")
	plt.grid()
	plt.title(f"Execution time for {last_part_title}")
	plt.xlabel(x_label)
	plt.ylabel("Execution time (s)")
	time_plot.savefig(f"{RESULTS_DIR}{study_for}_time.png")

	error_plot = plt.figure(3)
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
	plt.title(f"Error for a variation of {last_part_title}")
	plt.xlabel(x_label)
	plt.ylabel("Error")
	error_plot.savefig(f"{RESULTS_DIR}{study_for}_error.png")

	accuracy_plot = plt.figure(4)
	plt.plot(study_for_data, accuracy, marker='o', color="blue")
	plt.grid()
	plt.title("Accuracy")
	plt.xlabel(x_label)
	plt.ylabel("Accuracy")
	accuracy_plot.savefig(f"{RESULTS_DIR}{study_for}_accuracy.png")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--study", type=str, default="alpha")
	parser.add_argument("--retrained", type=bool, default=False)
	parser.add_argument("--generate_tables_latex", type=bool, default=False)

	args = parser.parse_args()
	main(args)
