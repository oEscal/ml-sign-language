import argparse
import pickle
import random
import time
import json

from classifiers import NeuralNetwork, Classifier
from utils import read_file

import numpy as np

PATH_SAVE = "results/neural_networks/"


def main(args):
	file_id = args.file_id

	X, y = read_file("dataset/merged_train_set.csv")

	# layer sizes
	hidden_layer_size = args.hidden_layer_size

	alpha = args.alpha  # learning rate
	Lambda = args.Lambda
	num_iterations = args.num_iterations
	activation = args.activation
	batch_size = args.batch_size
	
	from_classifier_id = args.from_classifier_id

	time_init = time.time()
	if from_classifier_id:
		X_cv, y_cv = read_file("dataset/merged_cv_set.csv")
		X, y = np.append(X, X_cv, axis=0), np.append(y, y_cv, axis=0)

		with open(f"{PATH_SAVE}classifier_id{from_classifier_id}", 'rb') as file:
			classifier: Classifier = pickle.load(file)

		time_init = time.time()
		classifier.X = X
		classifier.y = y.ravel()
		classifier.set_new_number_iter(num_iterations)
		classifier.train(from_previous=True)
	else:
		classifier = NeuralNetwork(X=X, y=y.ravel(), alpha=alpha, Lambda=Lambda, hidden_layer_sizes=(hidden_layer_size,),
		                           activation=activation, iterations=num_iterations, batch_size=batch_size)
		classifier.train()

	classifier.save_classifier(f"{PATH_SAVE}{args.classifier_file}_id{file_id}")

	save_file = open(f"{PATH_SAVE}{args.time_file}", 'a')
	save_file.write(json.dumps({
		'hidden_layer1': hidden_layer_size,
		'alpha': alpha,
		'lambda': Lambda,
		'num_iterations': num_iterations,
		'file_id': file_id,
		'batch_size': batch_size,
		'activation': activation,
		'from_classifier_id': from_classifier_id,
		'time': time.time() - time_init
	}))
	save_file.write("\n")
	save_file.close()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--Lambda", type=float, default=0)
	parser.add_argument("--alpha", type=float, default=1.0)
	parser.add_argument("--num_iterations", type=int, default=200)
	parser.add_argument("--hidden_layer_size", type=int, default=50)
	parser.add_argument("--batch_size", type=int, default=10)
	parser.add_argument("--activation", type=str, default="logistic")

	parser.add_argument("--classifier_file", type=str, default="classifier")
	parser.add_argument("--time_file", type=str, default="time.json")

	parser.add_argument("--file_id", type=int, default=random.randint(100, 100000))

	parser.add_argument("--from_classifier_id", type=int, default=None)

	args = parser.parse_args()
	main(args)
