import argparse
import random
import time
import json

from classifiers import NeuralNetwork

from utils import read_file

PATH_SAVE = "results/neural_networks/"


def main(args):
	file_id = args.theta_file_id

	X, y = read_file("dataset/sign_mnist_train.csv")
	time_init = time.time()

	# layer sizes
	hidden_layer_size = 50

	alpha = args.alpha  # learning rate
	Lambda = args.Lambda
	num_iterations = args.num_iterations
	activation = "logistic"
	batch_size = 10

	classifier = NeuralNetwork(X=X, y=y.ravel(), alpha=alpha, Lambda=Lambda, hidden_layer_sizes=(hidden_layer_size,),
	                           activation=activation, max_iter=num_iterations, batch_size=batch_size)
	classifier.train()
	classifier.save_classifier(f"{PATH_SAVE}{args.classifier_file}_id{file_id}")
#
	save_file = open(f"{PATH_SAVE}{args.time_file}", 'a')
	save_file.write(json.dumps({
		'hidden_layer1': hidden_layer_size,
		'alpha': alpha,
		'lambda': Lambda,
		'num_iterations': num_iterations,
		'file_id': file_id,
		'batch_size': batch_size,
		'activation': activation,
		'time': time.time() - time_init
	}))
	save_file.write("\n")
	save_file.close()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--Lambda", type=float, default=0)
	parser.add_argument("--alpha", type=float, default=1.0)
	parser.add_argument("--num_iterations", type=int, default=200)

	parser.add_argument("--classifier_file", type=str, default="classifier")
	parser.add_argument("--time_file", type=str, default="time.json")

	parser.add_argument("--theta_file_id", type=int, default=random.randint(100, 100000))

	args = parser.parse_args()
	main(args)
