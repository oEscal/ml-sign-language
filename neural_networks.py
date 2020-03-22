import time

import numpy as np

from utils import sigmoid, sigmoid_gradient, read_file


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):
	"""nn_params contains the parameters unrolled into a vector

	compute the cost and gradient of the neural network
	"""
	# Reshape nn_params back into the parameters Theta1 and Theta2
	theta1 = nn_params[:((input_layer_size + 1) * hidden_layer_size)].reshape(hidden_layer_size, input_layer_size + 1)
	theta2 = nn_params[((input_layer_size + 1) * hidden_layer_size):].reshape(num_labels, hidden_layer_size + 1)

	m = X.shape[0]
	J = 0
	X = np.hstack((np.ones((m, 1)), X))
	y10 = np.zeros((m, num_labels))

	a1 = sigmoid(X @ theta1.T)
	a1 = np.hstack((np.ones((m, 1)), a1))  # hidden layer
	a2 = sigmoid(a1 @ theta2.T)  # output layer

	for i in range(1, num_labels + 1):
		y10[:, i - 1][:, np.newaxis] = np.where(y == i, 1, 0)
	for j in range(num_labels):
		J = J + sum(-y10[:, j] * np.log(a2[:, j]) - (1 - y10[:, j]) * np.log(1 - a2[:, j]))

	cost = 1 / m * J
	reg_J = cost + Lambda / (2 * m) * (np.sum(theta1[:, 1:] ** 2) + np.sum(theta2[:, 1:] ** 2))

	# Implement the backpropagation algorithm to compute the gradients

	grad1 = np.zeros((theta1.shape))
	grad2 = np.zeros((theta2.shape))

	for i in range(m):
		xi = X[i, :]  # 1 X 401
		a1i = a1[i, :]  # 1 X 26
		a2i = a2[i, :]  # 1 X 10
		d2 = a2i - y10[i, :]
		d1 = theta2.T @ d2.T * sigmoid_gradient(np.hstack((1, xi @ theta1.T)))
		grad1 = grad1 + d1[1:][:, np.newaxis] @ xi[:, np.newaxis].T
		grad2 = grad2 + d2.T[:, np.newaxis] @ a1i[:, np.newaxis].T

	grad1 = 1 / m * grad1
	grad2 = 1 / m * grad2

	grad1_reg = grad1 + (Lambda / m) * np.hstack((np.zeros((theta1.shape[0], 1)), theta1[:, 1:]))
	grad2_reg = grad2 + (Lambda / m) * np.hstack((np.zeros((theta2.shape[0], 1)), theta2[:, 1:]))

	return cost, grad1, grad2, reg_J, grad1_reg, grad2_reg


def randInitializeWeights(L_in, L_out):
	"""
	randomly initializes the weights of a layer with L_in incoming connections and L_out outgoing connections.
	"""

	epi = (6 ** 1 / 2) / (L_in + L_out) ** 1 / 2

	W = np.random.rand(L_out, L_in + 1) * (2 * epi) - epi

	return W


def gradientDescentnn(X, y, initial_nn_params, alpha, num_iters, Lambda, input_layer_size, hidden_layer_size,
                      num_labels):
	"""
	Take in numpy array X, y and theta and update theta by taking num_iters gradient steps
	with learning rate of alpha

	return theta and the list of the cost of theta during each iteration
	"""
	Theta1 = initial_nn_params[:((input_layer_size + 1) * hidden_layer_size)].reshape(hidden_layer_size,
	                                                                                  input_layer_size + 1)
	Theta2 = initial_nn_params[((input_layer_size + 1) * hidden_layer_size):].reshape(num_labels, hidden_layer_size + 1)

	J_history = []

	for i in range(num_iters):
		nn_params = np.append(Theta1.flatten(), Theta2.flatten())
		cost, grad1, grad2 = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)[
		                     3:]
		Theta1 = Theta1 - (alpha * grad1)
		Theta2 = Theta2 - (alpha * grad2)
		J_history.append(cost)

	nn_paramsFinal = np.append(Theta1.flatten(), Theta2.flatten())
	return nn_paramsFinal, J_history


if __name__ == "__main__":
	time_init = time.time()

	X, y = read_file("dataset/sign_mnist_train.csv")
	m = X.shape[0]

	input_layer_size = X.shape[1] + 1
	hidden_layer_size = 25

	X = np.append(np.ones((m, 1)), X, axis=1)

	alpha = 1  # learning rate
	num_iterations = 1000
	Lambda = 0.1
	num_labels = 10

	initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
	initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
	initial_nn_params = np.append(initial_Theta1.flatten(), initial_Theta2.flatten())

	nnTheta, nnJ_history = gradientDescentnn(X, y, initial_nn_params, alpha, num_iterations, Lambda, input_layer_size,
	                                         hidden_layer_size, num_labels)

	Theta1 = nnTheta[:((input_layer_size + 1) * hidden_layer_size)].reshape(hidden_layer_size, input_layer_size + 1)
	Theta2 = nnTheta[((input_layer_size + 1) * hidden_layer_size):].reshape(num_labels, hidden_layer_size + 1)

	np.save("theta1", Theta1)
	np.save("theta1", Theta2)
	print(time.time() - time_init)

	# pred3 = predict(Theta1, Theta2, X)
	# print("Training Set Accuracy:", sum(pred3[:, np.newaxis] == y)[0] / m * 100, "%")
