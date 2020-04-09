import sys
from enum import Enum, unique
from os import listdir
import logging
import json
import pickle

from sklearn import svm
from sklearn.linear_model import LogisticRegression as LogisticRegression_sklearn
from sklearn.metrics import classification_report, mean_squared_error, precision_score, confusion_matrix, \
    accuracy_score, log_loss
from abc import ABCMeta
from sklearn.neural_network import MLPClassifier
import numpy as np

from utils import save_object

logger = logging.getLogger("classifiers")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_classifiers(path):
    folders = [f for f in listdir(path)]
    classifiers = {}

    for folder_name in folders:
        folder_path = f"{path}/{folder_name}"
        if folder_name == 'others':
            continue
        classifiers[folder_name] = []
        for file_name in [f for f in listdir(folder_path)]:
            file_path = f"{folder_path}/{file_name}"

            with open(file_path, 'rb') as output:
                classifier = pickle.load(output)
                classifiers[folder_name].append(classifier)

    return classifiers


def save_best_classifiers(classifiers_list, file_name='best_classifiers'):
    best = sorted(classifiers_list, reverse=True,
                  key=lambda c: (c.params['valid_score'],
                                 c.params['Test set Accuracy'], c.params['Test set Accuracy']))[0]

    save_object(best, f'{file_name}/{best.name}/{best.variation_param}')

    return best


def pick_best_classier_param(folder_path):
    files = [f for f in listdir(folder_path)]
    classifiers_list = []
    for file_name in files:
        with open(f"{folder_path}/{file_name}", "rb") as output:
            classifiers_list.append(pickle.load(output))

    best_classifier: Classifier = sorted(classifiers_list, reverse=True,
                                         key=lambda c: (c.params['valid_score'], c.params['Test set Accuracy']))[0]

    return best_classifier, eval(f"best_classifier.classifier.{best_classifier.variation_param}")


class Classifier(metaclass=ABCMeta):
    def __init__(self, name, classifier, X: np.ndarray, y: np.ndarray, variation_param=None, nn=False):
        self.name = name
        self.classifier = classifier
        self.params = {}
        self.variation_param = variation_param

        self.X: np.ndarray = X
        self.y: np.ndarray = y

        self.history = None

        self.train_scores = None
        self.valid_scores = None

        self.nn = nn

    def __train_model(self):
        logger.info("Training model...")
        return self.classifier.fit(self.X, self.y)

    def __train_nn_model(self):
        logger.info("Training model...")
        return self.classifier._fit(self.X, self.y)

    def predict(self, x):
        logger.info("Predicting...")
        return self.classifier.predict(x)

    def predict_prob(self, x):
        logger.info("Predicting...")
        return self.classifier.predict_proba(x)

    def error(self, x, y):
        logger.info(f"Calculating error")
        return mean_squared_error(y, self.predict(x)) / 2

    def log_error(self, x, y):
        logger.info(f"Calculating error")
        return log_loss(y, self.predict_prob(x))

    def set_new_number_iter(self, iterations):
        self.classifier.max_iter = iterations

    def train(self, from_previous=False):
        if from_previous:
            self.classifier.warm_start = from_previous

        logger.info(f"Starting train: {self.name}")
        self.history = self.__train_model() if not self.nn else self.__train_nn_model()

    def save_classifier(self, file_name="classifier"):
        save_object(self, file_name)

    def save_history(self, file_name="history"):
        save_object(self.history, file_name)

    def generate_report(self, X, y):
        return classification_report(y_true=y, y_pred=self.predict(X))

    def precision(self, X, y, average=None):
        return precision_score(y_true=y, y_pred=self.predict(X), average=average, zero_division=1)

    def accuracy(self, X, y, label='accuracy'):
        self.params[label] = accuracy_score(y_true=y, y_pred=self.predict(X))
        return self.params[label]

    def confusion_matrix(self, X, y, label='confusion_matrix'):
        self.params[label] = confusion_matrix(y_true=y, y_pred=self.predict(X))
        return self.params[label]

    def update_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value
        logger.info(f"Params updated with {kwargs}")
        return self.params

    def save_report(self, file_name="report.json"):
        with open(file_name, 'w') as file:
            file.write(json.dumps(self.generate_report()))
        logger.info(f"Report saved into file: {file_name}")

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Classifier : {self.name} ->  {self.params} -> Best value for: {self.variation_param}\n"


class PolynomialSvm(Classifier):
    def __init__(self, classifier, X, y, variation_param):
        self.X = X
        self.y = y
        # self.C = C
        # self.degree = degree
        self.classifier = classifier
        self.variation_param = variation_param
        super().__init__(self.__class__.__name__, classifier, self.X, self.y, self.variation_param)

    def save_classifier(self, file_name=None):
        super().save_classifier(
            file_name if file_name is not None else f'classifiers/{self.name}_{self.variation_param}/'
                                                    f'{eval(f"self.classifier.{self.variation_param}")}.classifier')

    def __str__(self):
        return super().__str__()  # + f"C->{self.C}\tdegree->{self.degree}\n"


class NeuralNetwork(Classifier):
    def __init__(self, X, y, alpha, Lambda, hidden_layer_sizes, iterations, activation, batch_size, solver="sgd",
                 variation_param=None, verbose=False):
        self.alpha = alpha
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = iterations
        self.variation_param = variation_param
        super().__init__(self.__class__.__name__,
                         MLPClassifier(alpha=Lambda, learning_rate_init=alpha, activation=activation,
                                       hidden_layer_sizes=self.hidden_layer_sizes, solver=solver,
                                       max_iter=iterations, verbose=verbose, n_iter_no_change=iterations,
                                       batch_size=batch_size),
                         X, y, variation_param=self.variation_param, nn=True)

    def save_classifier(self, file_name=None):
        super().save_classifier(
            file_name if file_name is not None else f'classifiers/{self.name}_alpha_{self.alpha}_'
                                                    f'hidden_size_{self.hidden_layer_sizes}_max_iter_{self.max_iter}')

    def __str__(self):
        return super().__str__() + f"alpha->{self.alpha}\thidden_layer_sizes->{self.hidden_layer_sizes}\tmax_iter->{self.max_iter}\n"


class LogisticRegression(Classifier):
    def __init__(self, classifier, X, y, variation_param):
        self.X = X
        self.y = y
        # self.C = C
        # self.max_iter = max_iter
        self.variation_param = variation_param
        # LogisticRegression_sklearn(C=C, verbose=verbose, max_iter=max_iter, n_jobs=-1),
        super().__init__(self.__class__.__name__, classifier, self.X, self.y, self.variation_param)

    def save_classifier(self, file_name=None):
        super().save_classifier(
            file_name if file_name is not None else f'classifiers/{self.name}_{self.variation_param}/'
                                                    f'{eval(f"self.classifier.{self.variation_param}")}.classifier')  # _C_{self.C}_max_iter_{self.max_iter}')

    def __str__(self):
        return super().__str__()  # + f"C->{self.C}\tmax_iter->{self.max_iter}\n"


class RbfSvm(Classifier):
    def __init__(self, classifier, X, y, variation_param):
        self.X = X
        self.y = y
        # self.C = C
        # self.max_iter = max_iter
        self.variation_param = variation_param
        # LogisticRegression_sklearn(C=C, verbose=verbose, max_iter=max_iter, n_jobs=-1),
        super().__init__(self.__class__.__name__, classifier, self.X, self.y, self.variation_param)

    def save_classifier(self, file_name=None):
        super().save_classifier(
            file_name if file_name is not None else f'classifiers/{self.name}_{self.variation_param}/'
                                                    f'{eval(f"self.classifier.{self.variation_param}")}.classifier')  # _C_{self.C}_max_iter_{self.max_iter}')

    def __str__(self):
        return super().__str__()  # + f"C->{self.C}\tmax_iter->{self.max_iter}\n"
