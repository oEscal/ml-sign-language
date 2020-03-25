import sys
from enum import Enum, unique
from os import listdir
import logging
import json
import pickle

from sklearn import svm
from sklearn.linear_model import LogisticRegression as LogisticRegression_sklearn
from sklearn.metrics import classification_report, mean_squared_error
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


def get_bests_classifier(path):
    folders = [f for f in listdir(path)]
    classifiers = {}

    for folder_name in folders:
        folder_path = f"{path}/{folder_name}"
        classifiers[folder_name] = []

        for file_name in [f for f in listdir(folder_path)]:
            file_path = f"{folder_path}/{file_name}"
            
            with open(file_path, 'rb') as output:
                classifier = pickle.load(output)
                classifiers[folder_name].append(classifier)

    return classifiers


@unique
class ErrorLabel(Enum):
    TEST = "Test set error"
    CV = "Cross validation error"
    TRAIN = "Train set error"

    def __str__(self):
        return self.value


class Classifier(metaclass=ABCMeta):
    def __init__(self, name, classifier: MLPClassifier, X: np.ndarray, y: np.ndarray, variation_param=None):
        self.name = name
        self.classifier = classifier
        self.params = {}
        self.variation_param = variation_param

        self.X: np.ndarray = X
        self.y: np.ndarray = y

        self.history: MLPClassifier = None

    def __train_model(self, x, y):
        logger.info("Training model...")
        return self.classifier.fit(x, y)

    def predict(self, x):
        logger.info("Predicting...")
        return self.classifier.predict(x)

    def error(self, x, y):
        logger.info(f"Calculating error")
        return mean_squared_error(y, self.predict(x)) / 2

    def train(self):
        logger.info(f"Starting train: {self.name}")
        self.history = self.__train_model(self.X, self.y)

    def save_classifier(self, file_name="classifier"):
        save_object(self, file_name)

    def save_history(self, file_name="history"):
        save_object(self.history, file_name)

    def generate_report(self):
        return classification_report(y_true=self.y, y_pred=self.predict(self.X))

    def save_report(self, file_name="report.json"):
        with open(file_name, 'w') as file:
            file.write(json.dumps(self.generate_report()))
        logger.info(f"Report saved into file: {file_name}")

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Classifier : {self.name} ->  {self.params} -> Best value for: {self.variation_param}\n"


class PolynomialSvm(Classifier):
    def __init__(self, C, degree, variation_param, verbose=False):
        self.C = C
        self.degree = degree
        self.variation_param = variation_param
        super().__init__(self.__class__.__name__,
                         svm.SVC(kernel='poly', C=self.C, probability=True, degree=self.degree, verbose=verbose),
                         self.variation_param)

    def save_classifier(self, file_name=None):
        super().save_classifier(
            file_name if file_name is not None else f'{self.name}_C_{self.C}_degree_{self.degree}')

    def __str__(self):
        return super().__str__() + f"C->{self.C}\tdegree->{self.degree}\n"


class NeuralNetwork(Classifier):
    def __init__(self, X, y, alpha, Lambda, hidden_layer_sizes, max_iter, activation, solver="sgd", variation_param=None,
                 verbose=False):
        self.alpha = alpha
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.variation_param = variation_param
        super().__init__(self.__class__.__name__,
                         MLPClassifier(alpha=Lambda, learning_rate_init=alpha, activation=activation,
                                       hidden_layer_sizes=self.hidden_layer_sizes, solver=solver,
                                       max_iter=max_iter, verbose=verbose), X, y, self.variation_param)

    def save_classifier(self, file_name=None):
        super().save_classifier(
            file_name if file_name is not None else f'{self.name}_alpha_{self.alpha}_'
                                                    f'hidden_size_{self.hidden_layer_sizes}_max_iter_{self.max_iter}')

    def __str__(self):
        return super().__str__() + f"alpha->{self.alpha}\thidden_layer_sizes->{self.hidden_layer_sizes}\tmax_iter->{self.max_iter}\n"


class LogisticRegression(Classifier):
    def __init__(self, C, max_iter, variation_param, verbose=False):
        self.C = C
        self.max_iter = max_iter
        self.variation_param = variation_param
        super().__init__(self.__class__.__name__,
                         LogisticRegression_sklearn(C=C, verbose=verbose, max_iter=max_iter, n_jobs=-1),
                         self.variation_param)

    def save_classifier(self, file_name=None):
        super().save_classifier(
            file_name if file_name is not None else f'{self.name}_C_{self.C}_max_iter_{self.max_iter}')

    def __str__(self):
        return super().__str__() + f"C->{self.C}\tmax_iter->{self.max_iter}\n"

    """
    class GaussianSvm:
        def __init__(self, C, gamma):
            self.svc = svm.SVC(kernel='rbf', C=C, probability=True, gamma=gamma, verbose=2)
            self.train_score = 0
            self.test_score = 0
            self.cross_validation_score = 0
            self.report = None
    
        def train_model(self, x, y, train_score=True):
            logger.info("Training model")
            self.svc.fit(x, y)
    
            if train_score:
                self.train_score = accuracy_score(y, self.svc.predict(x))
    
        def cross_validation(self, x, y):
            logger.info("Cross validation")
            self.cross_validation_score = accuracy_score(y, self.svc.predict(x))
    
        def predict(self, x, y):
            logger.info("Predict")
            predict_list = self.svc.predict(x)
            self.test_score = accuracy_score(y, predict_list)
            self.report = classification_report(y, predict_list)
    
        def save_report(self, file_name=None):
            save_classifier(self, file_name if file_name is not None else self.__class__.__name__)
    
        def __str__(self):
            return f"{self.__class__.__name__}\n\tTrain Score: {self.train_score}\n\tCv Score: {self.cross_validation_score}\n\t" \
                   f"Train Score: {self.test_score}\nReport: {self.report}\n"
    """
