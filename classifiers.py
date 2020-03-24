from sklearn import svm
from pathlib import Path
import pickle
from enum import Enum, unique
from sklearn.linear_model import LogisticRegression as LogisticRegression_sklearn
from sklearn.metrics import classification_report
from abc import abstractmethod, ABCMeta
from sklearn.neural_network import MLPClassifier
from os import listdir
import numpy as np


def save_classifier_object(classifier, file_name):
    path = f"classifiers_/{classifier.name}_{classifier.variation_param}"
    Path(path).mkdir(parents=True, exist_ok=True)  # if folder doesnt exists, crete one

    with open(f"{path}/{file_name}", 'wb') as output:
        pickle.dump(classifier, output)


def save_best_param(classifier_list):
    best_classifier = sorted(classifier_list, key=lambda c: (
        c.params[ErrorLabel.CV], c.params[ErrorLabel.TEST], c.params[ErrorLabel.TRAIN]))[0]
    path = f"best_param_classifiers/{best_classifier.name}"

    Path(path).mkdir(parents=True, exist_ok=True)  # if folder doesnt exists, crete one
    with open(f"{path}/{best_classifier.variation_param}", 'wb') as output:
        pickle.dump(best_classifier, output)


def get_classifiers(path):
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
    def __init__(self, name, classifier, variation_param):
        self.name = name
        self.classifier = classifier
        self.report = None
        self.params = {}
        self.variation_param = variation_param

    def train_model(self, x, y):
        print("Training model...")
        self.classifier.fit(x, y)

    def predict(self, x):
        print("Predicting...")
        return self.classifier.predict(x)

    def measure_error(self, label, x, y):
        print(f"Calculating error -> {label}...")
        l = np.where(self.predict(x) != y.T, 1, 0)
        list_sum = np.sum(l)
        list_size = max(l.shape)
        self.params[label] = list_sum / list_size
        print(f"{label}->{self.params[label]}\n")

    def generate_report(self, x, y):
        print("Generating report:")
        self.report = classification_report(y, self.predict(x))
        print(self.report + '\n')

    def save_classifier(self, file_name=None):
        save_classifier_object(self, file_name if file_name is not None else self.name)

    def measure_all_error(self, x_train, y_train, x_cv, y_cv, x_test, y_test):
        self.measure_error(ErrorLabel.TRAIN, x_train, y_train)
        self.measure_error(ErrorLabel.CV, x_cv, y_cv)
        self.measure_error(ErrorLabel.TEST, x_test, y_test)

    def startup(self, x_train, y_train, x_cv, y_cv, x_test, y_test):
        print(f"Starting {self.name}")

        self.train_model(x_train, y_train)

        self.measure_all_error(x_train, y_train, x_cv, y_cv, x_test, y_test)

        self.generate_report(x_test, y_test)

        self.save_classifier()

    @abstractmethod
    def plot(self, x, y):
        pass

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

    def plot(self, x, y):
        pass

    def __str__(self):
        return super().__str__() + f"C->{self.C}\tdegree->{self.degree}\n"


class NeuralNetwork(Classifier):
    def __init__(self, alpha, hidden_layer_sizes, max_iter, variation_param, verbose=False):
        self.alpha = alpha
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.variation_param = variation_param
        super().__init__(self.__class__.__name__,
                         MLPClassifier(alpha=self.alpha, hidden_layer_sizes=self.hidden_layer_sizes,
                                       max_iter=max_iter, verbose=verbose), self.variation_param)

    def save_classifier(self, file_name=None):
        super().save_classifier(
            file_name if file_name is not None else f'{self.name}_alpha_{self.alpha}_'
                                                    f'hidden_size_{self.hidden_layer_sizes}_max_iter_{self.max_iter}')

    def plot(self, x, y):
        pass

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

    def plot(self, x, y):
        pass

    def __str__(self):
        return super().__str__() + f"C->{self.C}\tmax_iter->{self.max_iter}\n"
