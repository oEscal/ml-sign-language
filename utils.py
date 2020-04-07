import math
import pickle
from numbers import Number
from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import rescale
from pathlib import Path
import os

from sklearn.base import is_classifier, clone
from sklearn.metrics import check_scoring
from sklearn.model_selection import learning_curve, check_cv
from sklearn.model_selection._validation import _fit_and_score
from sklearn.utils import indexable, Parallel, delayed


def read_file(path_file: str, shuffle=False) -> (np.ndarray, np.ndarray):
    """Function to read datafile and returns a tuple with the following format: (X, y). X represents all the features
        and y represents all the outputs for each data example
    """
    data = pd.read_csv(path_file, header=None, skiprows=1).values
    if shuffle:
        np.random.shuffle(data)

    return data[:, 1:], data[:, 0].reshape(data.shape[0], 1)


def represent_data_graphically(data: np.ndarray, file_save: str, rows: int = 10, cols: int = 10):
    data_image_size = int(math.sqrt(len(data[0, :])))
    data_len = len(data)

    fig, axis = plt.subplots(rows, cols, figsize=(data_image_size, data_image_size))
    for row in range(rows):
        for col in range(cols):
            example_id = np.random.randint(data_len)
            axis[row, col].imshow(data[example_id, :].reshape(data_image_size, data_image_size, order="F"))
    plt.savefig(file_save)


def sigmoid(z: Union[Number, np.ndarray]) -> np.ndarray:
    """return the sigmoid of z
    """
    return .5 * (1 + np.tanh(.5 * z))


# Computes the gradient of sigmoid function
def sigmoid_gradient(z):
    """computes the gradient of the sigmoid function
    """
    sigmoid_val = sigmoid(z)
    return sigmoid_val * (1 - sigmoid_val)


def rescale_image(data, factor):
    data_size = int(data.shape[0] ** 0.5)
    img = rescale(data.reshape(data_size, data_size), factor, mode='reflect')
    x = img.shape[0] ** 2
    return img.reshape(x, 1).ravel()


def rescale_dataset(dataset, factor=0.75):
    rescaled_data = []
    for img in dataset:
        rescaled_data.append(rescale_image(img, factor))
    return np.asarray(rescaled_data)


def plot_image(data):
    data_size = int(data.shape[0] ** 0.5)
    img = data.reshape(data_size, data_size)
    plt.imshow(img)
    plt.show()


def save_object(obj, file_name):
    base_file_name = os.path.basename(file_name)
    Path(file_name.replace(base_file_name, '')).mkdir(parents=True, exist_ok=True)
    with open(file_name, 'wb') as file:
        pickle.dump(obj, file)


def convert_image(data):
    k = np.where(data * 255 > 128, 1, 0)
    return k


def validation_curve(estimator, X, y, param_name, param_range, groups=None,
                     cv=None, scoring=None, n_jobs=None, pre_dispatch="all",
                     verbose=0, error_score=np.nan):
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorer = check_scoring(estimator, scoring=scoring)

    parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch,
                        verbose=verbose)
    out = parallel(delayed(_fit_and_score)(
        clone(estimator), X, y, scorer, train, test, verbose,
        parameters={param_name: v}, fit_params=None, return_train_score=True,
        error_score=error_score, return_estimator=True, return_times=True)
                   # NOTE do not change order of iteration to allow one time cv splitters
                   for train, test in cv.split(X, y, groups) for v in param_range)

    out = np.asarray(out)
    estimators = out[:, 4]
    out_scores = np.asarray(out[:, :2])
    fit_time = out[:, 2]
    score_time = out[:, 3]
    n_params = len(param_range)
    n_cv_folds = out_scores.shape[0] // n_params
    out_scores = out_scores.reshape(n_cv_folds, n_params, 2).transpose((2, 1, 0))

    return estimators, np.float64(out_scores[0]), np.float64(out_scores[1]), np.float64(fit_time), \
           np.float64(score_time)


def plot_validation_curve(train_scores, test_scores, title, xlabel, ylabel, param_range):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0.0, 1.1)

    plt.plot(param_range, train_scores, label="Training Score", color="blue")
    plt.plot(param_range, test_scores, label="Cross-validation score", color="orange")

    plt.legend(loc="best")
    plt.show()


def plot_time_per_parameter(fit_times, score_times, title, xlabel, ylabel, param_range):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    pass
