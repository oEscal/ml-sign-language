from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from utils import *

import pickle


def main():
    classifiers = {
        'polynomial_SVM': '',
        'rbf_SVM': '',
        'Neural Network': '',
        'KNeighbors': ''
    }

    x_train, y_train = read_file('dataset/sign_mnist_train.csv')
    x_test, y_test = read_file('dataset/sign_mnist_test.csv')

    # x_train, x_test = rescale_dataset(x_train / 255, factor=0.5), rescale_dataset(x_test / 255, factor=0.5)

    print("Data rescaled")

    svc = svm.SVC(random_state=8, degree=3)
    print(dir(svc))

    svc = KNeighborsClassifier(3)
    grid_search = GridSearch(svc)

    params, score, estimator = grid_search.fit_data(x_train, y_train.ravel())

    print("Score -> ", score)

    with open('params.pickle', 'wb') as output:
        pickle.dump(params, output)

    with open('estimator.pickle', 'wb') as output:
        pickle.dump(estimator, output)

    with open('score.pickle', 'wb') as output:
        pickle.dump(score, output)
    with open('estimator.pickle', 'rb') as output:
        best_svc = pickle.load(output)

    print("The training accuracy is: ")
    print(accuracy_score(y_train, best_svc.predict(x_train)))
    print("The test accuracy is: ")
    print(accuracy_score(y_test, best_svc.predict(x_test)))


main()
