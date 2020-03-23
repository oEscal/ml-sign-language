from classifiers import *
from utils import read_file


def main():
    lambdas = (0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000)
    degrees = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    alphas = (0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000)
    iterations = [200, 500, 1000, 2000]

    classifiers = {

        'PolynomialSvm': [PolynomialSvm(1 / lambda_value, degrees[0], "C") for lambda_value in lambdas],
        'Neural Network': [NeuralNetwork(alpha, (50,), iterations[0], "alpha") for alpha in alphas],
        'Logistic Regression': [LogisticRegression(1 / lambda_value, iterations[0], "C")
                                for lambda_value in lambdas],

        # 'KNeighborsClassifier': [KNeighborsClassifier(3)],
        # 'GaussianProcessClassifier': [GaussianProcessClassifier(1.0 * RBF(1.0))],
        # 'DecisionTreeClassifier': [DecisionTreeClassifier(max_depth=5)],
        # 'RandomForestClassifier': [RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)],
        # 'GaussianNB': [GaussianNB()],

    }

    x_train, y_train = read_file('dataset/sign_mnist_train.csv')
    x_cv, y_cv = read_file('dataset/sign_mnist_cv.csv')
    x_test, y_test = read_file('dataset/sign_mnist_test.csv')

    # x_train, x_cv, x_test = rescale_dataset(x_train, factor=0.5), \
    #                        rescale_dataset(x_cv, factor=0.5), rescale_dataset(x_test, factor=0.5)

    for classifier_name, classifier_list in classifiers.items():
        print(classifier_name)
        for classifier in classifier_list:
            classifier.startup(x_train, y_train.ravel(), x_cv, y_cv.ravel(), x_test, y_test.ravel())

        save_best_param(classifier_list)


if __name__ == '__main__':
    main()
