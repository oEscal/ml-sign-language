from classifiers import *
from utils import read_file, convert_image


def f(filename, x, y):
    pixel_size = x.shape[-1]
    headlines = ['label']
    for i in range(1, pixel_size + 1):
        headlines.append(f'pixel{i}')

    np.savetxt(filename, np.c_[y, x], delimiter=',', header=','.join(headlines))


def main():
    lambdas = (0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000)
    degrees = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    alphas = (0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000)
    iterations = [200, 500, 1000, 2000]

    """
    x_train, y_train = read_file('dataset/sign_mnist_train.csv')
    x_cv, y_cv = read_file('dataset/sign_mnist_cv.csv')
    x_test, y_test = read_file('dataset/sign_mnist_test.csv')

    x = np.concatenate((x_train, x_cv, x_test))
    y = np.concatenate((y_train, y_cv, y_test))

    f('big_data2.csv', x, y)
    
        
    x_data, y_data = read_file('big_data2.csv', shuffle=True)

    data_size = x_data.shape[0]
    train_size = int(data_size * 0.6)
    cv_size = train_size + int(data_size * 0.2)

    x_train, y_train = x_data[:train_size], y_data[:train_size]
    x_cv, y_cv = x_data[train_size:cv_size], y_data[train_size:cv_size]
    x_test, y_test = x_data[cv_size:], y_data[cv_size:]

    f('dataset/merged_train_set.csv', x_train, y_train)
    f('dataset/merged_cv_set.csv', x_cv, y_cv)
    f('dataset/merged_test_set.csv', x_test, y_test)
    """

    x_train, y_train = read_file('dataset/merged_train_set.csv')
    x_cv, y_cv = read_file('dataset/merged_cv_set.csv')
    x_test, y_test = read_file('dataset/merged_test_set.csv')

    print(x_train[0][0])

    x_train = x_train / 255
    x_cv = x_cv / 255
    x_test = x_test / 255
    
    classifiers = {

        'PolynomialSvm': [PolynomialSvm(0.1, degree_value, "degree", verbose=True) for degree_value in degrees],
        # 000.1
        # max_iter

        # 'Neural Network': [
        #    NeuralNetwork(X=x_train, y=y_train.ravel(), alpha=0.01, Lambda=0.0, batch_size=10,
        #                  activation="logistic",
        #                  iterations=1000, hidden_layer_sizes=(100, 100,), verbose=True)],
        # 'Logistic Regression': [LogisticRegression(1000, iteration_value, "iter") for iteration_value in iterations],

        # 'KNeighborsClassifier': [KNeighborsClassifier(3)],
        # 'GaussianProcessClassifier': [GaussianProcessClassifier(1.0 * RBF(1.0))],
        # 'DecisionTreeClassifier': [DecisionTreeClassifier(max_depth=5)],
        # 'RandomForestClassifier': [RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)],
        # 'GaussianNB': [GaussianNB()],

    }

    for classifier_name, classifier_list in classifiers.items():
        print(classifier_name)
        for classifier in classifier_list:
            classifier.train()
            classifier.save_classifier()
            print(classifier.accuracy(x_train, y_train))
            print(classifier.accuracy(x_cv, y_cv))
            print(classifier.accuracy(x_test, y_test))
            print(classifier.confusion_matrix(x_test, y_test, "confusion_matrix_test_set"))


if __name__ == '__main__':
    main()
