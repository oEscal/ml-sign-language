from classifiers import *
from utils import read_file, plot_validation_curve
from utils import validation_curve

from sklearn import svm
from sklearn.model_selection import PredefinedSplit


def f(filename, x, y):
    pixel_size = x.shape[-1]
    headlines = ['label']
    for i in range(1, pixel_size + 1):
        headlines.append(f'pixel{i}')

    np.savetxt(filename, np.c_[y, x], delimiter=',', header=','.join(headlines))


def set_validation_score_and_curve(classifier, x_train, y_train, x_cv, y_cv, x_test, y_test, parameter,
                                   parameter_values, classifier_class):
    data_x, data_y = np.concatenate((x_train, x_cv)), np.concatenate((y_train, y_cv))

    train_indices = np.full((x_train.shape[0],), -1, dtype=int)
    cv_indices = np.full((x_cv.shape[0],), 0, dtype=int)
    ps = PredefinedSplit(np.append(train_indices, cv_indices))

    estimators_svm, train_scores_svm, valid_scores_svm, fit_times, score_times = validation_curve(
        classifier, data_x, data_y.ravel(), parameter, parameter_values, cv=ps, n_jobs=-1)

    for i in range(estimators_svm.shape[0]):
        classifier = estimators_svm[i]
        train_score = train_scores_svm[i]
        valid_score = valid_scores_svm[i]
        fit_time = fit_times[i]
        score_time = score_times[i]

        c: Classifier = eval(classifier_class)
        c.update_params(train_score=train_score, valid_score=valid_score,
                        fit_time=fit_time, score_time=score_time)

        c.accuracy(x_test, y_test, "Test set Accuracy")
        c.confusion_matrix(x_cv, y_cv, "CV confusion matrix")
        c.confusion_matrix(x_test, y_test, "Test confusion matrix")
        c.save_classifier()


def main():
    C = (0.001, 0.002, 0.01, 0.02, 0.1, 0.2, 1, 5, 10, 50, 100, 500, 1000)
    degrees = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    alphas = (0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000)
    iterations = [200, 500, 1000, 2000]

    x_train, y_train = read_file('dataset/merged_train_set.csv')
    x_cv, y_cv = read_file('dataset/merged_cv_set.csv')
    x_test, y_test = read_file('dataset/merged_test_set.csv')

    x_train = x_train / 255
    x_cv = x_cv / 255
    x_test = x_test / 255

    set_validation_score_and_curve(
        svm.SVC(kernel='poly', C=C[0], probability=True, degree=degrees[0], verbose=True),
        x_train[:100], y_train[:100], x_cv[:100], y_cv[:100], x_test[:100], y_test[:100], "C", C,
        "PolynomialSvm(classifier, x_train, y_train, parameter)")

    classifiers = get_classifiers("classifiers")
    for classifier_name, classifier_list in classifiers.items():
        print(classifier_name)
        train_scores = []
        valid_scores = []
        for classifier in classifier_list:
            train_scores.append(classifier.params['train_score'])
            valid_scores.append(classifier.params['valid_score'])

        train_scores = np.array(train_scores)
        valid_scores = np.array(valid_scores)
        plot_validation_curve(train_scores, valid_scores, f"Validation Curve with SVM Degree:{degrees[0]}",
                              "C", "Score", C)


if __name__ == '__main__':
    main()
