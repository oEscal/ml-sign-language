from pathlib import Path

from sklearn import svm

from classifiers import *
from utils import read_file, plot_validation_curve, plot_time_per_parameter, plot_test_accuracy, plot_image
from utils import validation_curve
from sklearn.model_selection import PredefinedSplit
from sklearn.linear_model import LogisticRegression as LogisticRegressionSKlearn
import sys

C = (0.001, 0.002, 0.01, 0.02, 0.1, 0.2, 1, 5, 10, 50, 100, 500, 1000)
degree = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
alphas = (0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000)
max_iter = [200, 500, 1000, 2000]
gamma = (0.001, 0.002, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000)


def plot_data(classifier_list):
    train_scores = []
    valid_scores = []
    fit_times = []
    score_times = []
    tests_accuracy = []

    for classifier in classifier_list:
        train_scores.append(classifier.params['train_score'])
        valid_scores.append(classifier.params['valid_score'])
        fit_times.append(classifier.params['fit_time'])
        score_times.append(classifier.params['score_time'])
        tests_accuracy.append(classifier.params['Test set Accuracy'])

    train_scores = np.array(train_scores)
    valid_scores = np.array(valid_scores)
    fit_times = np.array(fit_times)
    score_times = np.array(score_times)
    tests_accuracy = np.array(tests_accuracy)

    classifier_name = f"{classifier_list[0].name}_{classifier_list[0].variation_param}"

    Path(f"graficos/{classifier_name}").mkdir(parents=True, exist_ok=True)
    plot_validation_curve(1 - train_scores, 1 - valid_scores,
                          f"Error for variation of  {classifier_list[0].variation_param}",
                          classifier_list[0].variation_param, "Error", eval(classifier_list[0].variation_param),
                          f"{classifier_name}/validation_curve.png")

    plot_time_per_parameter(fit_times, score_times, f"Time of fitting and scoring processes with {classifier_name}",
                            classifier_list[0].variation_param, "Time (s)",
                            eval(classifier_list[0].variation_param),
                            f"{classifier_name}/time_per_parameter.png")
    """
    plot_test_accuracy(eval(classifier_list[0].variation_param), 1 - tests_accuracy,
                       f"Test set error with {classifier_name}", classifier_list[0].variation_param, "Error",
                       f"{classifier_name}/test_accuracy.png")
    """


def set_validation_score_and_curve(classifier, x_train, y_train, x_cv, y_cv, x_test, y_test, parameter,
                                   parameter_values, classifier_class):
    data_x, data_y = np.concatenate((x_train, x_cv)), np.concatenate((y_train, y_cv))

    train_indices = np.full((x_train.shape[0],), -1, dtype=int)
    cv_indices = np.full((x_cv.shape[0],), 0, dtype=int)
    ps = PredefinedSplit(np.append(train_indices, cv_indices))

    estimators_svm, train_scores_svm, valid_scores_svm, fit_times, score_times = validation_curve(
        classifier, data_x, data_y.ravel(), parameter, parameter_values, cv=ps, n_jobs=-1)

    classifier_list = []
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

        classifier_list.append(c)

    plot_data(classifier_list)


def main():
    x_train, y_train = read_file('dataset/merged_train_set.csv')
    x_cv, y_cv = read_file('dataset/merged_cv_set.csv')
    x_test, y_test = read_file('dataset/merged_test_set.csv')

    x_train = x_train / 255
    x_cv = x_cv / 255
    x_test = x_test / 255

    print(x_train.shape)

    set_validation_score_and_curve(
        svm.SVC(kernel='rbf', C=1, probability=True, degree=degree[0], verbose=True, gamma=1),
        x_train, y_train, x_cv, y_cv, x_test, y_test, "gamma", gamma,
        "RbfSvm(classifier, x_train, y_train, parameter)")

    _, best_gamma = pick_best_classier_param("classifiers/RbfSvm_gamma")

    set_validation_score_and_curve(
        svm.SVC(kernel='rbf', C=C[0], probability=True, degree=degree[len(degree) // 2],
                verbose=True, gamma=best_gamma),
        x_train, y_train, x_cv, y_cv, x_test, y_test, "C", C,
        "RbfSvm(classifier, x_train, y_train, parameter)")

    set_validation_score_and_curve(
        svm.SVC(kernel='poly', C=1, probability=True, degree=degree[len(degree) // 2], verbose=True, gamma=1),
        x_train, y_train, x_cv, y_cv, x_test, y_test, "gamma", gamma,
        "PolynomialSvm(classifier, x_train, y_train, parameter)")

    _, best_svm_gamma = pick_best_classier_param("classifiers/PolynomialSvm_gamma")

    set_validation_score_and_curve(
        svm.SVC(kernel='poly', C=C[0], probability=True, degree=degree[len(degree) // 2], verbose=True,
                gamma=best_svm_gamma),
        x_train, y_train, x_cv, y_cv, x_test, y_test, "C", C,
        "PolynomialSvm(classifier, x_train, y_train, parameter)")

    _, best_svm_C = pick_best_classier_param("classifiers/PolynomialSvm_C")

    set_validation_score_and_curve(
        svm.SVC(kernel='poly', C=best_svm_C, probability=True, degree=degree[0], verbose=True, gamma=best_svm_gamma),
        x_train, y_train, x_cv, y_cv, x_test, y_test, "degree", degree,
        "PolynomialSvm(classifier, x_train, y_train, parameter)")

    set_validation_score_and_curve(
        LogisticRegressionSKlearn(C=C[len(C) // 2], verbose=True, max_iter=1000, n_jobs=-1),
        x_train, y_train, x_cv, y_cv, x_test, y_test, "C", C,
        "LogisticRegression(classifier, x_train, y_train, parameter)")

    _, best_lr_C = pick_best_classier_param("classifiers/RbfSvm_gamma")

    set_validation_score_and_curve(
        LogisticRegressionSKlearn(C=best_lr_C, verbose=True, max_iter=1000, n_jobs=-1),
        x_train, y_train, x_cv, y_cv, x_test, y_test, "max_iter", max_iter,
        "LogisticRegression(classifier, x_train, y_train, parameter)")


if __name__ == '__main__':
    main()
