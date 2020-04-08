from pathlib import Path

from classifiers import *
from utils import read_file, plot_validation_curve, plot_time_per_parameter, plot_test_accuracy, plot_image
from utils import validation_curve

from sklearn import svm
from sklearn.model_selection import PredefinedSplit
from sklearn.linear_model import LogisticRegression as LogisticRegressionSKlearn


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
    degree = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    alphas = (0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000)
    max_iter = [200, 500, 1000, 2000]

    x_train, y_train = read_file('dataset/merged_train_set.csv')
    x_cv, y_cv = read_file('dataset/merged_cv_set.csv')
    x_test, y_test = read_file('dataset/merged_test_set.csv')

    import cv2
    im = cv2.imread("live_images/opencv_frame_0.png")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    new_img = []
    for line in im:
        new_img.append(np.mean(line, axis=1))

    
    
    new_img = np.array(new_img)
    #new_img = np.append((0,new_img))
    new_img = new_img.reshape(785,1)

    x_train = x_train / 255
    x_cv = x_cv / 255
    x_test = x_test / 255

    """
    set_validation_score_and_curve(
        svm.SVC(kernel='rbf', C=C[0], probability=True, degree=degree[0], verbose=True),
        x_train, y_train, x_cv, y_cv, x_test, y_test, "C", C,
        "RbfSvm(classifier, x_train, y_train, parameter)")

    set_validation_score_and_curve(
        svm.SVC(kernel='poly', C=C[0], probability=True, degree=degree[0], verbose=True),
        x_train, y_train, x_cv, y_cv, x_test, y_test, "C", C,
        "PolynomialSvm(classifier, x_train, y_train, parameter)")

    set_validation_score_and_curve(
        LogisticRegressionSKlearn(C=C[-1], verbose=True, max_iter=1000, n_jobs=-1),
        x_train, y_train, x_cv, y_cv, x_test, y_test, "C", C,
        "LogisticRegression(classifier, x_train, y_train, parameter)")

    set_validation_score_and_curve(
        svm.SVC(kernel='poly', C=C[0], probability=True, degree=degree[0], verbose=True),
        x_train, y_train, x_cv, y_cv, x_test, y_test, "degree", degree,
        "PolynomialSvm(classifier, x_train, y_train, parameter)")

    set_validation_score_and_curve(
        LogisticRegressionSKlearn(C=C[0], verbose=True, max_iter=1000, n_jobs=-1),
        x_train, y_train, x_cv, y_cv, x_test, y_test, "max_iter", max_iter,
        "LogisticRegression(classifier, x_train, y_train, parameter)")
    """

    classifiers = get_classifiers("classifiers_shuffle")
    for classifier_name, classifier_list in classifiers.items():

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

            print(classifier.predict(new_img / 255))

        train_scores = np.array(train_scores)
        valid_scores = np.array(valid_scores)
        fit_times = np.array(fit_times)
        score_times = np.array(score_times)
        tests_accuracy = np.array(tests_accuracy)

        """
        Path(f"graficos/{classifier_name}").mkdir(parents=True, exist_ok=True)
        plot_validation_curve(train_scores, valid_scores, f"Validation Curve with {classifier_name}",
                              classifier_list[0].variation_param, "Score", eval(classifier_list[0].variation_param),
                              f"{classifier_name}/validation_curve.png")

        plot_time_per_parameter(fit_times, score_times, f"Time of fitting and scoring processes with {classifier_name}",
                                classifier_list[0].variation_param, "Time (s)",
                                eval(classifier_list[0].variation_param),
                                f"{classifier_name}/time_per_parameter.png")
        plot_test_accuracy(eval(classifier_list[0].variation_param), tests_accuracy,
                           f"Test set accuracy with {classifier_name}", classifier_list[0].variation_param, "Accuracy",
                           f"{classifier_name}/test_accuracy.png")
    """


if __name__ == '__main__':
    main()
