from sklearn.model_selection import ShuffleSplit

from utils import read_file, represent_data_graphically
import numpy as np
import matplotlib.pyplot as plt

from pylab import title, figure, xlabel, ylabel, xticks, bar, legend, axis, savefig
from classifiers import get_classifiers, save_best_classifiers
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve


def plot_image(img, file_name=None, show=False):
    title_name = f"Data set image"

    img_size = int(img.shape[0] ** 0.5)
    img = img.reshape(img_size, img_size)

    plt.imshow(img)
    savefig(file_name if file_name is not None else f"graficos/{title_name.replace(' ', '_')}.png")
    if show:
        plt.show()

    plt.clf()


def plot_label_frequencies(labels, term, file_name=None, show=False):
    title_name = f"{term} label frequency"

    title(title_name)
    xlabel('Label')
    ylabel('Frequency')

    unique, counts = np.unique(labels, return_counts=True)
    for u, c in np.asarray((unique, counts)).T:
        bar(u, c, width=0.25)

    savefig(file_name if file_name is not None else f"graficos/{title_name.replace(' ', '_')}.png")

    if show:
        plt.show()

    plt.clf()


def learning_curve(estimator, x_train, y_train):
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))

    # X, y = load_digits(return_X_y=True)

    title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    plot_learning_curve(estimator, title, x_train, y_train, axes=axes[:, 1], ylim=(0.0, 10),
                        cv=cv, n_jobs=4)

    plt.show()


def main():
    x_train, y_train = read_file('dataset/merged_train_set.csv')
    x_cv, y_cv = read_file('dataset/merged_cv_set.csv')
    x_test, y_test = read_file('dataset/merged_test_set.csv')

    # original_train_x, original_train_y = read_file('dataset/original_train.csv')
    # original_test_x, original_test_y = read_file('dataset/original_test.csv')

    # plot_label_frequencies(y_train, "Train set", file_name='graficos/hist_training.png')
    # plot_label_frequencies(y_cv, "Cross Validation set", file_name='graficos/hist_cv.png')
    # plot_label_frequencies(y_test, "Test set", file_name='graficos/hist_test.png')

    # original_train_x, original_test_x = original_train_x / 255, original_test_x / 255

    # plot_image(x_train[0])

    # best_classifiers = get_bests_classifier("best_param_classifiers")

    # represent_data_graphically(x_train, 'graficos/ok.png')

    classifiers = get_classifiers("classifiers")

    for classifier_name, classifier_list in classifiers.items():
        best = save_best_classifiers(classifier_list)
        #print(best.name)

        #learning_curve(best.classifier, x_train[:100], y_train[:100])
        print(best.history.loss_curve_)
        # print(best.accuracy(original_train_x, original_train_y, "accuracy train original"))
        # print(best.accuracy(original_test_x, original_test_y, "accuracy test original"))


#
# for name, classifier_list in classifiers.items():
#     print(name)
#     for classifier in classifier_list:
#         print(classifier)
#
#     save_best_param(classifier_list)


if __name__ == '__main__':
    main()
