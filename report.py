from os import listdir

from utils import read_file, represent_data_graphically
import numpy as np
import matplotlib.pyplot as plt

from pylab import title, figure, xlabel, ylabel, xticks, bar, legend, axis, savefig
from classifiers import get_classifiers


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


def main():
    x_train, y_train = read_file('dataset/merged_train_set.csv')
    x_cv, y_cv = read_file('dataset/merged_cv_set.csv')
    x_test, y_test = read_file('dataset/merged_test_set.csv')

    plot_label_frequencies(y_train, "Train set", file_name='graficos/hist_training.png')
    plot_label_frequencies(y_cv, "Cross Validation set", file_name='graficos/hist_cv.png')
    plot_label_frequencies(y_test, "Test set", file_name='graficos/hist_test.png')

    # plot_image(x_train[0])

    # best_classifiers = get_bests_classifier("best_param_classifiers")

    # represent_data_graphically(x_train, 'graficos/ok.png')

    # classifiers = get_classifiers("best_param_classifiers")


#
# for name, classifier_list in classifiers.items():
#     print(name)
#     for classifier in classifier_list:
#         print(classifier)
#
#     save_best_param(classifier_list)


if __name__ == '__main__':
    main()
