from abc import abstractmethod, ABCMeta

from sklearn import svm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ShuffleSplit
from pathlib import Path
import pickle


def grid_search_template(estimator, grid, cv):
    return GridSearchCV(estimator=estimator,
                        param_grid=grid,
                        scoring='accuracy',
                        cv=cv,
                        verbose=1,
                        n_jobs=-1)


def random_search_template(estimator, grid):
    return RandomizedSearchCV(estimator=estimator,
                              param_distributions=grid,
                              n_iter=50,
                              scoring='accuracy',
                              cv=3,
                              verbose=1,
                              random_state=8,
                              n_jobs=-1)


def save_classifier(classifier, file_name):
    Path(f"classifiers").mkdir(parents=True, exist_ok=True)  # if folder doesnt exists, crete one

    with open(f"classifiers/{file_name}", 'wb') as output:
        pickle.dump(classifier, output)


class Classifier(metaclass=ABCMeta):
    def __init__(self, svc, grid, grid_search, random_search):
        self.svc = svc
        self.grid = grid
        self.grid_search = grid_search
        self.random_search = random_search

    @abstractmethod
    def grid_search_cv(self, x, y, save=False, file_name=None):
        pass

    @abstractmethod
    def random_search(self, x, y, save=False, file_name=None):
        pass

    def __str__(self):
        return self.svc

    def __repr__(self):
        return self.__str__()


class PolynomialSvm(Classifier):
    def __init__(self, C, degree):
        # .svc = svm.SVC(kernel='poly')
        self.grid = {
            'C': C,
            'kernel': ['poly'],
            'degree': degree,
            'probability': [True]
        }

        super().__init__(svm.SVC(kernel='poly'), self.grid)

    def grid_search_cv(self, x, y, save=False, file_name=None):
        cv_sets = ShuffleSplit(n_splits=3, test_size=.33, random_state=8)

        self.grid_search = grid_search_template(self.svc, self.grid, cv_sets)

        self.grid_search.fit(x, y)

        if save:
            pass

    def random_search(self, x, y, save=False, file_name=None):
        self.random_search = random_search_template(self.svc, self.grid)

        self.random_search.fit(x, y)

        if save:
            pass


class RadialBasisSvm(Classifier):
    def __init__(self, C, gamma):
        self.svc = svm.SVC(kernel='rbf')
        self.grid = {
            'C': C,
            'kernel': ['rbf'],
            'gamma': gamma,
            'probability': [True]
        }
        super().__init__()

    def grid_search_cv(self, x, y, save=False, file_name=None):
        cv_sets = ShuffleSplit(n_splits=3, test_size=.33, random_state=8)

        self.grid_search = grid_search_template(self.svc, self.grid, cv_sets)

        self.grid_search.fit(x, y)

        if save:
            pass

    def random_search(self, x, y, save=False, file_name=None):
        self.random_search = random_search_template(self.svc, self.grid)

        self.random_search.fit(x, y)

        if save:
            pass
