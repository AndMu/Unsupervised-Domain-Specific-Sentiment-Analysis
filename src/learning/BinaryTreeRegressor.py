
from sklearn.base import BaseEstimator, RegressorMixin, clone
import numpy as np
from sklearn.calibration import CalibratedClassifierCV


class BinaryTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimator=None, verbose=False):
        self.base_estimator = base_estimator
        self.verbose = verbose
        self._fitted_estimator = None
        self.counter = 0

    def fit(self, X, y):
        self._fitted_estimator = self._fit(X, y)
        return self

    def _fit(self, X, y):
        labels = np.unique(y)
        labels.sort()
        if len(labels) == 1:
            if self.verbose:
                print('Leaf', labels)
            return labels

        try:
            counts = [y.count(label) for label in labels]
        except AttributeError:
            unique, allcounts = np.unique(y, return_counts=True)
            counts = [allcounts[np.searchsorted(unique, label)] for label in labels]

        CalibratedClassifierCV
        total = len(y)
        div = [abs(0.5 - (sum(counts[:i + 1]) / float(total))) for i in range(0, len(counts))]
        split_point = div.index(min(div))
        split = labels[split_point]
        left_labels = labels[:split_point + 1]
        right_labels = labels[split_point + 1:]
        if self.verbose:
            print('Training:', labels, counts, div, split, left_labels, right_labels)

        bin_y = [label in left_labels for label in y]
        self.counter += 1
        node_estimator = self.base_estimator.copy(self.counter)
        node_estimator.fit(X, np.array(bin_y))

        left_indexes = [i for i, label in enumerate(y) if label in left_labels]
        left_X = X[left_indexes]
        left_y = [label for label in y if label in left_labels]

        right_indexes = [i for i, label in enumerate(y) if label in right_labels]
        right_X = X[right_indexes]
        right_y = [label for label in y if label in right_labels]

        if self.verbose:
            print('Left/right fit size:', len(left_y), len(right_y))

        return node_estimator, self._fit(left_X, left_y), self._fit(right_X, right_y)

    def predict(self, X):
        y_pred = list()
        for x in X:
            y_pred.append(self._predict(x, self._fitted_estimator))
        return y_pred

    def _predict(self, x, estimator):
        if len(estimator) == 1:
            return estimator[0]
        if estimator[0].predict_proba(x):
            return self._predict(x, estimator[1])
        else:
            return self._predict(x, estimator[2])