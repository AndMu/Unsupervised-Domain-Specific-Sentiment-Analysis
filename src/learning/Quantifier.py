from sklearn import svm
import numpy as np

from learning import logger
from sklearn.model_selection import StratifiedKFold, KFold


class Quantifier(object):
    def __init__(self, clf=None, reference_label=None, n_folds=50, seed=0):
        if clf is None:
            self._clf = svm.SVC(kernel='linear', probability=True)
        else:
            self._clf = clf
        self._n_folds = n_folds
        self._seed = seed
        self._reference_label = reference_label

    def fit(self, X, y):
        labels = np.unique(y)
        if len(labels) != 2:
            raise Exception("A binary setup is required")

        min_count = X.shape[0]
        self._min_label = None
        for label in labels:
            unique, allcounts = np.unique(y, return_counts=True)
            count = allcounts[np.searchsorted(unique, label)]
            if count <= min_count:
                min_count = count
                self._min_label = label

        if self._reference_label is None:
            self._reference_label = self._min_label

        if not self._reference_label in labels:
            raise Exception("Reference label does not appear in training data")

        if min_count >= self._n_folds:
            cv = StratifiedKFold(n_splits=min(X.shape[0], self._n_folds), shuffle=True, random_state=self._seed)
        else:
            cv = KFold(n_splits=min(X.shape[0], self._n_folds), shuffle=True, random_state=self._seed)

        tp = 0
        fp = 0
        ptp = 0
        pfn = 0
        pfp = 0
        ptn = 0

        for train_cv, test_cv in cv.split(X, y):
            logger.info("Split...")
            tp, fp, ptp, pfn, pfp, ptn = self._fit_fold(X, y, train_cv, test_cv)
            tp += tp
            fp += fp
            ptp += ptp
            pfn += ptn
            pfp += pfp
            ptn += ptn

        positives = min_count
        negatives = X.shape[0] - positives
        self._tpr = tp / positives
        self._fpr = fp / negatives
        self._ptpr = ptp / (ptp + pfn)
        self._pfpr = pfp / (pfp + ptn)
        self._clf.fit(X, y)
        if self._clf.classes_[0] == self._min_label:
            self._pos_idx = 0
            self._neg_idx = 1
        else:
            self._neg_idx = 0
            self._pos_idx = 1

    def _fit_fold(self, X, y, train_cv, test_cv ):
        tp = 0
        fp = 0
        ptp = 0
        pfn = 0
        pfp = 0
        ptn = 0
        train_X = X[train_cv]
        train_y = np.array([y[i] for i in train_cv])
        test_X = X[test_cv]
        test_y = np.array([y[i] for i in test_cv])
        self._clf.fit(train_X, train_y)
        if self._clf.classes_[0] == self._min_label:
            self._pos_idx = 0
            self._neg_idx = 1
        else:
            self._neg_idx = 0
            self._pos_idx = 1

        predicted_y = self._clf.predict(test_X)
        for (true_label, predicted_label) in zip(test_y, predicted_y):
            if true_label == predicted_label and true_label == self._min_label:
                tp += 1
            elif true_label != predicted_label and true_label != self._min_label:
                fp += 1
        probas = self._clf.predict_proba(test_X)
        for (label, proba) in zip(test_y, probas):
            if label == self._min_label:
                if proba.shape[0] > 1:
                    ptp += proba[self._pos_idx]
                    pfp += proba[self._neg_idx]
                else:
                    ptp += proba
                    pfp += 1 - proba
            else:
                if proba.shape[0] > 1:
                    ptn += proba[self._neg_idx]
                    pfn += proba[self._pos_idx]
                else:
                    ptn += proba
                    pfn += proba
            pfn += 1 - self._pos_idx
        return tp, fp, ptp, pfn, pfp, ptn

    def predict(self, X):
        observed = X.shape[0]
        predicted_positive = 0
        positive_probability_sum = 0.0

        predicted_y = self._clf.predict_proba(X)

        predicted_positive += list(predicted_y).count(self._min_label)

        probabilities = self._clf.predict_proba(X)
        for probability in probabilities:
            if probability.shape[0] > 1:
                positive_probability_sum += probability[self._pos_idx]
            else:
                positive_probability_sum += probability

        CC_prevalence = predicted_positive / observed
        ACC_prevalence = max(0.0, min(1.0, (CC_prevalence - self._fpr) / max(0.0001, (self._tpr - self._fpr))))
        PCC_prevalence = positive_probability_sum / observed
        PACC_prevalence = max(0.0, min(1.0, (PCC_prevalence - self._pfpr) / max(0.0001, (
            self._ptpr - self._pfpr))))

        if self._min_label != self._reference_label:
            CC_prevalence = 1 - CC_prevalence
            ACC_prevalence = 1 - ACC_prevalence
            PCC_prevalence = 1 - PCC_prevalence
            PACC_prevalence = 1 - PACC_prevalence

        results = list()
        results.append(('CC', self._reference_label, CC_prevalence))
        results.append(('ACC', self._reference_label, ACC_prevalence))
        results.append(('PCC', self._reference_label, PCC_prevalence))
        results.append(('PACC', self._reference_label, PACC_prevalence))

        return results