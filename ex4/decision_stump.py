from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """returns the misclassification_error value of the params objects"""
    diff_sum = 0
    for i in range(len(y_true)):
        if np.sign(y_true[i]) != np.sign(y_pred[i]):
            diff_sum += abs(y_true[i])
    error_val = diff_sum / len(y_true)
    return error_val


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        loss_value = 1
        all_signs = [-1, 1]
        for idx in range(X.shape[1]):
            for sign in all_signs:
                catoff, min_loss = self._find_threshold(X[:, idx], y, sign)
                if min_loss < loss_value:
                    self.sign_ = sign
                    self.j_ = idx
                    self.threshold_ = catoff
                    loss_value = min_loss


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        to_predict_by = X.T[self.j_]
        predict_values = np.where(to_predict_by < self.threshold_, -self.sign_, self.sign_)
        return predict_values

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        sorted_idx = np.argsort(values)
        labels = labels[sorted_idx]
        values = values[sorted_idx]
        val_of_conc = (values[1:] + values[:-1]) / 2
        catoffs = np.concatenate([[-np.inf], val_of_conc, [np.inf]])
        minimum_loss = np.sum(labels)
        losses = np.append(minimum_loss, minimum_loss - np.cumsum(labels * sign))
        tup = (float(catoffs[np.argmin(losses)]), float(losses[np.argmin(losses)]))
        return tup

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        diff_sum = 0
        pred_val = self.predict(X)
        for i in range(len(y)):
            if pred_val[i] != y[i]:
                diff_sum += 1
        loss_val = diff_sum / len(y)
        return loss_val
