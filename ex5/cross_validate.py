from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator

# Written by Avinoam Nukrai, Huji spring 2022

def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float],
                   cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_set = np.zeros(cv)
    validation_set = np.zeros(cv)
    split_x, split_y = np.array_split(X, cv), np.array_split(y, cv)
    for k in range(cv):
        sub_x = np.concatenate(split_x[:k] + split_x[k + 1:])
        x_k = split_x[k]
        sub_y = np.concatenate(split_y[:k] + split_y[k + 1:])
        y_k = split_y[k]
        train_pred = estimator.predict(sub_x)
        train_set[k] = scoring(sub_y, train_pred)
        validate_pred = estimator.predict(x_k)
        validation_set[k] = scoring(y_k, validate_pred)
        k_X_s = split_x[k]
        k_y_s = split_y[k]
        without_k_X = np.concatenate([split_x[i] for i in range(cv) if i != k])
        without_k_y = np.concatenate([split_y[i] for i in range(cv) if i != k])
        model = estimator.fit(without_k_X, without_k_y)
        train_prediction = model.predict(without_k_X)
        test_prediction = model.predict(k_X_s)
        train_set[k] = scoring(without_k_y, train_prediction)
        validation_set[k] = scoring(k_y_s, test_prediction)
    train_score = np.mean(train_set)
    validation_score = np.mean(validation_set)
    return float(train_score), float(validation_score)

