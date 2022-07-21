from typing import NoReturn
import numpy as np
from IMLearn import BaseEstimator
from IMLearn.desent_methods import GradientDescent
from IMLearn.desent_methods.modules import LogisticModule, RegularizedModule, L1, L2


class LogisticRegression(BaseEstimator):
    """
    Logistic Regression Classifier

    Attributes
    ----------
    solver_: GradientDescent, default=GradientDescent()
        Descent method solver to use for the logistic regression objective optimization

    penalty_: str, default="none"
        Type of regularization term to add to logistic regression objective. Supported values
        are "none", "l1", "l2"

    lam_: float, default=1
        Regularization parameter to be used in case `self.penalty_` is not "none"

    alpha_: float, default=0.5
        Threshold value by which to convert class probability to class value

    include_intercept_: bool, default=True
        Should fitted model include an intercept or not

    coefs_: ndarray of shape (n_features,) or (n_features+1,)
        Coefficients vector fitted by linear regression. To be set in
        `LogisticRegression.fit` function.
    """

    def __init__(self,
                 include_intercept: bool = True,
                 solver: GradientDescent = GradientDescent(),
                 penalty: str = "none",
                 lam: float = 1,
                 alpha: float = .5):
        """
        use LogisticModule if penalty = "none", else use RegularizedModule with the model in penalty with lamda = lam.

        Instantiate a linear regression estimator

        Parameters
        ----------
        solver: GradientDescent, default=GradientDescent()
            Descent method solver to use for the logistic regression objective optimization

        penalty: str, default="none"
            Type of regularization term to add to logistic regression objective. Supported values
            are "none", "l1", "l2"

        lam: float, default=1
            Regularization parameter to be used in case `self.penalty_` is not "none"

        alpha: float, default=0.5
            Threshold value by which to convert class probability to class value

        include_intercept: bool, default=True
            Should fitted model include an intercept or not
        """
        super().__init__()
        norm_list = ["none", "l1", "l2"]
        self.include_intercept_ = include_intercept
        self.solver_ = solver
        self.lam_ = lam
        self.penalty_ = penalty
        self.alpha_ = alpha
        if penalty not in norm_list:
            raise ValueError("Supported penalty types are: none, l1, l2")
        self.coefs_ = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Logistic regression model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model using specified `self.optimizer_` passed when instantiating class and includes an intercept
        if specified by `self.include_intercept_
        """
        norm_list = ["l1", "l2"]
        X_intercept = np.ones((X.shape[0], X.shape[1] + 1))
        X_intercept[:, 1:] = X
        X_intercept = X_intercept if self.include_intercept_ else X
        w_size = X_intercept.shape[1]
        w = np.random.normal(0, 1, w_size)/np.sqrt(w_size)
        if self.penalty_ == norm_list[0]:
            base_model = RegularizedModule(fidelity_module=LogisticModule(w),regularization_module=L1(w),lam=self.lam_,weights=w)
        elif self.penalty_ == norm_list[1]:
            base_model = RegularizedModule(fidelity_module=LogisticModule(w),regularization_module=L2(w),lam=self.lam_,weights=w)
        else:
            base_model = LogisticModule(w)
        self.coefs_ = self.solver_.fit(base_model, X_intercept, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        prob_prediction = self.predict_proba(X)
        prediction = np.where(prob_prediction < self.alpha_, 0, 1)
        return prediction

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities of samples being classified as `1` according to sigmoid(Xw)

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict probability for

        Returns
        -------
        probabilities: ndarray of shape (n_samples,)
            Probability of each sample being classified as `1` according to the fitted model
        """
        # add column of 1's
        X_intercept = np.ones((X.shape[0], X.shape[1] + 1))
        X_intercept[:, 1:] = X
        X_intercept = X_intercept if self.include_intercept_ else X
        pred = X_intercept @ self.coefs_
        val = 1.0 / (1.0 + np.exp(-pred))
        return val



    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification error

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under misclassification error
        """
        from ...metrics import misclassification_error
        return misclassification_error(y,self._predict(X))
