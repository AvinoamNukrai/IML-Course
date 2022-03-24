from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet
from scipy.stats import multivariate_normal

# Written by Avinoam Nukrai, Spring 2022 Hebrew U

class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """
    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=True
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has
            not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in
             `UnivariateGaussian.fit` function.

        var_: float
            Estimated variance initialized as None. To be
             set in `UnivariateGaussian.fit` function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        # raise NotImplementedError()

        self.mu_ = np.mean(X)
        self.fitted_ = True
        if self.biased_:
            self.var_ = np.var(X)
        else:
            self.var_ = np.var(X, ddof=1)
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        self.fitted_ = True
        return np.array((1 / (np.sqrt(2 * np.pi * self.var_)) *
                         np.exp(-0.5 * ((X - self.mu_) ** 2) * self.var_)))
        # raise NotImplementedError()

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        # raise NotImplementedError()
        pdf_array = np.array((1 / (np.sqrt(2 * np.pi * sigma)) *
                              np.exp(-0.5 * ((X - mu) ** 2) * sigma)))
        sum_of_pdfs_logs = np.log(pdf_array).sum()
        return sum_of_pdfs_logs


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.ft`
            function.

        cov_: float
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.ft`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = np.mean(X, axis=0)
        self.cov_ = np.cov(X.T)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        pdf_observe = []
        for sample in X:
            pdf_observe.append(1 / (np.sqrt((2 * np.pi) ** sample.size *
            np.det(self.cov_))) * np.exp(-(
                np.linalg.solve(
                    self.cov_, sample - self.mu_).T.dot(
                    sample - self.mu_)) / 2))
        return pdf_observe

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        cov : float
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        sigma_cov = np.matrix(cov)
        sigma_cov_det = np.linalg.det(sigma_cov)
        sigma_cov_inverse = sigma_cov.I
        comp1 = X.size * np.log(2 * np.pi)
        comp2 = len(mu) * np.log(sigma_cov_det)
        comp3 = 0
        for x in X:
            comp3 += np.linalg.multi_dot([(x - mu).T, sigma_cov_inverse, (x - mu)])
        return -0.5 * (comp1 + comp2 + comp3)
        # return np.sum(np.log(multivariate_normal.pdf(X, mu, cov)))
