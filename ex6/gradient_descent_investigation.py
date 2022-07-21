import copy
import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type
from pandas import DataFrame
from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
import plotly.express as px
from sklearn.metrics import roc_curve, auc
from IMLearn.model_selection import cross_validate
import plotly.graph_objects as go
from IMLearn.metrics import misclassification_error
MAX_ITER = 20000


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values_list, weights_list = [], []

    def call(weights, val):
        weights_list.append(np.array(weights))
        values_list.append(val)

    return call, values_list, weights_list


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    dict_name = {L1: "L1", L2: "L2"}
    for eta in etas:
        for base_model in [L1, L2]:
            callback_, val_list, weights_list = get_gd_state_recorder_callback()
            curr_gradient = GradientDescent(learning_rate=FixedLR(eta), callback=callback_)
            curr_base = base_model(weights=copy.deepcopy(init))
            curr_gradient.fit(curr_base, X=None, y=None)
            descent_plot = plot_descent_path(module=base_model,descent_path=np.array(weights_list),title=f"for {eta} learning rate and {dict_name[base_model]} base model")
            descent_plot.show()
            norm_plot = px.line(x=list(range(1,len(val_list)+1)),y=np.array(val_list),title=f"norm for {eta} learning rate and {dict_name[base_model]} base model")
            norm_plot.show()
            print(f"lowest loss for {eta} and {dict_name[base_model]} is: ", np.min(np.array(val_list)))


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    ############################################################################
    #   Optimize the L1 objective using different decay-rate values of the     #
    #   exponentially decaying learning rate and Plot algorithm's convergence  #
    #                   for the different values of gamma                      #
    ############################################################################

    plot_norm = []
    for gamma in gammas:
        callback_, val_list, weights_list = get_gd_state_recorder_callback()
        curr_gradient = GradientDescent(learning_rate=ExponentialLR(base_lr=eta, decay_rate=gamma), callback=callback_)
        curr_base = L1(weights=copy.deepcopy(init))
        curr_gradient.fit(curr_base, X=None, y=None)
        print(f"lowest loss with eta={eta} and gamma={gamma} for L1 is: ", np.min(np.array(val_list)))
        plot_norm.append(go.Scatter(x=list(range(1, len(val_list) + 1)), y=np.array(val_list), mode="lines", name=f"{gamma} decay rate",
                                    line=dict(width=1.2)))
    figure = go.Figure(plot_norm, layout=go.Layout(title="convergence rate for different decay rates",
                                                   xaxis_title="iterations number", yaxis_title="convergence rate"))
    figure.show()

    ##########################################################################
    #                   Plot descent path for gamma=0.95                     #
    ##########################################################################

    callback_, val_list, weights_list = get_gd_state_recorder_callback()
    curr_gradient = GradientDescent(learning_rate=ExponentialLR(base_lr=eta, decay_rate=0.95), callback=callback_)
    curr_base = L1(weights=copy.deepcopy(init))
    curr_gradient.fit(curr_base, X=None, y=None)
    descent_plot = plot_descent_path(module=L1, descent_path=np.array(weights_list),
                                     title=f"for {eta} learning rate and 0.95 decay rate for L1 base model")
    descent_plot.show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    lamdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test=np.array(X_train), np.array(y_train),np.array(X_test), np.array(y_test)

    # Plotting convergence rate of logistic regression over SA heart disease data

    logistic_regression_model = LogisticRegression(solver=GradientDescent(FixedLR(1e-4), max_iter=MAX_ITER)) \
        .fit(np.array(X_train), np.array(y_train))

    fpr, tpr, thresholds = roc_curve(y_train, logistic_regression_model.predict_proba(X_train))

    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'), name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()

    best_alpha = thresholds[np.argmax(tpr - fpr)]
    logistic_regression_model = LogisticRegression(solver=GradientDescent(FixedLR(1e-4), max_iter=MAX_ITER), alpha=best_alpha) \
        .fit(X_train, y_train)
    print(f"test error with alpha = {best_alpha} is {logistic_regression_model.loss(X_test, y_test)}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    validation_array_l1 = np.zeros(len(lamdas))
    validation_array_l2 = np.zeros(len(lamdas))
    gradient_descent_model = GradientDescent(FixedLR(1e-4), max_iter=MAX_ITER)
    for index, lam_ in enumerate(lamdas):
        logistic_regression_model_l1 = LogisticRegression(solver=gradient_descent_model, lam=lam_, penalty="l1")
        logistic_regression_model_l2 = LogisticRegression(solver=gradient_descent_model, lam=lam_, penalty="l2")

        _, validation_array_l1[index] = cross_validate(logistic_regression_model_l1, X_train, y_train,
                                                                     misclassification_error)
        _, validation_array_l2[index] = cross_validate(logistic_regression_model_l2, X_train, y_train,
                                                                     misclassification_error)
    best_lam_l1 = lamdas[np.argmin(validation_array_l1)]
    best_lam_l2 = lamdas[np.argmin(validation_array_l2)]
    logistic_regression_model_l1_best = LogisticRegression(solver=gradient_descent_model, lam=best_lam_l1, penalty="l1")
    logistic_regression_model_l2_best = LogisticRegression(solver=gradient_descent_model, lam=best_lam_l2, penalty="l2")
    logistic_regression_model_l1_best.fit(X_train,y_train)
    logistic_regression_model_l2_best.fit(X_train,y_train)
    l1_loss = logistic_regression_model_l1_best.loss(X_test,y_test)
    l2_loss = logistic_regression_model_l2_best.loss(X_test,y_test)
    print(f"the test loss with L1 base model with lambda= {best_lam_l1} is: {l1_loss}")
    print(f"the test loss with L2 base model with lambda= {best_lam_l2} is: {l2_loss}")

if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
