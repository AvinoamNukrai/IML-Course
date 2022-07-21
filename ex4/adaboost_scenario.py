import numpy as np
from typing import Tuple
from IMLearn.learners.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def decision_surface(predict, xrange, yrange, t, density=120, dotted=False,
                     colorscale=custom, showscale=True):
    xrange, yrange = np.linspace(*xrange, density), np.linspace(*yrange,
                                                                density)
    xx, yy = np.meshgrid(xrange, yrange)
    pred = predict(np.c_[xx.ravel(), yy.ravel()], t)

    if dotted:
        return go.Scatter(x=xx.ravel(), y=yy.ravel(), opacity=1,
                          mode="markers", marker=dict(color=pred, size=1,
                                                      colorscale=colorscale,
                                                      reversescale=False),
                          hoverinfo="skip", showlegend=False)
    return go.Contour(x=xrange, y=yrange, z=pred.reshape(xx.shape),
                      colorscale=colorscale, reversescale=False, opacity=.7,
                      connectgaps=True, hoverinfo="skip", showlegend=False,
                      showscale=showscale)


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(wl=lambda: DecisionStump(), iterations=n_learners)
    a = adaboost.fit(train_X, train_y)
    D = a.D_
    train_error, test_error = [], []
    for t in range(1, n_learners):
        test_error.append(adaboost.partial_loss(test_X, test_y, t))
        train_error.append(adaboost.partial_loss(train_X, train_y, t))

    num_of_iterations = [i for i in range(1, n_learners)]
    plt.plot(num_of_iterations, test_error)
    plt.plot(num_of_iterations, train_error)
    plt.title("adaboost error")
    plt.xlabel("num of iterations")
    plt.ylabel("Error")
    plt.legend(["Train", "Test"])
    plt.grid()
    plt.show()


    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[f"{t} classifiers" for t in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, j in enumerate(T):
        fig.add_traces([decision_surface(a.partial_predict, lims[0],
                                         lims[1], j, showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                                   mode="markers",
                                   showlegend=False,
                                   marker=dict(color=test_y,
                                               colorscale=[custom[0],
                                                           custom[-1]],
                                               line=dict(color="black",
                                                         width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig.update_layout(
        title="Decision Boundaries",
        margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.write_html('tmp.html', auto_open=True)

    # Question 3: Decision surface of best performing ensemble
    from IMLearn.metrics import accuracy

    min_error_test_check = np.argmin(test_error)
    adb_accur = accuracy(test_y, adaboost.partial_predict(test_X,
                                                                  min_error_test_check))
    fig = make_subplots(rows=1, cols=1)
    fig.add_traces([decision_surface(a.partial_predict, lims[0],
                                     lims[1], min_error_test_check,
                                     showscale=False),
                    go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                               mode="markers",
                               showlegend=False,
                               marker=dict(color=test_y,
                                           colorscale=[custom[0],
                                                       custom[-1]],
                                           line=dict(color="black",
                                                     width=1)))], )
    fig.update_layout(
        title=f"Decision Surface of min error.\n size: {min_error_test_check}, accuracy: {adb_accur}",
        margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.write_html('tmp.html', auto_open=True)

    # Question 4: Decision surface with weighted samples
    D = D / np.max(D) * 5
    fig = make_subplots(rows=1, cols=1)
    fig.add_traces([decision_surface(a.partial_predict, lims[0],
                                     lims[1], n_learners,
                                     showscale=False),
                    go.Scatter(x=train_X[:, 0], y=train_X[:, 1],
                               mode="markers",
                               showlegend=False,
                               marker=dict(color=train_y,
                                           colorscale=[custom[0],
                                                       custom[-1]],
                                           line=dict(color="black",
                                                     width=1), size=D))], )
    fig.update_layout(
        title="training set with a point size proportional to it’s weight",
        margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.write_html('tmp.html', auto_open=True)


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0, 250, 5000, 500)
    fit_and_evaluate_adaboost(0.4, 250, 5000, 500)