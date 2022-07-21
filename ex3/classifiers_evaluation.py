from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from math import atan2, pi
pio.templates.default = "simple_white"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    all_data = np.load(filename)
    design_matrix = all_data[:, :2]
    vector_classifier = all_data[:, 2].astype(int)
    return design_matrix, vector_classifier


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def calc_loses_of_perceptron(fit):
            losses.append(fit.loss(X, y))
            fit.fitted_ = True

        Perceptron(callback=calc_loses_of_perceptron).fit(X, y)

        # Plot figure
        go.Figure([go.Scatter(x=np.arange(0, len(losses)), y=losses,
                              mode='markers+lines',
                              showlegend=False)],
                  layout=go.Layout(
                      title=r"$\text{Loss as Function of Fitting Iteration "
                            r"on " + n + " dataset}$",
                      xaxis_title=r"$\text{Fitting Iteration}$",
                      yaxis_title=r"$\text{Loss}$", height=300)).show()


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit models and predict over training set
        gua_naive_b = GaussianNaiveBayes().fit(X, y)
        gua_naive_b_pred = gua_naive_b.predict(X)
        lda = LDA().fit(X, y)
        lda_pred = lda.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy

        fig = make_subplots(rows=1, cols=2, subplot_titles=[
            "Classifier: Gaussian Naive Bayes, Accuracy: " + "{:.2f}".format(accuracy(y, gua_naive_b_pred)),
            "Classifier: LDA, Accuracy: " + "{:.2f}".format(
                accuracy(y, lda_pred))], vertical_spacing=.05, horizontal_spacing=0.03)
        models_pred = [gua_naive_b_pred, lda_pred]
        models = [gua_naive_b, lda]
        lims = np.array([X.min(axis=0), X.max(axis=0)]).T + np.array([-.4, .4])
        symbols = np.array(["x", "circle"])

        # Add traces for data-points setting symbols and colors
        for i, mod in enumerate(models):
            predictions_correctness = (models_pred[i] == y).astype(int)
            fig.add_traces([decision_surface(mod.predict, lims[0], lims[1], showscale=False),
                            go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                       marker=dict(color=y, symbol=symbols[predictions_correctness],
                                                   colorscale=[custom[0], custom[-1]],
                                                   line=dict(color="black", width=1)))],
                           rows=(i // 3) + 1, cols=(i % 3) + 1)

        models_mus = [gua_naive_b.mu_, lda.mu_]
        for i, mu in enumerate(models_mus):
            fig.add_traces([go.Scatter(x=mu[:, 0], y=mu[:, 1], mode="markers",showlegend=False,
                                       marker=dict(color='black', size=15,line=dict(color='black',width=2),\
                                                   symbol='x'))],rows=(i // 3) + 1, cols=(i % 3) + 1)
        for i in range(len(gua_naive_b.classes_)):
            cov_matrix = np.diag(gua_naive_b.vars_[i])
            l1, l2 = tuple(np.linalg.eigvalsh(cov_matrix)[::-1])
            theta_t = atan2(l1 - cov_matrix[0, 0], cov_matrix[0, 1]) if cov_matrix[0, 1] != 0 else (
                np.pi / 2 if cov_matrix[0, 0] < cov_matrix[1, 1] else 0)
            t = np.linspace(0, 2 * pi, 100)
            x_s = (l1 * np.cos(theta_t) * np.cos(t)) - (l2 * np.sin(theta_t) * np.sin(t))
            y_s = (l1 * np.sin(theta_t) * np.cos(t)) + (l2 * np.cos(theta_t) * np.sin(t))
            ellipse_draw = go.Scatter(x=gua_naive_b.mu_[i][0] + x_s, y=gua_naive_b.mu_[i][1] + y_s,
                                      mode="lines", marker_color="black")
            fig.add_traces([ellipse_draw], rows=1, cols=1)
        for i in range(len(lda.classes_)):
            l1, l2 = tuple(np.linalg.eigvalsh(lda.cov_)[::-1])
            theta_t = atan2(l1 - lda.cov_[0, 0], lda.cov_[0, 1]) if lda.cov_[0, 1] != 0 else (
                np.pi / 2 if lda.cov_[0, 0] < lda.cov_[1, 1] else 0)
            t = np.linspace(0, 2 * pi, 100)
            x_s = (l1 * np.cos(theta_t) * np.cos(t)) - (l2 * np.sin(theta_t) * np.sin(t))
            y_s = (l1 * np.sin(theta_t) * np.cos(t)) + (l2 * np.cos(theta_t) * np.sin(t))
            ellipse_draw = go.Scatter(x=lda.mu_[i][0] + x_s, y=lda.mu_[i][1] + y_s, mode="lines", marker_color="black")
            fig.add_traces([ellipse_draw], rows=1, cols=2)

        fig.update_layout(title=rf"Dataset: {f}",margin=dict(t=120), showlegend=False) \
            .update_xaxes(visible=False).update_yaxes(visible=False)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
