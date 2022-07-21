from __future__ import annotations
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, \
    RidgeRegression
from sklearn.linear_model import Lasso
from utils import *
import plotly.express as px


# Written by Avinoam Nukrai, Huji spring 2022


# for q1
def generate_for_model(X, func, train_X, train_y, test_X, test_y):
    without_noice = pd.DataFrame(data={'x': X, 'y_without_noice': func})
    train = pd.DataFrame(data={'train_X': train_X.reshape(-1), 'train_y': train_y.reshape(-1)})
    test = pd.DataFrame(data={'test_X': test_X.reshape(-1), 'test_y': test_y.reshape(-1)})
    fig = px.scatter(without_noice, x='x', y='y_without_noice')
    fig.add_scatter(x=train['train_X'], y=train['train_y'],
                    name="train", mode="markers")
    fig.add_scatter(x=test['test_X'], y=test['test_y'], name="test", mode="markers")
    return fig

# for q1
def create_sampels_response(n_samples, noise):
    noise_ = np.random.normal(size=n_samples, loc=0, scale=noise)
    samples = np.linspace(-1.2, 2, n_samples)
    func = (X + 3) * (X + 2) * (X + 1) * (X - 1) * (X - 2)
    response = func + noise_
    return samples, response

# for q2
def perform_pfd_cv(train_x, train_y):
    train_scores = np.zeros(11)
    validation_scores = np.zeros(11)
    for k in range(11):
        model = PolynomialFitting(k)
        train_scores[k], validation_scores[k] = cross_validate(model, train_X,
                                                               train_y,
                                                               mean_square_error)

    degree = np.linspace(0, 10, 11)
    df = pd.DataFrame(data={'Degree': degree, 'Train_scores': train_scores, 'Validation_scores': validation_scores})
    fig = px.line(df, x='Degree', y=['Train_scores', 'Validation_scores'])
    fig.show()


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X, y = create_sampels_response(n_samples, noise)
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(y), 2 / 3)
    train_X = train_X.to_numpy().reshape((train_X.shape[0],))
    train_y = train_y.to_numpy()
    test_X = test_X.to_numpy().reshape((test_X.shape[0],))
    test_y = test_y.to_numpy()
    generate_for_model(X, func, train_X, train_y, test_X, test_y).show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    perform_pfd_cv(train_X, train_y)

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = np.argmin(validation_scores)
    model = PolynomialFitting(best_k)
    model.fit(train_X.reshape(-1), train_y)
    loss = model.loss(test_X.reshape(-1), test_y)
    print(f"Best degree of polynomial is: {best_k}, and the test error is: {loss}")

# for q7
def preform_diff_vals_reg_cv(train_X, train_y, models, range, errors):
    for model in models:
        train_scores = np.zeros(len(range))
    validation_scores = np.zeros(len(range))
    for i, k in enumerate(range):
        train_scores[i], validation_scores[i] =\
            cross_validate(models[model](k), train_X, train_y, mean_square_error)
    errors[model] = range[np.argmin(validation_scores)]
    print(f'min_error of {model} model is: {errors[model]}')

    fig = px.line()
    fig.add_scatter(x=range, y=train_scores, name="train_score")
    fig.add_scatter(x=range, y=validation_scores, name="validation_score")
    fig.show()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X, train_y, test_X, test_y = X[:n_samples, :], y[:n_samples], X[n_samples:, :], y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    models1 = {"Lasso": lambda a: Lasso(alpha=a, max_iter=10000, tol=1e-4), "Ridge": lambda a: RidgeRegression(a, True)}
    errors = {}
    range1 = np.linspace(0, 50, n_evaluations)
    preform_diff_vals_reg_cv(train_X, train_y, models1, range1, errors)

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    models_2 = {'Lasso': Lasso(alpha=errors['Lasso'], max_iter=10000, tol=1e-4),
                'Ridge': RidgeRegression(errors['Ridge'], True),
                'Linear Regression': LinearRegression()}
    for mod in models_2:
        model = models_2[mod]
        model.fit(train_X, train_y)
        print(f"Best {mod} Test Error : {mean_square_error(test_y, model.predict(test_X))}")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
