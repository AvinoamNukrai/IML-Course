from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import math
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"

# Written by Avinoam Nukrai, Spring 2022 Hebrew U

def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    uv = UnivariateGaussian()
    simpelsArray = np.random.normal(10, 1, 1000)
    uv.fit(simpelsArray)
    print((uv.mu_, uv.var_))

    # Question 2 - Empirically showing sample mean is consistent
    intervals = np.linspace(10, 1000, 100)
    estimated_means = []
    for sample in intervals:
        simpelsArray = np.random.normal(10, 1, int(sample))
        uv.fit(simpelsArray)
        estimated_means.append(np.abs(uv.mu_ - 10))
    go.Figure([go.Scatter(x=intervals, y=estimated_means,
                          mode='markers+lines', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{(5) Estimation of Expectation As Function Of Number Of Samples}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="r$\hat\mu$",
                               height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    intervals = np.linspace(5, 15, 690)
    estimated_means = uv.pdf(intervals)
    go.Figure([go.Scatter(x=intervals, y=estimated_means, mode='lines', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{(5) Estimation of Expectation As Function Of Number Of Samples}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="r$\hat\mu$",
                               height=700)).show()


    # raise NotImplementedError()

def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mv = MultivariateGaussian()
    mean_vec = np.array([0,0,4,0])
    cov_var_matrix = np.matrix([[1, 0.2, 0, 0.5]
                                ,[0.2, 2, 0, 0]
                                ,[0, 0, 1, 0]
                                ,[0.5, 0, 0, 1]])
    sampels = np.random.multivariate_normal(mean_vec, cov_var_matrix, 1000)
    mv.fit(sampels)
    print(mv.mu_)
    print(mv.cov_)
    # raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    log_like_array = []
    f1_pos = np.linspace(-10, 10, 200)
    f3_pos = np.linspace(-10, 10, 200)
    max_log_like = -math.inf
    max_f1_f3_indices = [-1, -1]
    for f1 in f1_pos:
        temp_list = []
        for f3 in f3_pos:
            temp_mu_vec = np.array([f1, 0, f3, 0])
            log_like_per_indices = MultivariateGaussian.log_likelihood(
                temp_mu_vec, cov_var_matrix, sampels)
            if max_log_like < log_like_per_indices: # for Q6
                max_log_like = log_like_per_indices
                max_f1_f3_indices[0], max_f1_f3_indices[1] =\
                    round(f1, 3), round(f3, 3)

            temp_list.append(log_like_per_indices)
        log_like_array.append(temp_list)
    go.Figure(
        data=go.Heatmap(z=log_like_array, x=f1_pos, y=f3_pos)).update_layout(
        xaxis_title='f1', yaxis_title='f3',
        title='The log likelihood of the given expectations [f1,0,f3,0]',
        xaxis_nticks=36).show()
    # raise NotImplementedError()

    # Question 6 - Maximum likelihood
    print(max_log_like)
    print(max_f1_f3_indices)
    # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()


