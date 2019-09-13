import numpy as np
import scipy.stats as st
from scipy.optimize import bisect
from sklearn.utils import check_random_state

from bask.utils import get_progress_bar


__all__ = ["evaluate_acquisitions"]


def evaluate_acquisitions(X, gpr, acquisition_functions=None, n_samples=10, progress=False,
                          random_state=None, **kwargs):
    n_cand_points = len(X)
    n_acqs = len(acquisition_functions)
    acq_output = np.zeros((n_acqs, n_cand_points))
    random_state = check_random_state(random_state)
    trace_sample_i = random_state.choice(len(gpr.chain_), replace=False, size=n_samples)
    param_backup = np.copy(gpr.theta)
    pbar = get_progress_bar(progress, len(trace_sample_i))
    for i in trace_sample_i:
        gpr.theta = gpr.chain_[i]
        with gpr.noise_set_to_zero():
            mu, std = gpr.predict(X, return_std=True)
            for j, acq in enumerate(acquisition_functions):
                tmp_out = acq(mu, std, **kwargs)
                if np.all(np.isfinite(tmp_out)):
                    acq_output[j] += tmp_out
        pbar.update(1)
    gpr.theta = param_backup
    return acq_output / n_samples


def _ei_f(x):
    return x * st.norm.cdf(x) + st.norm.pdf(x)


def expected_improvement(mu, std, y_opt=None, **kwargs):
    if y_opt is None:
        y_opt = mu.max()
    values = np.zeros_like(mu)
    mask = std > 0
    inner = (y_opt - mu[mask]) / std[mask]
    values[mask] = _ei_f(inner) * std[mask]
    return values


def top_two_ei(mu, std, y_opt=None, **kwargs):
    values = np.zeros_like(mu)
    ei = expected_improvement(mu, std, y_opt, **kwargs)
    i_max_ei = np.argmax(ei)
    mask = std > 0
    outer = np.sqrt(np.power(std[mask], 2) + np.power(std[i_max_ei], 2))
    inner = (mu[i_max_ei] - mu[mask]) / outer
    values[mask] = outer * _ei_f(inner)
    return values


def expectation(mu, std, **kwargs):
    return -mu


def lcb(mu, std, alpha=1.86, **kwargs):
    if alpha == 'inf':
        return std
    return mu - alpha * std


def max_value_search(mu, std, n_min_samples=1000, **kwargs):
    def probf(x):
        return np.exp(np.sum(st.norm.logcdf(-(x - mu) / std), axis=0))
    idx = np.argmin(mu)
    right = mu[idx].flatten()
    left = right
    i = 0
    while probf(left) < 0.75:
        left = 2. ** i * np.min(mu - 5. * std) + (1. - 2. ** i) * right
        i += 1
    # Binary search for 3 percentiles
    q1, med, q2 = map(lambda val: bisect(lambda x: probf(x) - val, left, right, maxiter=10000, xtol=0.01),
                      [0.25, 0.5, 0.75])
    beta = (q1 - q2) / (np.log(np.log(4. / 3.)) - np.log(np.log(4.)))
    alpha = med + beta * np.log(np.log(2.))
    mins = -np.log(-np.log(np.random.rand(n_min_samples).astype(np.float32))) * beta + alpha

    gamma = (mu[:, None] - mins[None, :]) / std[:, None]
    return np.sum(gamma * st.norm().pdf(gamma) / (2. * st.norm().cdf(gamma)) - st.norm().logcdf(gamma), axis=1)


