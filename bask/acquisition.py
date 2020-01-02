from abc import ABC, abstractmethod
import numpy as np
import scipy.stats as st
from scipy.optimize import bisect
from sklearn.utils import check_random_state


from bask.utils import get_progress_bar


__all__ = ["evaluate_acquisitions"]


class Acquisition(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class UncertaintyAcquisition(Acquisition, ABC):
    @abstractmethod
    def __call__(self, mu, std, *args, **kwargs):
        pass


class SampleAcquisition(Acquisition, ABC):
    @abstractmethod
    def __call__(self, gp_sample, *args, **kwargs):
        pass


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
            mu_generated = False
            sample_generated = False
            for j, acq in enumerate(acquisition_functions):
                if isinstance(acq, UncertaintyAcquisition):
                    if not mu_generated:
                        mu_generated = True
                        mu, std = gpr.predict(X, return_std=True)
                    tmp_out = acq(mu, std, **kwargs)
                if isinstance(acq, SampleAcquisition):
                    if not sample_generated:
                        sample_generated = True
                        sample = gpr.sample_y(X, random_state=random_state).flatten()
                    tmp_out = acq(sample, **kwargs)
                if np.all(np.isfinite(tmp_out)):
                    acq_output[j] += tmp_out
        pbar.update(1)
    gpr.theta = param_backup
    return acq_output / n_samples


def _ei_f(x):
    return x * st.norm.cdf(x) + st.norm.pdf(x)


class ExpectedImprovement(UncertaintyAcquisition):
    def __call__(self, mu, std, *args, y_opt=None, **kwargs):
        if y_opt is None:
            y_opt = mu.min()
        values = np.zeros_like(mu)
        mask = std > 0
        inner = (y_opt - mu[mask]) / std[mask]
        values[mask] = _ei_f(inner) * std[mask]
        return values


class TopTwoEI(ExpectedImprovement):
    def __call__(self, mu, std, *args, y_opt=None, **kwargs):
        ei = super().__call__(mu, std, *args, y_opt=y_opt, **kwargs)
        values = np.zeros_like(mu)
        i_max_ei = np.argmax(ei)
        mask = std > 0
        outer = np.sqrt(np.power(std[mask], 2) + np.power(std[i_max_ei], 2))
        inner = (mu[i_max_ei] - mu[mask]) / outer
        values[mask] = outer * _ei_f(inner)
        return values


class Expectation(UncertaintyAcquisition):
    def __call__(self, mu, std, *args, **kwargs):
        return -mu


class LCB(UncertaintyAcquisition):
    def __call__(self, mu, std, *args, alpha=1.86, **kwargs):
        if alpha == 'inf':
            return std
        return alpha * std - mu


class MaxValueSearch(UncertaintyAcquisition):
    def __call__(self, mu, std, *args, n_min_samples=1000, **kwargs):
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


class ThompsonSampling(SampleAcquisition):
    def __call__(self, gp_sample, *args, **kwargs):
        return -gp_sample
