from abc import ABC, abstractmethod
import numpy as np
import scipy.stats as st
from scipy.linalg import cholesky, cho_solve
from scipy.optimize import bisect
from sklearn.utils import check_random_state


from bask.utils import get_progress_bar


__all__ = [
    "evaluate_acquisitions",
    "ExpectedImprovement",
    "TopTwoEI",
    "Expectation",
    "LCB",
    "MaxValueSearch",
    "ThompsonSampling",
    "VarianceReduction",
    "PVRS",
]


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


class FullGPAcquisition(Acquisition, ABC):
    @abstractmethod
    def __call__(self, X, gp, *args, **kwargs):
        pass


def evaluate_acquisitions(
    X,
    gpr,
    acquisition_functions=None,
    n_samples=10,
    progress=False,
    random_state=None,
    **kwargs
):
    """Run a set of acquisitions functions on a given set of points.

    Parameters
    ----------
    X : ndarray, shape (n, d), float
        Set of points for which to evaluate the acquisition functions
    gpr : BayesGPR object
        Gaussian process for which the posterior distribution of the kernel
        hyperparameters is available.
    acquisition_functions : list of Acquisition objects
        List of aquisition functions to evaluate.
        They each should inherit from one of these:
            - :class:`FullGPAcquisition`
            - :class:`UncertaintyAcquisition`
            - :class:`SampleAcquisition`
    n_samples : int, default=10
        Number of posterior samples to draw from the GP. The acquisition
        functions will be evaluated for each of the sampled kernels.
        Exceptions are Acquisition functions inheriting from
        :class:`FullGPAcquisition`.
    progress : bool, default=False
        Show a progress bar
    random_state : int or RandomState or None, optional, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
    kwargs : dict
        Any additional keyword arguments are passed on to each acquisition
        function.

    Returns
    -------
    acq_output : float ndarray, shape (len(acquisition_functions), len(X))
        The acquisition functions evaluated on all of the input points.

    """
    n_cand_points = len(X)
    n_acqs = len(acquisition_functions)
    acq_output = np.zeros((n_acqs, n_cand_points))
    random_state = check_random_state(random_state)
    trace_sample_i = random_state.choice(len(gpr.chain_), replace=False, size=n_samples)
    param_backup = np.copy(gpr.theta)
    pbar = get_progress_bar(progress, len(trace_sample_i))
    for i_acq, acq in enumerate(acquisition_functions):
        if isinstance(acq, FullGPAcquisition):
            out = acq(X, gpr, random_state=random_state, **kwargs)
            if np.all(np.isfinite(out)):
                acq_output[i_acq] = out
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
                elif isinstance(acq, SampleAcquisition):
                    if not sample_generated:
                        sample_generated = True
                        sample = gpr.sample_y(X, random_state=random_state).flatten()
                    tmp_out = acq(sample, **kwargs)
                else:
                    continue
                if np.all(np.isfinite(tmp_out)):
                    acq_output[j] += tmp_out / n_samples
        pbar.update(1)
    gpr.theta = param_backup
    return acq_output


def _ei_f(x):
    return x * st.norm.cdf(x) + st.norm.pdf(x)


class ExpectedImprovement(UncertaintyAcquisition):
    """Select the point maximizing the expected improvement over the current
    optimum.

    Parameters
    ----------
    y_opt : float, default=None
        The value of the current optimum. If it is None, it will use the
        minimum y value of the evaluated points.
    """

    def __call__(self, mu, std, *args, y_opt=None, **kwargs):
        if y_opt is None:
            y_opt = mu.min()
        values = np.zeros_like(mu)
        mask = std > 0
        inner = (y_opt - mu[mask]) / std[mask]
        values[mask] = _ei_f(inner) * std[mask]
        return values


class TopTwoEI(ExpectedImprovement):
    """Select the point with the highest expected improvement over the
    point with the maximum expected improvement overall.

    Parameters
    ----------
    y_opt : float, default=None
        The value of the current optimum. If it is None, it will use the
        minimum y value of the evaluated points.
    """

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
    """Select the point with the lowest estimated mean."""

    def __call__(self, mu, std, *args, **kwargs):
        return -mu


class LCB(UncertaintyAcquisition):
    """Select the point with the lowest lower confidence bound.

    Parameters
    ----------
    alpha : positive float, alpha=1.96
        Number of standard errors to substract from the mean estimate.
    """

    def __call__(self, mu, std, *args, alpha=1.96, **kwargs):
        if alpha == "inf":
            return std
        return alpha * std - mu


class MaxValueSearch(UncertaintyAcquisition):
    """Select points based on their mutual information with the optimum value.

    Parameters
    ----------
    n_min_samples : int, default=1000
        Number of samples for the optimum distribution

    References
    ----------
    [1] Wang, Z. & Jegelka, S.. (2017). Max-value Entropy Search for Efficient
        Bayesian Optimization. Proceedings of the 34th International Conference
        on Machine Learning, in PMLR 70:3627-3635
    """

    def __call__(self, mu, std, *args, n_min_samples=1000, **kwargs):
        def probf(x):
            return np.exp(np.sum(st.norm.logcdf(-(x - mu) / std), axis=0))

        idx = np.argmin(mu)
        right = mu[idx].flatten()
        left = right
        i = 0
        while probf(left) < 0.75:
            left = 2.0 ** i * np.min(mu - 5.0 * std) + (1.0 - 2.0 ** i) * right
            i += 1
        # Binary search for 3 percentiles
        q1, med, q2 = map(
            lambda val: bisect(
                lambda x: probf(x) - val, left, right, maxiter=10000, xtol=0.01
            ),
            [0.25, 0.5, 0.75],
        )
        beta = (q1 - q2) / (np.log(np.log(4.0 / 3.0)) - np.log(np.log(4.0)))
        alpha = med + beta * np.log(np.log(2.0))
        mins = (
            -np.log(-np.log(np.random.rand(n_min_samples).astype(np.float32))) * beta
            + alpha
        )

        gamma = (mu[:, None] - mins[None, :]) / std[:, None]
        return np.sum(
            gamma * st.norm().pdf(gamma) / (2.0 * st.norm().cdf(gamma))
            - st.norm().logcdf(gamma),
            axis=1,
        )


class ThompsonSampling(SampleAcquisition):
    """Sample a random function from the GP and select its optimum."""

    def __call__(self, gp_sample, *args, **kwargs):
        return -gp_sample


class VarianceReduction(FullGPAcquisition):
    """A criterion which tries to find the region where it can reduce the
    global variance the most.

    This criterion is suitable for active learning, where the goal is to
    uniformly estimate the target function and not only its optimum.
    """

    def __call__(self, X, gp, *args, **kwargs):
        n = len(X)
        covs = np.empty(n)
        for i in range(n):
            X_train_aug = np.concatenate([gp.X_train_, [X[i]]])
            K = gp.kernel_(X_train_aug)
            if np.iterable(gp.alpha):
                K[np.diag_indices_from(K)] += np.concatenate([gp.alpha, [0.0]])
            L = cholesky(K, lower=True)
            K_trans = gp.kernel_(X, X_train_aug)
            v = cho_solve((L, True), K_trans.T)
            cov = K_trans.dot(v)
            covs[i] = np.diag(cov).sum()
        return covs


class PVRS(FullGPAcquisition):
    """Implements the predictive variance reduction search algorithm.

    The algorithm draws a set of Thompson samples (samples from the optimum
    distribution) and proposes the point which reduces the predictive variance
    of these samples the most.

    References
    ----------
    [1] Nguyen, Vu, et al. "Predictive variance reduction search." Workshop on
    Bayesian optimization at neural information processing systems (NIPSW). 2017.
    """

    def __call__(self, X, gp, *args, n_thompson=10, random_state=None, **kwargs):
        n = len(X)
        thompson_sample = gp.sample_y(
            X, sample_mean=True, n_samples=n_thompson, random_state=random_state
        )
        thompson_points = np.array(X)[np.argmin(thompson_sample, axis=0)]
        covs = np.empty(n)
        for i in range(n):
            X_train_aug = np.concatenate([gp.X_train_, [X[i]]])
            K = gp.kernel_(X_train_aug)
            if np.iterable(gp.alpha):
                K[np.diag_indices_from(K)] += np.concatenate([gp.alpha, [0.0]])
            L = cholesky(K, lower=True)
            K_trans = gp.kernel_(thompson_points, X_train_aug)
            v = cho_solve((L, True), K_trans.T)
            cov = K_trans.dot(v)
            covs[i] = np.diag(cov).sum()
        return covs
