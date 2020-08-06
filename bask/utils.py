import collections

from arviz import hdi
import numpy as np
from scipy.spatial.distance import cdist, euclidean
from scipy.stats import halfnorm, invgamma
from skopt.learning.gaussian_process.kernels import Matern, ConstantKernel

from bask.priors import make_roundflat


__all__ = [
    "optimum_intervals",
    "geometric_median",
    "r2_sequence",
    "guess_priors",
    "construct_default_kernel",
]


def optimum_intervals(
    optimizer,
    hdi_prob=0.95,
    multimodal=True,
    opt_samples=200,
    space_samples=500,
    only_mean=True,
    random_state=None,
):
    """Estimate highest density intervals for the optimum.

    Employs Thompson sampling to obtain samples from the optimum distribution.
    For each dimension separately, it will then estimate highest density
    intervals.

    Parameters
    ----------
    optimizer : bask.Optimizer object
        The optimizer instance for which the highest density intervals are to
        be computed.
    hdi_prob : float, default=0.95
        The total probability each interval should cover.
    multimodal : bool, default=True
        If True, more than one interval can be returned for one parameter.
    opt_samples : int, default=200
        Number of samples to generate from the optimum distribution.
    space_samples : int, default=500
        Number of samples to cover the optimization space with.
    only_mean : bool, default=True
        If True, it will only sample optima from the mean Gaussian process.
        This is usually faster, but can underestimate the uncertainty.
        If False, it will also sample the hyperposterior of the kernel parameters.
    random_state : int, RandomState instance or None, optional (default: None)
        The generator used to initialize the centers. If int, random_state is
        the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.

    Returns
    -------
    intervals : list of ndarray
        Outputs an array of size (n_modes, 2) for each dimension in the
        optimization space.

    Raises
    ------
    NotImplementedError
        If the user calls the function with an optimizer containing at least one
        categorical parameter.
    """
    if optimizer.space.is_partly_categorical:
        raise NotImplementedError(
            "Highest density interval not implemented for categorical parameters."
        )
    X = optimizer.space.rvs(n_samples=space_samples, random_state=random_state)
    X = optimizer.space.transform(X)
    optimum_samples = optimizer.gp.sample_y(
        X, sample_mean=only_mean, n_samples=opt_samples, random_state=random_state
    )
    X_opt = X[np.argmin(optimum_samples, axis=0)]

    intervals = []
    for i, col in enumerate(X_opt.T):
        raw_interval = hdi(col, hdi_prob=hdi_prob, multimodal=multimodal)
        intervals.append(optimizer.space.dimensions[i].inverse_transform(raw_interval))
    return intervals


def geometric_median(X, eps=1e-5):
    """Compute the geometric median for the given array of points.

    The geometric median is the point minimizing the euclidean (L2) distance
    to all points.


    Parameters
    ----------
    X : numpy array
        (n_points, n_dim)
    eps : float
        Stop the computation if the euclidean distance of the last two
        computed points is smaller than eps

    Returns
    -------

    """
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < eps:
            return y1

        y = y1


def _recursive_priors(kernel, prior_list):
    if hasattr(kernel, "kernel"):  # Unary operations
        _recursive_priors(kernel.kernel, prior_list)
    elif hasattr(kernel, "k1"):  # Binary operations
        _recursive_priors(kernel.k1, prior_list)
        _recursive_priors(kernel.k2, prior_list)
    elif hasattr(kernel, "kernels"):  # CompoundKernel
        # It seems that the skopt kernels are not compatible with the
        # CompoundKernel. This is therefore not officially supported.
        for k in kernel.kernels:
            _recursive_priors(k, prior_list)
    else:
        name = type(kernel).__name__
        if name in ["ConstantKernel", "WhiteKernel"]:
            if name == "ConstantKernel" and kernel.constant_value_bounds == "fixed":
                return
            if name == "WhiteKernel" and kernel.noise_level_bounds == "fixed":
                return
            # We use a half-normal prior distribution on the signal variance and
            # noise. The input x is sampled in log-space, which is why the
            # change of variables is necessary.
            # This prior assumes that the function values are standardized.
            # Note, that we do not know the structure of the kernel, which is
            # why this is just only a best guess.
            prior_list.append(
                lambda x: halfnorm(scale=2.0).logpdf(np.sqrt(np.exp(x)))
                + x / 2.0
                - np.log(2.0),
            )
        elif name in ["Matern", "RBF"]:
            # Here we apply a round-flat prior distribution to any lengthscale
            # parameter we find. We assume the input variables are normalized
            # to lie in [0, 1].
            # For common optimization problems, we expect the lengthscales to
            # lie in the range [0.1, 0.6]. The round-flat prior allows values
            # outside the range, if supported by enough datapoints.
            if isinstance(kernel.length_scale, (collections.Sequence, np.ndarray)):
                n_priors = len(kernel.length_scale)
            else:
                n_priors = 1
            roundflat = make_roundflat(
                lower_bound=0.1,
                upper_bound=0.6,
                lower_steepness=2.0,
                upper_steepness=8.0,
            )
            for _ in range(n_priors):
                prior_list.append(lambda x: roundflat(np.exp(x)) + x)
        else:
            raise NotImplementedError(
                f"Unable to guess priors for this kernel: {kernel}."
            )


def construct_default_kernel(dimensions):
    """Construct a Matern kernel as default kernel to be used in the optimizer.

    Parameters
    ----------
    dimensions : list of dimensions
        Elements are skopt.space.Dimension instances (Real, Integer
        or Categorical) or any other valid value that defines skopt
        dimension (see skopt.Optimizer docs)

    Returns
    -------
    kernel : kernel object
        The kernel specifying the covariance function of the GP used in the
        optimization.
    """
    n_parameters = len(dimensions)
    kernel = ConstantKernel(
        constant_value=1.0, constant_value_bounds=(0.1, 2.0)
    ) * Matern(
        length_scale=[0.3] * n_parameters, length_scale_bounds=(0.2, 0.5), nu=2.5
    )
    return kernel


def guess_priors(kernel):
    """Guess suitable priors for the hyperparameters of a given kernel.

    This function recursively explores the given (composite) kernel and
    adds suitable priors each encountered hyperparameter.

    Here we use a half-Normal(0, 2.0) prior for all ConstantKernels and
    WhiteKernels, and an round-flat(0.1, 0.6) prior for all lengthscales.
    Change of variables is applied, since inference is done in log-space.

    Parameters
    ----------
    kernel : Kernel object.
        Can be a single kernel (e.g. Matern), a Product or Sum kernel, or a
        CompoundKernel.

    Returns
    -------
    priors : list of functions.
        The function returns the list of priors in the same order as the vector
        theta provided by the kernel. Each prior evaluates the logpdf of its
        argument.
    """
    priors = []
    _recursive_priors(kernel, priors)
    return priors


def phi(d, n_iter=10):
    if d == 1:
        return 1.61803398874989484820458683436563
    elif d == 2:
        return 1.32471795724474602596090885447809
    x = 2.0000
    for i in range(n_iter):
        x = pow(1 + x, 1 / (d + 1))
    return x


def r2_sequence(n, d, seed=0.5):
    """Output ``n`` points of the infinite R2 quasi-random sequence.

    Parameters
    ----------
    n : int
        Number of points to generate
    d : int
        Number of dimensions for each point
    seed : float in [0, 1], default=0.5
        Seed value for the sequence

    Returns
    -------
    z : ndarray, shape (n, d)
        ``n`` points of the R2 sequence
    """
    g = phi(d)
    alpha = np.zeros(d)
    for j in range(d):
        alpha[j] = pow(1 / g, j + 1) % 1
    z = np.zeros((n, d))

    for i in range(n):
        z[i] = (seed + alpha * (i + 1)) % 1
    return z


class _NoOpPBar(object):
    """This class implements the progress bar interface but does nothing"""

    def __init__(self):
        pass

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def update(self, count):
        pass


def get_progress_bar(display, total):
    """Get a progress bar interface with given properties
    If the tqdm library is not installed, this will always return a "progress
    bar" that does nothing.
    Args:
        display (bool or str): Should the bar actually show the progress? Or a
                               string to indicate which tqdm bar to use.
        total (int): The total size of the progress bar.
    """
    if display is True:
        return tqdm.tqdm(total=total)
    else:
        return _NoOpPBar()
