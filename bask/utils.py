import collections

import numpy as np
from scipy.spatial.distance import cdist, euclidean
from scipy.stats import halfnorm
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern

# We import r2_sequence here for backwards compatibility reasons:
from bask.init import r2_sequence
from bask.priors import make_roundflat

__all__ = [
    "geometric_median",
    "r2_sequence",
    "guess_priors",
    "construct_default_kernel",
    "validate_zeroone",
]


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
        return tqdm.tqdm(total=total)  # noqa: F821
    else:
        return _NoOpPBar()


def validate_zeroone(arr):
    """Check if all entries of the input are between 0 and 1.

    Parameters
    ----------
    X : ndarray
        Array containing arbitrary values.

    Raises
    ------
    ValueError
        If the values of the array are not between 0 and 1 (inclusive).
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if np.any(arr < 0) or np.any(arr > 1):
        raise ValueError("Not all values of the array are between 0 and 1.")
    return
