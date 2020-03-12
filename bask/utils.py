import numpy as np
from scipy.spatial.distance import cdist, euclidean
from scipy.stats import halfnorm, invgamma
from skopt.learning.gaussian_process.kernels import Matern, ConstantKernel


__all__ = [
    "geometric_median",
    "r2_sequence",
    "guess_priors",
    "construct_default_kernel",
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
        for k in kernel.kernels:
            _recursive_priors(k, prior_list)
    else:
        name = type(kernel).__name__
        if name in ["ConstantKernel", "WhiteKernel"]:
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
            # Here we apply an inverse gamma distribution to any lengthscale
            # parameter we find. We assume the input variables are normalized
            # to lie in [0, 1]. The specific values for a and scale were
            # obtained by fitting the 1% and 99% quantile to 0.15 and 0.8.
            prior_list.append(
                lambda x: invgamma(a=8.286, scale=2.4605).logpdf(np.exp(x)) + x,
            )
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
        length_scale=[0.3] * n_parameters, length_scale_bounds=(0.05, 1.0), nu=2.5
    )
    return kernel


def guess_priors(kernel):
    """Guess suitable priors for the hyperparameters of a given kernel.

    This function recursively explores the given (composite) kernel and
    adds suitable priors each encountered hyperparameter.

    Here we use a half-Normal(0, 2.0) prior for all ConstantKernels and
    WhiteKernels, and an invGamma(a=8.286, scale=2.4605) prior for all
    lengthscales. Change of variables is applied, since inference is done in
    log-space.

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
