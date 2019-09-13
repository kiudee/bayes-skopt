import numpy as np
from scipy.spatial.distance import cdist, euclidean
from scipy.stats import halfnorm, invgamma


__all__ = ["geometric_median", "r2_sequence", "guess_priors"]


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


def guess_priors(n_parameters):
    priors = [lambda x: halfnorm(scale=2.0).logpdf(np.sqrt(np.exp(x))) + x / 2.0 - np.log(2.0)]
    priors.extend([lambda x: invgamma(a=8.92, scale=1.73).logpdf(np.exp(x)) + x for _ in range(n_parameters)])
    priors.append(lambda x: halfnorm(scale=1.0).logpdf(np.sqrt(np.exp(x))) + x / 2.0 - np.log(2.0))
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
