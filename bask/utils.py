import numpy as np
from scipy.spatial.distance import cdist, euclidean
from scipy.stats import halfnorm, invgamma


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