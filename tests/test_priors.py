import numpy as np
from numpy.testing import assert_almost_equal
from scipy.integrate import quad

from bask.priors import make_roundflat


def test_make_roundflat():
    prior = make_roundflat()
    value = quad(lambda x: np.exp(prior(x)), 0.0, 10.0)[0]
    assert_almost_equal(value, 1.0)
