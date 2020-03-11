from numpy.testing import assert_almost_equal
from sklearn.gaussian_process.kernels import CompoundKernel
from skopt.learning.gaussian_process.kernels import (
    ExpSineSquared,
    Matern,
    ConstantKernel,
    WhiteKernel,
    Exponentiation,
    DotProduct,
    Product,
    RationalQuadratic,
    RBF,
)
from bask.utils import guess_priors, construct_default_kernel


def test_construct_default_kernel():
    kernel = construct_default_kernel([0, 1])
    assert len(kernel.theta) == 3


def test_guess_priors():
    """Construct a complicated kernel and check if priors are constructed
    correctly."""
    kernel = Exponentiation(
        ConstantKernel(constant_value_bounds="fixed") * Matern()
        + WhiteKernel()
        + CompoundKernel([RBF(), Matern()]),
        2.0,
    )

    priors = guess_priors(kernel)

    assert len(priors) == 4
    expected = [
        -1.737085713764618,
        -4.107091211892862,
        -1.737085713764618,
        -1.737085713764618,
    ]
    for p, v in zip(priors, expected):
        assert_almost_equal(p(0.0), v)
