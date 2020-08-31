import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from skopt.learning.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    Exponentiation,
    Matern,
    WhiteKernel,
)

from bask.utils import construct_default_kernel, guess_priors, validate_zeroone


def test_construct_default_kernel():
    kernel = construct_default_kernel([0, 1])
    assert len(kernel.theta) == 3


def test_guess_priors():
    """Construct a complicated kernel and check if priors are constructed
    correctly."""
    kernel = Exponentiation(
        ConstantKernel(constant_value_bounds="fixed") * Matern()
        + WhiteKernel()
        + RBF(length_scale=(1.0, 1.0)),
        2.0,
    )

    priors = guess_priors(kernel)

    assert len(priors) == 4
    expected = [
        -0.02116327824572739,
        -2.112906921232193,
        -0.02116327824572739,
        -0.02116327824572739,
    ]
    for p, v in zip(priors, expected):
        assert_almost_equal(p(-0.9), v)


def test_validate_zeroone():
    validate_zeroone(np.linspace(0, 1, num=10))
    validate_zeroone([0.0, 0.5, 1.0])
    with pytest.raises(ValueError):
        validate_zeroone(np.array((0.1, -0.1)))
    with pytest.raises(ValueError):
        validate_zeroone(np.array((0.1, np.inf)))
