import numpy as np
import pytest
from scipy.stats import halfnorm, invgamma
from skopt.learning.gaussian_process.kernels import RBF, ConstantKernel

from bask.acquisition import (
    LCB,
    PVRS,
    Expectation,
    ExpectedImprovement,
    MaxValueSearch,
    ThompsonSampling,
    TopTwoEI,
    VarianceReduction,
    evaluate_acquisitions,
)
from bask.bayesgpr import BayesGPR


@pytest.fixture
def minimal_gp():
    kernel = ConstantKernel(
        constant_value=1**2, constant_value_bounds=(0.01**2, 1**2)
    ) * RBF(length_scale=1.0, length_scale_bounds=(0.5, 1.5))
    gp = BayesGPR(random_state=1, normalize_y=False, kernel=kernel)
    return gp


@pytest.fixture
def minimal_priors():
    return [
        lambda x: halfnorm(scale=1.0).logpdf(np.sqrt(np.exp(x)))
        + x / 2.0
        - np.log(2.0),
        lambda x: invgamma(a=5.0, scale=1.0).logpdf(np.exp(x)) + x,
        lambda x: halfnorm(scale=1.0).logpdf(np.sqrt(np.exp(x)))
        + x / 2.0
        - np.log(2.0),
    ]


@pytest.mark.parametrize(
    "acq_func, n_samples, expected",
    [
        (MaxValueSearch, 1, 37),
        (VarianceReduction, 0, 50),
        (PVRS, 0, 38),
        (LCB, 1, 38),
        (ExpectedImprovement, 1, 33),
        (Expectation, 1, 30),
        (ThompsonSampling, 1, 25),
        (TopTwoEI, 1, 32),
    ],
)
def test_acquisition(acq_func, n_samples, expected, minimal_gp, minimal_priors):
    x = np.array([-2.0, -1.0, 1.0, 2.0])[:, None]
    y = np.array([0, -1, 1, 2])

    minimal_gp.fit(x, y, priors=minimal_priors, progress=False, n_burnin=1)

    x = np.linspace(-2.0, 2.0, num=101)[:, None]

    acq = evaluate_acquisitions(
        X=x,
        gpr=minimal_gp,
        acquisition_functions=[acq_func()],
        random_state=1,
        n_samples=n_samples,
    )
    assert np.argmax(acq) == expected
