import numpy as np
import pytest
from scipy.stats import halfnorm, invgamma
from skopt.learning.gaussian_process.kernels import ConstantKernel, RBF
from bask.acquisition import evaluate_acquisitions, PVRS, VarianceReduction
from bask.bayesgpr import BayesGPR


@pytest.fixture
def minimal_gp():
    kernel = ConstantKernel(
        constant_value=1 ** 2, constant_value_bounds=(0.01 ** 2, 1 ** 2)
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


def test_variance_reduction(minimal_gp, minimal_priors):
    x = np.array([-2.0, -1.0, 1.0, 2.0])[:, None]
    y = np.array([0, -1, 1, 2])

    minimal_gp.fit(x, y, priors=minimal_priors, progress=False, n_burnin=1)

    acq = evaluate_acquisitions(
        X=np.array([-2.0, -1.0, 0.0, 1.0, 2.0])[:, None],
        gpr=minimal_gp,
        acquisition_functions=[VarianceReduction()],
        random_state=1,
        n_samples=0,
    )
    assert np.argmax(acq) == 2


def test_pvrs(minimal_gp, minimal_priors):
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])[:, None]
    y = np.array([0, -1, 0, 1, 2])

    minimal_gp.fit(x, y, priors=minimal_priors, progress=False, n_burnin=1)

    acq = evaluate_acquisitions(
        x, gpr=minimal_gp, acquisition_functions=[PVRS()], random_state=1, n_samples=0
    )
    assert np.argmax(acq) == 1
