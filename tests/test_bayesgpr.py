import numpy as np
import pytest
from scipy.stats import halfnorm, invgamma
from bask.bayesgpr import BayesGPR
from skopt.learning.gaussian_process.kernels import ConstantKernel, RBF


@pytest.fixture
def minimal_gp():
    kernel = (
        ConstantKernel(constant_value=1 ** 2, constant_value_bounds=(0.01 ** 2, 1 ** 2))
        * RBF(length_scale=1.0, length_scale_bounds=(0.5, 1.5))
    )
    gp = BayesGPR(random_state=1, normalize_y=False, kernel=kernel)
    return gp


@pytest.fixture
def minimal_priors():
    return [
        lambda x: halfnorm(scale=1.).logpdf(np.sqrt(np.exp(x))) + x / 2.0 - np.log(2.0),
        lambda x: invgamma(a=5.0, scale=1.).logpdf(np.exp(x)) + x,
        lambda x: halfnorm(scale=1.).logpdf(np.sqrt(np.exp(x))) + x / 2.0 - np.log(2.0)
    ]


def test_noise_vector(minimal_gp, minimal_priors):
    X = np.array([[0.0], [0.0]])
    y = np.array([1.0, 0.0])
    noise_vector = np.array([1234, 0.0])
    minimal_gp.fit(X, y, noise_vector=noise_vector, n_burnin=1, progress=False, priors=minimal_priors)
    prediction = minimal_gp.predict(np.array([[0.0]]))
    assert prediction < 0.01  # The high noise is supposed to diminish the effect of the datapoint


def test_noise_set_to_zero(minimal_gp, minimal_priors):
    X = np.array([[0.1], [0.0], [-0.1]])
    y = np.array([0.0, 0.0, 0.0])
    minimal_gp.fit(X, y, n_burnin=1, progress=False, priors=minimal_priors)
    minimal_gp.theta = np.array([0.0, 0.0, 0.0])
    assert minimal_gp.predict(np.array([[0.0]]), return_std=True)[1] >= 1.0
    with minimal_gp.noise_set_to_zero():
        assert minimal_gp.predict(np.array([[0.0]]), return_std=True)[1] < 1.0
    assert minimal_gp.predict(np.array([[0.0]]), return_std=True)[1] >= 1.0


def test_sample_without_fit(minimal_gp):
    # Calling sample without data (X, y) or a previous fit, should raise a ValueError:
    with pytest.raises(ValueError):
        minimal_gp.sample()


