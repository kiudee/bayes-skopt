import numpy as np
import pytest
from bask.optimizer import Optimizer
from sklearn.utils.testing import assert_equal
from skopt.benchmarks import bench1


@pytest.fixture
def random_state():
    return np.random.RandomState(123)


def test_multiple_asks():
    # calling ask() multiple times without a tell() inbetween should
    # be a "no op"
    opt = Optimizer(dimensions=[(-2.0, 2.0)], n_initial_points=1)

    opt.run(bench1, n_iter=3, gp_burnin=0, n_samples=1)
    # tell() computes the next point ready for the next call to ask()
    # hence there are three after three iterations
    assert_equal(len(opt.Xi), 3)
    opt.ask()
    assert_equal(len(opt.Xi), 3)
    assert_equal(opt.ask(), opt.ask())


def test_initial_points():
    opt = Optimizer(dimensions=[(-2.0, 2.0)], n_initial_points=5)
    x = opt.ask()
    opt.tell([x], [0.0])
    assert opt._n_initial_points == opt.n_initial_points_ - 1

    opt.tell([x], [0.0])
    assert opt._n_initial_points == opt.n_initial_points_ - 2

    opt.tell([[0.1], [0.2], [0.3]], [0.0, 0.1, 0.2], replace=True)
    assert opt._n_initial_points == opt.n_initial_points_ - 3


def test_noise_vector():
    opt = Optimizer(dimensions=[(-2.0, 2.0)], n_initial_points=5)
    opt.tell(
        [[-1.0], [0.0], [1.0], [0.5]],
        [0.0, -1.0, 0.0, -1.0],
        noise_vector=[1.0, 1.0, 1.0, 0.0],
    )
    x = opt.ask()
    opt.tell([x], [0.0])
    # Test, if the less noisy optimum (at 0.5) had a stronger impact on the mean process
    # than the noisy optimum (at 0.0):
    y_noisy, y = opt.gp.predict([[0.5], [0.625]])
    assert y_noisy > y

    # Check, if passing a single point works correctly:
    x = opt.ask()
    opt.tell(x, 0.0, noise_vector=0.5)


def test_no_error_on_unknown_kwargs():
    opt = Optimizer(dimensions=[(-2.0, 2.0)], n_initial_points=5, unknown_argument=42)


def test_error_on_invalid_priors():
    with pytest.raises(ValueError):
        Optimizer(dimensions=[(-2.0, 2.0)], gp_priors=[])


def test_probability_of_improvement(random_state):
    opt = Optimizer(
        dimensions=[(-2.0, 2.0)], n_initial_points=0, random_state=random_state
    )
    opt.tell(
        [[-2.0], [-1.0], [0.0], [1.0], [2.0]], [2.0, 0.0, -2.0, 0.0, 2.0], gp_burnin=10
    )
    prob = opt.probability_of_optimality(
        threshold=1.0, n_random_starts=20, random_state=random_state
    )
    np.testing.assert_almost_equal(prob, 0.995)

    prob = opt.probability_of_optimality(
        threshold=[0.9, 0.5], n_random_starts=20, random_state=random_state
    )
    np.testing.assert_almost_equal(prob, [0.925, 0.765], decimal=1)


def test_expected_optimality_gap(random_state):
    opt = Optimizer(
        dimensions=[(-2.0, 2.0)], n_initial_points=0, random_state=random_state
    )
    opt.tell(
        [[-2.0], [-1.0], [0.0], [1.0], [2.0]], [2.0, 0.0, -2.0, 0.0, 2.0], gp_burnin=10
    )
    gap = opt.expected_optimality_gap(
        random_state=random_state,
        n_probabilities=10,
        n_space_samples=100,
        n_gp_samples=100,
        n_random_starts=10,
        tol=0.1,
    )
    np.testing.assert_almost_equal(gap, 0.307, decimal=2)
