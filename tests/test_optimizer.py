import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.utils.testing import assert_equal
from skopt.benchmarks import bench1

from bask.optimizer import Optimizer


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


@pytest.mark.parametrize("init_strategy", ("r2", "sb", "random"))
def test_initial_points(init_strategy):
    opt = Optimizer(
        dimensions=[(-2.0, 2.0)], n_initial_points=3, init_strategy=init_strategy
    )
    x = opt.ask()
    assert not isinstance(x[0], list)
    opt.tell([x], [0.0])
    assert opt._n_initial_points == opt.n_initial_points_ - 1

    opt.tell([x], [0.0])
    assert opt._n_initial_points == opt.n_initial_points_ - 2
    assert opt.gp.chain_ is None

    opt.tell([[0.1], [0.2], [0.3]], [0.0, 0.1, 0.2], replace=True)
    assert opt._n_initial_points == opt.n_initial_points_ - 3
    assert opt.gp.chain_ is not None


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


def test_run_with_noise(random_state):
    def func(x):
        return (np.sin(x) + random_state.randn()).item(), 1.0

    opt = Optimizer(dimensions=[(-2.0, 2.0)], n_initial_points=1)
    opt.run(func, n_iter=2, n_samples=1, gp_burnin=0)
    assert_almost_equal(opt.gp.alpha, np.ones(2))


def test_no_error_on_unknown_kwargs():
    Optimizer(dimensions=[(-2.0, 2.0)], n_initial_points=5, unknown_argument=42)


def test_error_on_invalid_priors():
    opt = Optimizer(dimensions=[(-2.0, 2.0)], gp_priors=[], n_initial_points=0)
    with pytest.raises(ValueError):
        opt.tell([(0.0,)], 0.0)


@pytest.mark.parametrize(
    "input,expected",
    [
        (dict(normalized_scores=False, threshold=1.0), 0.995),
        (dict(normalized_scores=False, threshold=(0.9, 0.5)), (0.99, 0.935)),
        (dict(normalized_scores=True, threshold=1.0), 0.99),
    ],
)
def test_probability_of_improvement(random_state, input, expected):
    opt = Optimizer(
        dimensions=[(-2.0, 2.0)], n_initial_points=0, random_state=random_state
    )
    opt.tell(
        [[-2.0], [-1.0], [0.0], [1.0], [2.0]], [2.0, 0.0, -2.0, 0.0, 2.0], gp_burnin=10
    )
    prob = opt.probability_of_optimality(
        threshold=input["threshold"],
        n_random_starts=20,
        random_state=random_state,
        normalized_scores=input["normalized_scores"],
    )
    np.testing.assert_almost_equal(prob, expected, decimal=2)


@pytest.mark.parametrize(
    "input,expected",
    [
        (dict(normalized_scores=False, use_mean_gp=True), 0.2),
        (dict(normalized_scores=True, use_mean_gp=True), 0.16),
        (dict(normalized_scores=True, use_mean_gp=False), 0.17),
    ],
)
def test_expected_optimality_gap(random_state, input, expected):
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
        use_mean_gp=input["use_mean_gp"],
        normalized_scores=input["normalized_scores"],
    )
    np.testing.assert_almost_equal(gap, expected, decimal=2)


def test_optimum_intervals():
    opt = Optimizer(
        dimensions=[(0.0, 1.0)], random_state=0, acq_func="mean", n_points=1
    )
    x = np.linspace(0, 1, num=30)[:, None]
    y = np.cos(np.pi * 4 * x).flatten()
    opt.tell(x.tolist(), y.tolist(), gp_burnin=0, progress=False, n_samples=0)

    intervals = opt.optimum_intervals(random_state=0, space_samples=100)
    assert len(intervals) == 1
    assert len(intervals[0]) == 2
    assert len(intervals[0][0]) == 2
    intervals = opt.optimum_intervals(
        random_state=0, space_samples=100, multimodal=False
    )
    assert len(intervals) == 1
    assert len(intervals[0]) == 2
