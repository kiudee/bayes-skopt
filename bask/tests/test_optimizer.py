from bask.optimizer import Optimizer
from sklearn.utils.testing import assert_equal
from skopt.benchmarks import bench1


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
