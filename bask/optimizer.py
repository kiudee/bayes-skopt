import warnings
import numpy as np
from sklearn.utils import check_random_state
from skopt.utils import create_result, normalize_dimensions, is_listlike, is_2Dlistlike

from . import acquisition
from .bayesgpr import BayesGPR
from .utils import r2_sequence, guess_priors, construct_default_kernel
from bask.acquisition import evaluate_acquisitions

__all__ = ["Optimizer"]

ACQUISITION_FUNC = {
    "ei": acquisition.ExpectedImprovement(),
    "lcb": acquisition.LCB(),
    "mean": acquisition.Expectation(),
    "mes": acquisition.MaxValueSearch(),
    "pvrs": acquisition.PVRS(),
    "ts": acquisition.ThompsonSampling(),
    "ttei": acquisition.TopTwoEI(),
    "vr": acquisition.VarianceReduction(),
}


class Optimizer(object):
    """Execute a stepwise Bayesian optimization.

    Parameters
    ----------
    dimensions : list, shape (n_dims,)
        List of search space dimensions.
        Each search dimension can be defined either as

        - a `(lower_bound, upper_bound)` tuple (for `Real` or `Integer`
          dimensions),
        - a `(lower_bound, upper_bound, "prior")` tuple (for `Real`
          dimensions),
        - as a list of categories (for `Categorical` dimensions), or
        - an instance of a `Dimension` object (`Real`, `Integer` or
          `Categorical`).
    n_points : int, default=500
        Number of random points to evaluate the acquisition function on.
    n_initial_points : int, default=10
        Number of initial points to sample before fitting the GP.
    init_strategy : string or None, default="r2"
        Sampling strategy to use for the initial ``n_initial_points``.
        "r2" computes points using the quasirandom R2 sequence. If the value
        is None or any other string, uniform random sampling is employed.
    gp_kernel : kernel object
        The kernel specifying the covariance function of the GP. If None is
        passed, a suitable default kernel is constructed.
        Note that the kernel’s hyperparameters are estimated using MCMC during
        fitting.
    gp_kwargs : dict, optional
        Dict of arguments passed to :class:`BayesGPR`.  For example,
        ``{'normalize_y': True}`` would allow the GP to normalize the output
        values before fitting.
    acq_func : string or Acquisition object, default="pvrs"
        Acquisition function to use as a criterion to select new points to test.
        By default we use "pvrs", which is a very robust criterion with fast
        convergence.
        Should be one of
            - 'pvrs' Predictive variance reductions search
            - 'mes' Max-value entropy search
            - 'ei' Expected improvement
            - 'ttei' Top-two expected improvement
            - 'lcb' Lower confidence bound
            - 'mean' Expected value of the GP
            - 'ts' Thompson sampling
            - 'vr' Global variance reduction
        Can also be a custom :class:`Acquisition` object.
    acq_func_kwargs : dict, optional
        Dict of arguments passed to :class:`Acquisition`.
    random_state : int or RandomState or None, optional, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.

    Attributes
    ----------
    Xi : list
        Points at which objective has been evaluated.
    yi : scalar
        Values of objective at corresponding points in `Xi`.
    space : Space
        An instance of :class:`skopt.space.Space`. Stores parameter search
        space used to sample points, bounds, and type of parameters.
    gp : BayesGPR object
        The current underlying GP model, which is used to calculate the
        acquisition function.
    gp_priors : list of callables
        List of prior distributions for the kernel hyperparameters of the GP.
        Each callable returns the logpdf of the prior distribution.
    n_initial_points_ : int
        Number of initial points to sample
    noisei : list of floats
        Additional pointwise noise which is added to the diagonal of the
        kernel matrix
    """
    def __init__(
        self,
        dimensions,
        n_points=500,
        n_initial_points=10,
        init_strategy="r2",
        gp_kernel=None,
        gp_kwargs=None,
        gp_priors=None,
        acq_func="pvrs",
        acq_func_kwargs=None,
        random_state=None,
    ):
        self.rng = check_random_state(random_state)

        if callable(acq_func):
            self.acq_func = acq_func
        else:
            self.acq_func = ACQUISITION_FUNC[acq_func]
        if acq_func_kwargs is None:
            acq_func_kwargs = dict()
        self.acq_func_kwargs = acq_func_kwargs

        self.space = normalize_dimensions(dimensions)
        self._n_initial_points = n_initial_points
        self.n_initial_points_ = n_initial_points
        self.init_strategy = init_strategy
        if self.init_strategy == "r2":
            self._initial_points = self.space.inverse_transform(
                r2_sequence(n=n_initial_points, d=self.space.n_dims)
            )
        self.n_points = n_points

        if gp_kwargs is None:
            gp_kwargs = dict()
        if gp_kernel is None:
            # For now the default kernel is not adapted to the dimensions,
            # which is why a simple list is passed:
            gp_kernel = construct_default_kernel(
                list(range(self.space.transformed_n_dims))
            )

        self.gp = BayesGPR(
            kernel=gp_kernel,
            random_state=self.rng.randint(0, np.iinfo(np.int32).max),
            **gp_kwargs,
        )
        # We are only able to guess priors now, since BayesGPR can add
        # another WhiteKernel, when noise is set to "gaussian":
        if gp_priors is None:
            gp_priors = guess_priors(self.gp.kernel)
        self.gp_priors = gp_priors

        self.Xi = []
        self.yi = []
        self.noisei = []
        self._next_x = None

    def ask(self, n_points=1):
        if n_points > 1:
            raise NotImplementedError(
                "Returning multiple points is not implemented yet."
            )
        if (
            self._n_initial_points > 0
        ):  # TODO: Make sure estimator is trained here always
            if self.init_strategy == "r2":
                return self._initial_points[self._n_initial_points - 1]
            return self.space.rvs()
        else:
            if not self.gp.kernel_:
                raise RuntimeError(
                    "Initialization is finished, but no model has been fit."
                )
            return self._next_x

    def tell(
        self,
        x,
        y,
        noise_vector=None,
        fit=True,
        replace=False,
        n_samples=0,
        gp_samples=100,
        gp_burnin=10,
        progress=False,
    ):
        # if y isn't a scalar it means we have been handed a batch of points

        # TODO (noise vector):
        #  1. Replace case should be easy
        #  2. Add case should add noise values to list
        #  -> What if noise_vector is None? (have to set noise to 0)
        if replace:
            self.Xi = []
            self.yi = []
            self.noisei = []
            self._n_initial_points = self.n_initial_points_
        if is_listlike(y) and is_2Dlistlike(x):
            self.Xi.extend(x)
            self.yi.extend(y)
            if noise_vector is None:
                noise_vector = [0.0] * len(y)
            elif not is_listlike(noise_vector) or len(noise_vector) != len(y):
                raise ValueError(
                    f"Vector of noise variances needs to be of equal length as `y`."
                )
            self.noisei.extend(noise_vector)
            self._n_initial_points -= len(y)
        elif is_listlike(x):
            self.Xi.append(x)
            self.yi.append(y)
            if noise_vector is None:
                noise_vector = 0.0
            elif is_listlike(noise_vector):
                raise ValueError(
                    f"Vector of noise variances is a list, while tell only received one datapoint."
                )
            self.noisei.append(noise_vector)
            self._n_initial_points -= 1
        else:
            raise ValueError(
                f"Type of arguments `x` ({type(x)}) and `y` ({type(y)}) "
                "not compatible."
            )

        if fit and self._n_initial_points <= 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.gp.pos_ is None or replace:
                    self.gp.fit(
                        self.space.transform(self.Xi),
                        self.yi,
                        noise_vector=np.array(self.noisei),
                        priors=self.gp_priors,
                        n_desired_samples=gp_samples,
                        n_burnin=gp_burnin,
                        progress=progress,
                    )
                else:
                    self.gp.sample(
                        self.space.transform(self.Xi),
                        self.yi,
                        noise_vector=np.array(self.noisei),
                        priors=self.gp_priors,
                        n_desired_samples=gp_samples,
                        n_burnin=gp_burnin,
                        progress=progress,
                    )

            X = self.space.transform(
                self.space.rvs(n_samples=self.n_points, random_state=self.rng)
            )
            acq_values = evaluate_acquisitions(
                X=X,
                gpr=self.gp,
                acquisition_functions=(self.acq_func,),
                n_samples=n_samples,
                progress=False,
                random_state=self.rng.randint(0, np.iinfo(np.int32).max),
                **self.acq_func_kwargs,
            ).flatten()

            self._next_x = self.space.inverse_transform(
                X[np.argmax(acq_values)].reshape((1, -1))
            )[0]

        return create_result(self.Xi, self.yi, self.space, self.rng, models=[self.gp])

    def run(self, func, n_iter=1, n_samples=5, gp_burnin=10):
        for _ in range(n_iter):
            x = self.ask()
            self.tell(x, func(x), n_samples=n_samples, gp_burnin=gp_burnin)

        return create_result(self.Xi, self.yi, self.space, self.rng, models=[self.gp])
