import warnings
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.utils import check_random_state
from skopt.utils import (
    create_result,
    normalize_dimensions,
    is_listlike,
    is_2Dlistlike,
    expected_minimum,
)

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
        **kwargs,
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
        elif len(gp_priors) != self.space.transformed_n_dims + 2:
            raise ValueError("The number of priors does not match the number of dimensions + 2.")
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

    def probability_of_optimality(
        self,
        threshold,
        n_space_samples=500,
        n_gp_samples=200,
        n_random_starts=100,
        use_mean_gp=True,
        random_state=None,
    ):
        """ Compute the probability that the current expected optimum cannot be improved by more than ``threshold``
        points.

        Parameters
        ----------
        threshold : float or list-of-floats
            Other points have to be better than the current optimum by at least a margin of size ``threshold``.
            If a list is passed, this will return a list of probabilities.
        n_space_samples : int, default=500
            Number of random samples used to cover the optimization space.
        n_gp_samples : int, default=200
            Number of functions to sample from the Gaussian process.
        n_random_starts : int, default=100
            Number of random positions to start the optimizer from in order to determine the global optimum.
        use_mean_gp : bool, default=True
            If True, random functions will be sampled from the consensus GP, which is usually faster, but could
            underestimate the variability. If False, the posterior distribution over hyperparameters is used to sample
            different GPs and then sample functions.
        random_state : int, RandomState instance, or None (default)
            Set random state to something other than None for reproducible results.

        Returns
        -------
        probabilities : float or list-of-floats
            Probabilities of the current optimum to be optimal wrt the given thresholds.
        """
        result = create_result(self.Xi, self.yi, self.space, self.rng, models=[self.gp])
        X_orig = [
            expected_minimum(
                result, random_state=random_state, n_random_starts=n_random_starts
            )[0]
        ]

        X_orig.extend(
            self.space.rvs(n_samples=n_space_samples, random_state=random_state)
        )
        X_trans = self.space.transform(X_orig)
        score_samples = self.gp.sample_y(
            X_trans,
            n_samples=n_gp_samples,
            sample_mean=use_mean_gp,
            random_state=random_state,
        )

        if not is_listlike(threshold):
            threshold = [threshold]
        probabilities = []
        for eps in threshold:
            probabilities.append(
                (
                    (score_samples[0][None, :] - eps - score_samples).max(axis=0) < 0.0
                ).mean()
            )
        if len(probabilities) == 1:
            return probabilities[0]
        return probabilities

    def expected_optimality_gap(
        self,
        max_tries=3,
        n_probabilities=50,
        n_space_samples=500,
        n_gp_samples=200,
        n_random_starts=100,
        tol=0.01,
        use_mean_gp=True,
        random_state=None,
    ):
        """ Estimate the expected optimality gap by repeatedly sampling functions consistent with the data.

        Parameters
        ----------
        max_tries : int, default=3
            Maximum amount of tries to compute the current global optimum.
            Raises a ValueError, if it fails.
        n_probabilities : int, default=50
            Number of probabilities to calculate in order to estimate the cumulative distribution function for the
            optimality gap.
        n_space_samples : int, default=500
            Number of random samples used to cover the optimization space.
        n_gp_samples : int, default=200
            Number of functions to sample from the Gaussian process.
        n_random_starts : int, default=100
            Number of random positions to start the optimizer from in order to determine the global optimum.
        tol : float, default=0.01
            Tolerance with which to determine the upper bound for the optimality gap.
        use_mean_gp : bool, default=True
            If True, random functions will be sampled from the consensus GP, which is usually faster, but could
            underestimate the variability. If False, the posterior distribution over hyperparameters is used to sample
            different GPs and then sample functions.
        random_state : int, RandomState instance, or None (default)
            Set random state to something other than None for reproducible results.

        Returns
        -------
        expected_gap : float
            The expected optimality gap of the current global optimum with respect to randomly sampled,
             consistent optima.
        """
        random_state = check_random_state(random_state)
        seed = random_state.randint(0, 2 ** 32 - 1, dtype=np.int64)

        def func(threshold):
            prob = self.probability_of_optimality(
                threshold=threshold,
                n_random_starts=n_random_starts,
                n_gp_samples=n_gp_samples,
                n_space_samples=n_space_samples,
                use_mean_gp=use_mean_gp,
                random_state=seed,
            )
            return (prob - 1.0) ** 2 + threshold ** 2 * 1e-3

        max_observed_gap = np.max(self.yi) - np.min(self.yi)
        for _ in range(max_tries):
            try:
                upper_threshold = minimize_scalar(
                    func, bounds=(0.0, max_observed_gap), tol=tol
                ).x
                break
            except ValueError:
                pass
        else:
            raise ValueError("Determining the upper threshold was not possible.")

        thresholds = list(np.linspace(0, upper_threshold, num=n_probabilities))
        probabilities = self.probability_of_optimality(
            thresholds,
            n_random_starts=n_random_starts,
            n_gp_samples=n_gp_samples,
            n_space_samples=n_space_samples,
            use_mean_gp=use_mean_gp,
            random_state=seed,
        )
        expected_gap = 0.0
        for i in range(0, len(probabilities) - 1):
            p = probabilities[i + 1] - probabilities[i]
            expected_gap += p * thresholds[i + 1]
        return expected_gap
