import emcee as mc
import numpy as np
from collections.abc import Iterable
from contextlib import contextmanager, nullcontext
from scipy.linalg import cholesky, cho_solve, solve_triangular
from sklearn.utils import check_random_state
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import WhiteKernel
from skopt.learning.gaussian_process.gpr import _param_for_white_kernel_in_Sum

from .utils import geometric_median


__all__ = ["BayesGPR"]


class BayesGPR(GaussianProcessRegressor):
    """ Gaussian process regressor of which the kernel hyperparameters are inferred in a fully Bayesian framework.

    The implementation is based on Algorithm 2.1 of Gaussian Processes for Machine Learning (GPML) by Rasmussen and
    Williams.

    In addition to standard scikit-learn estimator API, GaussianProcessRegressor:
       * allows prediction without prior fitting (based on the GP prior);
       * provides an additional method sample_y(X), which evaluates samples drawn from the GPR (prior or posterior or
         hyper-posterior) at given inputs;
       * exposes a method log_marginal_likelihood(theta), which can be used externally for other ways of selecting
         hyperparameters, e.g., via Markov chain Monte Carlo.
       * allows setting the kernel hyperparameters while correctly recalculating the required matrices
       * exposes a method noise_set_to_zero() which can be used as a context manager to temporarily set the prediction
         noise to zero. This is useful for evaluating acquisition functions for Bayesian optimization

    Parameters
    ----------
    kernel : kernel object
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel "1.0 * RBF(1.0)" is used as default. Note that
        the kernel's hyperparameters are set to the geometric median of the
        Markov chain Monte Carlo samples of the posterior.
    alpha : float or array-like, optional (default: 1e-10)
        Value added to the diagonal of the kernel matrix during fitting.
        Larger values correspond to increased noise level in the observations.
        This can also prevent a potential numerical issue during fitting, by
        ensuring that the calculated values form a positive definite matrix.
        If an array is passed, it must have the same number of entries as the
        data used for fitting and is used as datapoint-dependent noise level.
        Note that this is equivalent to adding a WhiteKernel with c=alpha.
        Allowing to specify the noise level directly as a parameter is mainly
        for convenience and for consistency with Ridge.
        Also note, that this class adds a WhiteKernel automatically if noise
        is set.
    optimizer : string or callable, optional (default: "fmin_l_bfgs_b")
        Can either be one of the internally supported optimizers for optimizing
        the kernel's parameters, specified by a string, or an externally
        defined optimizer passed as a callable. If a callable is passed, it
        must have the signature::
            def optimizer(obj_func, initial_theta, bounds):
                # * 'obj_func' is the objective function to be minimized, which
                #   takes the hyperparameters theta as parameter and an
                #   optional flag eval_gradient, which determines if the
                #   gradient is returned additionally to the function value
                # * 'initial_theta': the initial value for theta, which can be
                #   used by local optimizers
                # * 'bounds': the bounds on the values of theta
                ....
                # Returned are the best found hyperparameters theta and
                # the corresponding value of the target function.
                return theta_opt, func_min
        Per default, the 'fmin_l_bfgs_b' algorithm from scipy.optimize
        is used. If None is passed, the kernel's parameters are kept fixed.
        Available internal optimizers are::
            'fmin_l_bfgs_b'
        Note, that the kernel hyperparameters obtained are only used as the
        initial position of the Markov chain and will be discarded afterwards.
    n_restarts_optimizer : int, optional (default: 0)
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values. If greater than 0, all bounds
        must be finite. Note that n_restarts_optimizer == 0 implies that one
        run is performed.
    normalize_y : boolean, optional (default: False)
        Whether the target values y are normalized, i.e., the mean of the
        observed target values become zero. This parameter should be set to
        True if the target values' mean is expected to differ considerable from
        zero. When enabled, the normalization effectively modifies the GP's
        prior based on the data, which contradicts the likelihood principle;
        normalization is thus disabled per default.
    copy_X_train : bool, optional (default: True)
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.
    random_state : int, RandomState instance or None, optional (default: None)
        The generator used to initialize the centers. If int, random_state is
        the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.
    noise :  string, optional (default: "gaussian")
        If set to "gaussian", then it is assumed that `y` is a noisy
        estimate of `f(x)` where the noise is gaussian.
        A WhiteKernel will be added to the provided kernel.

    Attributes
    ----------
    X_train_ : array-like, shape = (n_samples, n_features)
        Feature values in training data (also required for prediction)
    y_train_ : array-like, shape = (n_samples, [n_output_dims])
        Target values in training data (also required for prediction)
    kernel_ : kernel object
        The kernel used for prediction. The structure of the kernel is the
        same as the one passed as parameter but with optimized hyperparameters
    L_ : array-like, shape = (n_samples, n_samples)
        Lower-triangular Cholesky decomposition of the kernel in ``X_train_``
    alpha_ : array-like, shape = (n_samples,)
        Dual coefficients of training data points in kernel space
    log_marginal_likelihood_value_ : float
        The log-marginal-likelihood of ``self.kernel_.theta``
    noise_ : float
        Estimate of the gaussian noise. Useful only when noise is set to
        "gaussian".
    chain_ : array-like, shape = (n_desired_samples, n_hyperparameters)
        Samples from the posterior distribution of the hyperparameters.
    pos_ : array-like, shape = (n_walkers, n_hyperparameters)
        Last position of the Markov chain. Useful for continuing sampling when new datapoints arrive.
        fit(X, y) internally uses an existing pos_ to resume sampling, if no other position is provided.
    """

    def __init__(
        self,
        kernel=None,
        alpha=1e-10,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        normalize_y=False,
        copy_X_train=True,
        random_state=None,
        noise="gaussian",
    ):
        if kernel is None:
            self._kernel = None
        else:
            self._kernel = kernel.clone_with_theta(kernel.theta)
        random_state = check_random_state(random_state)
        super().__init__(
            kernel,
            alpha,
            optimizer,
            n_restarts_optimizer,
            normalize_y,
            copy_X_train,
            random_state,
            noise,
        )
        self._alpha = self.alpha
        self._sampler = None
        self.chain_ = None
        self.pos_ = None
        self.kernel_ = None

    @property
    def theta(self):
        if self.kernel_ is not None:
            with np.errstate(divide="ignore"):
                return np.copy(self.kernel_.theta)
        return None

    @theta.setter
    def theta(self, theta):
        self.kernel_.theta = theta
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = cholesky(K, lower=True)
            L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
            self.K_inv_ = L_inv.dot(L_inv.T)
        except np.linalg.LinAlgError as exc:
            exc.args = (
                "The kernel, %s, is not returning a "
                "positive definite matrix. Try gradually "
                "increasing the 'alpha' parameter of your "
                "GaussianProcessRegressor estimator." % self.kernel_,
            ) + exc.args
            raise
        self.alpha_ = cho_solve((self.L_, True), self.y_train_)

    @contextmanager
    def noise_set_to_zero(self):
        current_theta = self.theta
        try:
            # Now we set the noise to 0, but do NOT recalculate the alphas!:
            white_present, white_param = _param_for_white_kernel_in_Sum(self.kernel_)
            self.kernel_.set_params(**{white_param: WhiteKernel(noise_level=0.0)})
            yield self
        finally:
            self.kernel_.theta = current_theta

    def _apply_noise_vector(self, n_instances, noise_vector):
        # We apply the noise vector to self.alpha here, to avoid having to pull up inherited code:
        if noise_vector is not None:
            if not np.iterable(self.alpha):
                alpha = np.ones(n_instances) * self.alpha
            elif not np.iterable(self._alpha):  # we already changed self.alpha before
                alpha = np.ones(n_instances) * self._alpha
            alpha[: len(noise_vector)] += noise_vector
            self.alpha = alpha

    def sample(
        self,
        X=None,
        y=None,
        noise_vector=None,
        n_threads=1,
        n_desired_samples=100,
        n_burnin=0,
        n_thin=1,
        n_walkers_per_thread=100,
        progress=False,
        priors=None,
        position=None,
        add=False,
        **kwargs
    ):
        """ Sample from the posterior distribution of the hyper-parameters."""
        def log_prob_fn(x, gp=self):
            lp = 0
            if isinstance(priors, Iterable):
                for prior, val in zip(priors, x):
                    lp += prior(val)
            else:  # Assume priors is a callable, which evaluates the log probability:
                lp += priors(x)
            try:
                lp = lp + gp.log_marginal_likelihood(theta=x)
            except ValueError:
                return -np.inf
            if not np.isfinite(lp):
                return -np.inf
            return lp

        if X is None and not hasattr(self, "X_train_"):
            raise ValueError(
                """
                It looks like you are trying to sample from the GP posterior without data. Pass X and y, or ensure that
                you call fit before sample.
                """
            )

        # Update data, if available:
        if X is not None:
            if self.normalize_y:
                self._y_train_mean = np.mean(y, axis=0)
                y = y - self._y_train_mean
            else:
                self._y_train_mean = np.zeros(1)
            self.X_train_ = np.copy(X) if self.copy_X_train else X
            self.y_train_ = np.copy(y) if self.copy_X_train else y

        self._apply_noise_vector(len(self.y_train_), noise_vector)

        n_dim = len(self.theta)
        n_walkers = n_threads * n_walkers_per_thread
        n_samples = np.ceil(n_desired_samples / n_walkers) + n_burnin
        pos = None
        if position is not None:
            pos = position
        elif self.pos_ is not None:
            pos = self.pos_
        # elif backup_file is not None:
        #     try:
        #         with open(backup_file, 'rb') as f:
        #             pos = np.load(f)
        #     except FileNotFoundError:
        #         pass
        if pos is None:
            theta = self.theta
            theta[np.isinf(theta)] = np.log(self.noise_)
            pos = [
                theta + 1e-2 * self.random_state.randn(n_dim) for _ in range(n_walkers)
            ]
        self._sampler = mc.EnsembleSampler(
            nwalkers=n_walkers,
            ndim=n_dim,
            log_prob_fn=log_prob_fn,
            threads=n_threads,
            **kwargs
        )
        rng = np.random.RandomState(
            self.random_state.randint(0, np.iinfo(np.int32).max)
        )
        self._sampler.random_state = rng.get_state()
        pos, prob, state = self._sampler.run_mcmc(pos, n_samples, progress=progress)
        # if backup_file is not None:
        #     with open(backup_file, "wb") as f:
        #         np.save(f, pos)
        chain = self._sampler.get_chain(flat=True, discard=n_burnin, thin=n_thin)
        if add and self.chain_ is not None:
            self.chain_ = np.concatenate([self.chain_, chain])
        else:
            self.chain_ = chain
        self.theta = geometric_median(self.chain_)
        self.pos_ = pos

    def fit(
        self,
        X,
        y,
        noise_vector=None,
        n_threads=1,
        n_desired_samples=100,
        n_burnin=10,
        n_walkers_per_thread=100,
        progress=True,
        priors=None,
        position=None,
        **kwargs
    ):
        self.kernel = self._kernel
        self._apply_noise_vector(len(y), noise_vector)
        super().fit(X, y)
        self.sample(
            n_threads=n_threads,
            n_desired_samples=n_desired_samples,
            n_burnin=n_burnin,
            n_walkers_per_thread=n_walkers_per_thread,
            progress=progress,
            priors=priors,
            position=position,
            add=False,
            **kwargs
        )

    def sample_y(self, X, sample_mean=False, noise=False, n_samples=1, random_state=0):
        rng = check_random_state(random_state)
        if sample_mean:
            if noise:
                cm = nullcontext(self)
            else:
                cm = self.noise_set_to_zero()
            with cm:
                samples = super().sample_y(X, n_samples=n_samples, random_state=rng)
            return samples
        ind = rng.choice(len(self.chain_), size=n_samples, replace=True)
        current_theta = self.theta
        current_K_inv = np.copy(self.K_inv_)
        current_L = np.copy(self.L_)
        current_alpha = np.copy(self.alpha_)
        result = np.empty((X.shape[0], n_samples))
        for i, j in enumerate(ind):
            theta = self.chain_[j]
            self.theta = theta
            if noise:
                cm = nullcontext(self)
            else:
                cm = self.noise_set_to_zero()
            with cm:
                result[:, i] = (
                    super().sample_y(X, n_samples=1, random_state=rng).flatten()
                )
        self.kernel_.theta = current_theta
        self.K_inv_ = current_K_inv
        self.alpha_ = current_alpha
        self.L_ = current_L
        return result
