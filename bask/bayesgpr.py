from collections.abc import Iterable
from contextlib import contextmanager, nullcontext

import emcee as mc
import numpy as np
import scipy.stats as st
from scipy.linalg import cho_solve, cholesky, solve_triangular
import sklearn
from sklearn.utils import check_random_state
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.gpr import _param_for_white_kernel_in_Sum
from skopt.learning.gaussian_process.kernels import WhiteKernel

from .utils import geometric_median, guess_priors, validate_zeroone

__all__ = ["BayesGPR"]


class BayesGPR(GaussianProcessRegressor):
    """ Gaussian process regressor of which the kernel hyperparameters are inferred in a
    fully Bayesian framework.

    The implementation is based on Algorithm 2.1 of Gaussian Processes for Machine
    Learning (GPML) by Rasmussen and Williams.

    In addition to standard scikit-learn estimator API, GaussianProcessRegressor:
       * allows prediction without prior fitting (based on the GP prior);
       * provides an additional method sample_y(X), which evaluates samples drawn from
         the GPR (prior or posterior or hyper-posterior) at given inputs;
       * exposes a method log_marginal_likelihood(theta), which can be used externally
         for other ways of selecting hyperparameters,
         e.g., via Markov chain Monte Carlo.
       * allows setting the kernel hyperparameters while correctly recalculating the
         required matrices
       * exposes a method noise_set_to_zero() which can be used as a context manager to
         temporarily set the prediction noise to zero.
         This is useful for evaluating acquisition functions for Bayesian optimization

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
        prior based on the data, which contradicts the likelihood principle.
    warp_inputs : boolean, optional (default: False)
        If True, each input dimension will be warped (internally) using the cumulative
        distribution function of a beta distribution [1]_. The parameters of each beta
        distribution will be inferred from the data. The input data needs to be
        in [0, 1].
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
        Last position of the Markov chain. Useful for continuing sampling when new
        datapoints arrive. fit(X, y) internally uses an existing pos_ to resume
        sampling, if no other position is provided.

    References
    ----------
    .. [1] Snoek, Jasper, Kevin Swersky, Richard Zemel, and Ryan P. Adams. “Input
       Warping for Bayesian Optimization of Non-Stationary Functions.”
       In Proceedings of the 31st International Conference on International
       Conference on Machine Learning - Volume 32, II–1674–II–1682.
       ICML’14. Beijing, China: JMLR.org, 2014.
    """

    def __init__(
        self,
        kernel=None,
        alpha=1e-10,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        normalize_y=False,
        warp_inputs=False,
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
        self.warp_inputs = warp_inputs
        self._sampler = None
        self.chain_ = None
        self.pos_ = None
        self.kernel_ = None

    @property
    def theta(self):
        """The current geometric median of the kernel hyperparameter distribution.

        The returned values are located in log space. Call `BayesGPR.kernel_` to obtain
        the values their original space.

        Returns
        -------
        ndarray
            Array containing the kernel hyperparameters in log space.

        """
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

    @property
    def X_train_(self):
        """ The training data which was used to train the Gaussian process.

        If input warping is used, it will return the warped instances.

        Returns
        -------
        array-like, shape = (n_samples, n_features)
            Feature values in training data (also required for prediction).
            If `warp_inputs=True`, will contain the warped inputs in [0, 1].
        """
        if hasattr(self, "_X_train_orig_"):
            if self.warp_inputs:
                return self._X_train_warped_
            return self._X_train_orig_
        return None

    @X_train_.setter
    def X_train_(self, X_train):
        self._X_train_orig_ = np.copy(X_train) if self.copy_X_train else X_train
        if self.warp_inputs:
            self._X_train_warped_ = np.copy(self._X_train_orig_)
            if hasattr(self, "warpers_"):
                for col, warper in enumerate(self.warpers_):
                    self._X_train_warped_[:, col] = warper(self._X_train_orig_[:, col])
            # If no warpers exist yet, we begin with an unwarped input space

    def warp(self, X):
        """Warp the input X using the existing warpers.

        Returns X if `warp_inputs=False` or if no warpers have been fit yet.

        Parameters
        ----------
        X : ndarray, shape (n_points, n_dims)
            Points in the original space which should be warped.
        """
        if self.warp_inputs and hasattr(self, "warpers_"):
            X_warped = np.empty_like(X)
            for col, warper in enumerate(self.warpers_):
                X_warped[:, col] = warper(X[:, col])
            X = X_warped
        return X

    def unwarp(self, X):
        """Unwarp the input X back to the original input space.

        Returns X if `warp_inputs=False` or if no warpers have been fit yet.

        Parameters
        ----------
        X : ndarray, shape (n_points, n_dims)
            Points in the warped space which should be transformed back to the input
            space.
        """
        if self.warp_inputs and hasattr(self, "warpers_"):
            X_orig = np.empty_like(X)
            for col, unwarper in enumerate(self.unwarpers_):
                X_orig[:, col] = unwarper(X[:, col])
            X = X_orig
        return X

    def rewarp(self):
        """Apply warping again to X_train_ after parameters have changed.

        Does nothing if `warp_inputs=False` or if no warpers have been fit yet.
        """
        if self.warp_inputs:
            if hasattr(self, "warpers_") and hasattr(self, "_X_train_orig_"):
                self._X_train_warped_ = np.empty_like(self._X_train_orig_)
                for col, warper in enumerate(self.warpers_):
                    self._X_train_warped_[:, col] = warper(self._X_train_orig_[:, col])

    def create_warpers(self, alphas, betas):
        """Create Beta CDFs and inverse CDFs for input (un)warping.

        Parameters
        ----------
        alphas : ndarray, shape (n_dims)
            Raw alpha parameters of the Beta distributions in log-space.
        betas : ndarray, shape (n_dims)
            Raw beta parameters of the Beta distributions in log-space.
        """
        if self.warp_inputs:
            self.warpers_ = []
            self.unwarpers_ = []
            self.warp_alphas_ = np.copy(alphas)
            self.warp_betas_ = np.copy(betas)
            for a_log, b_log in zip(alphas, betas):
                a, b = np.exp(a_log), np.exp(b_log)
                dist = st.beta(a=a, b=b)
                self.warpers_.append(dist.cdf)
                self.unwarpers_.append(dist.ppf)

    @contextmanager
    def noise_set_to_zero(self):
        """Context manager in which the noise of the Gaussian process is 0.

        This is useful when you want to predict the epistemic uncertainty of the
        Gaussian process without the noise.
        """
        current_theta = self.theta
        try:
            # Now we set the noise to 0, but do NOT recalculate the alphas!:
            white_present, white_param = _param_for_white_kernel_in_Sum(self.kernel_)
            self.kernel_.set_params(**{white_param: WhiteKernel(noise_level=0.0)})
            yield self
        finally:
            self.kernel_.theta = current_theta

    def _apply_noise_vector(self, n_instances, noise_vector):
        # We apply the noise vector to self.alpha here, to avoid having to pull up
        # inherited code:
        if noise_vector is not None:
            if not np.iterable(self.alpha):
                alpha = np.ones(n_instances) * self.alpha
            elif not np.iterable(self._alpha):  # we already changed self.alpha before
                alpha = np.ones(n_instances) * self._alpha
            alpha[: len(noise_vector)] += noise_vector
            self.alpha = alpha

    def _log_prob_fn(self, x, priors, warp_priors):
        lp = 0
        if self.warp_inputs:
            n_dim = self.X_train_.shape[1]
            x_warp = x[-2 * n_dim :]
            x_gp = x[: len(x) - 2 * n_dim]
            alphas, betas = x_warp[:n_dim], x_warp[n_dim:]
            self.create_warpers(alphas, betas)
            self.rewarp()
            for a_log, b_log in zip(alphas, betas):
                if isinstance(warp_priors, Iterable):
                    lp += warp_priors[0](a_log)
                    lp += warp_priors[1](b_log)
                else:
                    lp += warp_priors(a_log, b_log)
        else:
            x_gp = x
        if isinstance(priors, Iterable):
            for prior, val in zip(priors, x_gp):
                lp += prior(val)
        else:  # Assume priors is a callable, which evaluates the log probability:
            lp += priors(x_gp)
        try:
            lp = lp + self.log_marginal_likelihood(theta=x_gp)
        except ValueError:
            return -np.inf
        if not np.isfinite(lp):
            return -np.inf
        return lp

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
        warp_priors=None,
        position=None,
        add=False,
        **kwargs
    ):
        """Sample from the posterior distribution of the hyper-parameters.

        Parameters
        ----------
        X : ndarray, shape (n_points, n_dims), optional (default: None)
            Points at which the function is evaluated. If None, it will use the saved
            datapoints.
        y : ndarray, shape (n_points,), optional (default: None)
            Value(s) of the function at `X`. If None, it will use the saved values.
        noise_vector :
            Variance(s) of the function at `X`. If None, no additional noise is applied.
        n_threads : int, optional (default: 1)
            Number of threads to use during inference.
            This is currently not implemented.
        n_desired_samples : int, optional (default: 100)
            Number of hyperposterior samples to collect during inference. Must be a
            multiple of `n_walkers_per_thread`.
        n_burnin : int, optional (default: 0)
            Number of iterations to discard before collecting hyperposterior samples.
            Needs to be increased only, if the hyperposterior samples have not reached
            their typical set yet. Higher values increase the running time.
        n_thin : int, optional (default: 1)
            Only collect hyperposterior samples every k-th iteration. This can help
            reducing the autocorrelation of the collected samples, but reduces the
            total number of samples.
        n_walkers_per_thread : int, optional (default: 100)
            Number of MCMC ensemble walkers to employ during inference.
        progress : bool, optional (default: False)
            If True, show a progress bar during inference.
        priors : list or callable, optional (default: None)
            Log prior(s) for the kernel hyperparameters. Remember that the kernel
            hyperparameters are transformed into log space. Thus your priors need to
            perform the necessary change-of-variables.
        warp_priors : list or callable, optional (default: None)
            Log prior(s) for the parameters of the Beta distribution used to warp each
            dimension. Only used, if `warp_inputs=True`.
            By default uses a log-normal distribution with mean 0 and standard deviation
            of 0.5 for each parameter of the Beta distribution. This prior favors the
            identity transformation and sufficient data is needed to shift towards a
            stronger warping function.
        position : ndarray, shape (n_walkers, n_kernel_dims), optional (default: None)
            Starting position of the Markov chain. If None, it will use the current
            position. If this is None as well, it will try to initialize in a small
            ball.
        add : bool, optional (default: False)
            If True, all collected hyperposterior samples will be added to the existing
            samples in `BayesGPR.chain_`. Otherwise they will be replaced.
        kwargs : dict
            Additional keyword arguments for emcee.EnsembleSampler

        """

        if X is None and not hasattr(self, "X_train_") or self.kernel_ is None:
            raise ValueError(
                """
                It looks like you are trying to sample from the GP posterior without
                data. Pass X and y, or ensure that you call fit before sample.
                """
            )
        # We are only able to guess priors now, since BayesGPR can add
        # another WhiteKernel, when noise is set to "gaussian":
        if priors is None:
            priors = guess_priors(self.kernel_)

        if warp_priors is None:
            warp_priors = (
                st.norm(loc=0.0, scale=0.3).logpdf,
                st.norm(loc=0.0, scale=0.3).logpdf,
            )

        # Update data, if available:
        if X is not None:
            if self.normalize_y:
                self._y_train_mean = np.mean(y, axis=0)
                if int(sklearn.__version__[2:4]) >= 23:
                    self._y_train_std = np.std(y, axis=0)
            else:
                self._y_train_mean = np.zeros(1)
                if int(sklearn.__version__[2:4]) >= 23:
                    self._y_train_std = 1
            if int(sklearn.__version__[2:4]) >= 23:
                self.y_train_std_ = self._y_train_std
                self.y_train_mean_ = self._y_train_mean
            else:
                self.y_train_mean_ = self._y_train_mean
                self.y_train_std_ = 1
            y = (y - self.y_train_mean_) / self.y_train_std_

            if noise_vector is not None:
                noise_vector = np.array(noise_vector) / np.power(self.y_train_std_, 2)

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
        if self.warp_inputs:
            added_dims = self.X_train_.shape[1] * 2
            n_dim += added_dims
        if pos is None:
            theta = self.theta
            theta[np.isinf(theta)] = np.log(self.noise_)
            if self.warp_inputs:
                theta = np.concatenate([theta, np.zeros(added_dims)])
            pos = [
                theta + 1e-2 * self.random_state.randn(n_dim) for _ in range(n_walkers)
            ]
        self._sampler = mc.EnsembleSampler(
            nwalkers=n_walkers,
            ndim=n_dim,
            log_prob_fn=self._log_prob_fn,
            kwargs=dict(priors=priors, warp_priors=warp_priors),
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
        if self.warp_inputs:
            median = geometric_median(self.chain_)
            warp_params = median[len(self.theta) :]
            alphas = warp_params[: self.X_train_.shape[1]]
            betas = warp_params[self.X_train_.shape[1] :]
            self.create_warpers(alphas, betas)
            self.rewarp()
            self.theta = median[: len(self.theta)]
        else:
            self.theta = geometric_median(self.chain_)
        self.log_marginal_likelihood_value_ = self.log_marginal_likelihood(
            self.kernel_.theta, clone_kernel=False
        )
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
        warp_priors=None,
        position=None,
        **kwargs
    ):
        """Fit the Gaussian process model to the given training data.

        Parameters
        ----------
        X : ndarray, shape (n_points, n_dims)
            Points at which the function is evaluated. If None, it will use the saved
            datapoints.
        y : ndarray, shape (n_points,)
            Value(s) of the function at `X`. If None, it will use the saved values.
        noise_vector :
            Variance(s) of the function at `X`. If None, no additional noise is applied.
        n_threads : int, optional (default: 1)
            Number of threads to use during inference.
            This is currently not implemented.
        n_desired_samples : int, optional (default: 100)
            Number of hyperposterior samples to collect during inference. Must be a
            multiple of `n_walkers_per_thread`.
        n_burnin : int, optional (default: 0)
            Number of iterations to discard before collecting hyperposterior samples.
            Needs to be increased only, if the hyperposterior samples have not reached
            their typical set yet. Higher values increase the running time.
        n_walkers_per_thread : int, optional (default: 100)
            Number of MCMC ensemble walkers to employ during inference.
        progress : bool, optional (default: False)
            If True, show a progress bar during inference.
        priors : list or callable, optional (default: None)
            Log prior(s) for the kernel hyperparameters. Remember that the kernel
            hyperparameters are transformed into log space. Thus your priors need to
            perform the necessary change-of-variables.
        position : ndarray, shape (n_walkers, n_kernel_dims), optional (default: None)
            Starting position of the Markov chain. If None, it will use the current
            position. If this is None as well, it will try to initialize in a small
            ball.
        kwargs : dict
            Additional keyword arguments for BayesGPR.sample

        """
        self.kernel = self._kernel
        # In sklearn >= 23 the normalization includes scaling the output by the
        # standard deviation. We need to scale the noise_vector accordingly here:
        if (
            int(sklearn.__version__[2:4]) >= 23
            and self.normalize_y
            and noise_vector is not None
        ):
            y_std = np.std(y, axis=0)
            noise_vector = np.array(noise_vector) / np.power(y_std, 2)
        self._apply_noise_vector(len(y), noise_vector)
        super().fit(X, y)

        self.sample(
            n_threads=n_threads,
            n_desired_samples=n_desired_samples,
            n_burnin=n_burnin,
            n_walkers_per_thread=n_walkers_per_thread,
            progress=progress,
            priors=priors,
            warp_priors=warp_priors,
            position=position,
            add=False,
            **kwargs
        )

    def predict(
        self,
        X,
        return_std=False,
        return_cov=False,
        return_mean_grad=False,
        return_std_grad=False,
    ):
        if self.warp_inputs:
            validate_zeroone(X)
            X = self.warp(X)
        return super().predict(
            X, return_std, return_cov, return_mean_grad, return_std_grad
        )

    def sample_y(self, X, sample_mean=False, noise=False, n_samples=1, random_state=0):
        """Sample function realizations of the Gaussian process.

        Parameters
        ----------
        X : ndarray, shape (n_points, n_dims)
            Points at which to evaluate the functions.
        sample_mean : bool, optional (default: False)
            If True, the geometric median of the hyperposterior samples is used as the
            Gaussian process to sample from. If False, a new set of hyperposterior
            is used for each new sample.
        noise : bool, optional (default: False)
            If True, Gaussian noise is added to the samples.
        n_samples : int, optional (default: 1)
            Number of samples to draw from the Gaussian process(es).
        random_state : int or RandomState or None, optional, default=None
            Pseudo random number generator state used for random uniform sampling
            from lists of possible values instead of scipy.stats distributions.

        Returns
        -------
        result : ndarray, shape (n_points, n_samples)
            Samples from the Gaussian process(es)

        Raises
        ------
        ValueError
            If `warp_inputs=True` and the entries of X are not all between 0 and 1.
        """
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
        if self.warp_inputs:
            current_warp_alphas = np.copy(self.warp_alphas_)
            current_warp_betas = np.copy(self.warp_betas_)
        current_theta = self.theta
        n_dims = len(current_theta)
        current_K_inv = np.copy(self.K_inv_)
        current_L = np.copy(self.L_)
        current_alpha = np.copy(self.alpha_)
        result = np.empty((X.shape[0], n_samples))
        for i, j in enumerate(ind):
            if self.warp_inputs:
                validate_zeroone(X)
                theta = self.chain_[j][:n_dims]
                warp_params = self.chain_[j][n_dims:]
                alphas, betas = warp_params[: X.shape[1]], warp_params[X.shape[1] :]
                self.create_warpers(alphas, betas)
                self.rewarp()
            else:
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
        if self.warp_inputs:
            self.warp_alphas_ = current_warp_alphas
            self.warp_betas_ = current_warp_betas
        self.L_ = current_L
        return result
