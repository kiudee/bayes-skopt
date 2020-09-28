=======
History
=======

0.10.2 (2020-09-28)
-------------------
* Fix divide by zero encountered in log when evaluating acquisition functions
  without noise.

0.10.1 (2020-09-26)
-------------------
* Bump minimum arviz version to 0.10.0.

0.10.0 (2020-09-20)
-------------------
* Add new initialization using the Steinerberger sequence. This works better
  in high-dimensional problems than the R2 sequence.
* Fix exception when a categorical parameter is Iterable.

0.9.3 (2020-09-14)
------------------
* Make default priors for input warping more focused on the identity transform.
  This fixes issues with overfitting in high noise environments.

0.9.2 (2020-09-04)
------------------
* Fix incorrect recomputation of y mean when using ``normalize_y=True``.

0.9.1 (2020-09-02)
------------------
* Fix calculation of max-value entropy search and make it more robust.

0.9.0 (2020-08-31)
------------------
* Add support for automatic input warping. It can be activated by passing
  ``warp_inputs=True`` to ``BayesGPR``.

0.8.0 (2020-08-09)
------------------

* Add ``Optimizer.optimum_intervals`` which computes the highest density
  intervals for the optimal parameters.
* ``BayesGPR`` has ``normalize_y`` now set to ``True`` by default.
* Add option to normalize the optimality gap when computing
  ``Optimizer.expected_optimality_gap`` or
  ``Optimizer.probability_of_optimality`` (activated by default).
* ``Optimizer.run`` now accepts target functions that also return a noise
  estimate.
* ``Optimizer.run`` accepts the same arguments as ``Optimizer.tell``.

0.7.2 (2020-08-01)
------------------
* Fix ``guess_priors`` not correctly adding the prior for the ``WhiteKernel``.
  It is now called directly in ``BayesGPR.sample``.

0.7.1 (2020-07-28)
------------------
* Restrict length scale bounds of the default kernel to a tighter interval.
  This should help start the MCMC walkers in a region with higher likelihood.

0.7.0 (2020-07-26)
------------------
* Replace the default inverse gamma distribution prior for the lengthscales by the round-flat distribution.
* Fix ``guess_priors`` to correctly add kernels with multiple lengthscales.

0.6.0 (2020-05-21)
------------------

* Add ``Optimizer.expected_optimality_gap`` which estimates the expected optimality gap of the current global optimum
  to random optima sampled from the Gaussian process.
* Check that the list of priors has the correct length.
* Require emcee to be at least version 3.0.

0.5.0 (2020-05-21)
------------------

* Add ``Optimizer.probability_of_optimality`` which estimates the probability that the current global optimum is
  optimal within a certain tolerance. This can be used to make stopping rules.

0.4.1 (2020-05-19)
------------------

* Update and fix dependencies.

0.4.0 (2020-04-27)
------------------

* Add ``return_policy`` parameter to ``BayesSearchCV``. Allows the user to choose between returning the best
  observed configuration (in a noise-less setting) or the best predicted configuration (for noisy targets).

0.3.3 (2020-03-16)
------------------

* Fix error occuring when an unknown argument was passed to ``Optimizer``.

0.3.0 (2020-03-12)
------------------

* Add predictive variance reduction search criterion. This is the new default
  acquisition function.
* Implement ``BayesSearchCV`` for use with scikit-learn estimators and
  pipelines. This is an easy to use drop-in replacement for GridSearchCV or
  RandomSearchCV. It is implemented as a wrapper around skopt.BayesSearchCV.
* Determine default kernels and priors to use, if the user provides none.
* Add example notebooks on how to use the library.
* Add API documentation of the library.


0.2.0 (2020-03-01)
------------------

* Allow user to pass a vector of noise variances to ``tell``, ``fit`` and ``sample``.
  This can be used to warm start the optimization process.

0.1.2 (2020-02-16)
------------------

* Fix the ``tell`` method of the optimizer not updating ``_n_initial_points`` correctly,
  when using replace.

0.1.0 (2020-02-01)
------------------

* First release on PyPI.
