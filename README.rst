|Build Status| |Coverage|

Bayes-skopt
===========
Bayes-skopt (short ``bask``) is a library designed to optimize very costly and noisy black-box functions.
We extend Scikit-Optimize by doing the inference of the hyperparameters in a fully Bayesian framework.
Other than the typical maximum marginal likelihood estimation, suitable prior distribution can be employed to
prevent identifiability/convergence issues.

The library is built on top of Scikit-Optimize, Scikit-Learn, NumPy, SciPy and emcee.

Features
========

- A **fully Bayesian** variant of the ``GaussianProcessRegressor``.
- State of the art information-theoretic acquisition functions, such as the
  `Max-value entropy search <https://arxiv.org/abs/1703.01968>`__, for even faster
  convergence in simple regret.
- Familiar `Optimizer` interface known from Scikit-Optimize.


Installation
============

The latest development version of Bayes-skopt can be installed from Github as follows::

   pip install git+https://github.com/kiudee/bayes-skopt

Another option is to clone the repository and install Bayes-skopt using::

   python setup.py install

License
--------
`Apache License, Version 2.0 <https://github.com/kiudee/cs-ranking/blob/master/LICENSE>`_

.. |Build Status| image:: https://travis-ci.org/kiudee/bayes-skopt.svg?branch=master
   :target: https://travis-ci.org/kiudee/bayes-skopt
.. |Coverage| image:: https://coveralls.io/repos/github/kiudee/bayes-skopt/badge.svg?branch=master
   :target: https://coveralls.io/github/kiudee/bayes-skopt?branch=master