


.. image:: https://github.com/kiudee/bayes-skopt/raw/master/docs/images/header.png
   :width: 800 px
   :alt: Bayes-skopt header
   :align: center

===========
Bayes-skopt
===========

.. image:: https://mybinder.org/badge_logo.svg
        :target: https://mybinder.org/v2/gh/kiudee/bayes-skopt/master?filepath=examples

.. image:: https://img.shields.io/pypi/v/bask.svg
        :target: https://pypi.python.org/pypi/bask

.. image:: https://img.shields.io/travis/kiudee/bayes-skopt.svg
        :target: https://travis-ci.org/kiudee/bayes-skopt

.. image:: https://readthedocs.org/projects/bayes-skopt/badge/?version=latest
        :target: https://bayes-skopt.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

A fully Bayesian implementation of sequential model-based optimization


* Free software: Apache Software License 2.0
* Documentation: https://bayes-skopt.readthedocs.io.
* Built on top of the excellent `Scikit-Optimize (skopt) <https://github.com/scikit-optimize/scikit-optimize>`__.


Features
--------

- A **fully Bayesian** variant of the ``GaussianProcessRegressor``.
- State of the art information-theoretic acquisition functions, such as the
  `Max-value entropy search <https://arxiv.org/abs/1703.01968>`__ or
  `Predictive variance reduction search <https://bayesopt.github.io/papers/2017/13.pdf>`__, for even faster
  convergence in simple regret.
- Familiar `Optimizer` interface known from Scikit-Optimize.

Installation
------------

To install the latest stable release it is best to install the version on PyPI::

   pip install bask

The latest development version of Bayes-skopt can be installed from Github as follows::

   pip install git+https://github.com/kiudee/bayes-skopt

Another option is to clone the repository and install Bayes-skopt using::

   poetry install

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
