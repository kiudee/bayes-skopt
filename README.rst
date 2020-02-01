===========
Bayes-skopt
===========


.. image:: https://img.shields.io/pypi/v/bask.svg
        :target: https://pypi.python.org/pypi/bask

.. image:: https://img.shields.io/travis/kiudee/bayes-skopt.svg
        :target: https://travis-ci.org/kiudee/bayes-skopt

.. image:: https://readthedocs.org/projects/bayes-skopt/badge/?version=latest
        :target: https://bayes-skopt.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/kiudee/bayes-skopt/shield.svg
     :target: https://pyup.io/repos/github/kiudee/bayes-skopt/
     :alt: Updates



A fully Bayesian implementation of sequential model-based optimization


* Free software: Apache Software License 2.0
* Documentation: https://bayes-skopt.readthedocs.io.


Features
--------

- A **fully Bayesian** variant of the ``GaussianProcessRegressor``.
- State of the art information-theoretic acquisition functions, such as the
  `Max-value entropy search <https://arxiv.org/abs/1703.01968>`__, for even faster
  convergence in simple regret.
- Familiar `Optimizer` interface known from Scikit-Optimize.

Installation
------------

The latest development version of Bayes-skopt can be installed from Github as follows::

   pip install git+https://github.com/kiudee/bayes-skopt

Another option is to clone the repository and install Bayes-skopt using::

   python setup.py install

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
