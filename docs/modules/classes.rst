.. _api_ref:

=============
API Reference
=============

Bayes-skopt, or bask, builds on Scikit-Optimize and implements a fully
Bayesian sequential optimization framework of very noise black-box functions.



:mod:`bask`: module
====================

Base classes
------------
.. currentmodule:: bask

.. autosummary::
   :toctree: generated/
   :template: class.rst

    BayesGPR
    BayesSearchCV
    Optimizer

Functions
---------
.. currentmodule:: bask

.. autosummary::
   :toctree: generated/
   :template: function.rst

    geometric_median
    guess_priors
    construct_default_kernel
    r2_sequence

.. _acquisition_ref:

:mod:`bask.acquisition`: Acquisition
=====================================

.. automodule:: skopt.acquisition
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`acquisition` section for further details.

.. currentmodule:: bask

.. autosummary::
   :toctree: generated/
   :template: class.rst

    acquisition.PVRS
    acquisition.MaxValueSearch
    acquisition.ExpectedImprovement
    acquisition.TopTwoEI
    acquisition.LCB
    acquisition.Expectation
    acquisition.ThompsonSampling
    acquisition.VarianceReduction

.. currentmodule:: bask

.. autosummary::
   :toctree: generated/
   :template: function.rst

    acquisition.evaluate_acquisitions


.. _optimizer_ref:

:mod:`bask.optimizer`: Optimizer
=================================

.. automodule:: bask.optimizer
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`optimizer` section for further details.

.. currentmodule:: bask

.. autosummary::
   :toctree: generated/
   :template: class.rst

    optimizer.Optimizer

.. _utils_ref:

:mod:`bask.utils`: Utils functions.
====================================

.. automodule:: bask.utils
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`utils` section for further details.


.. currentmodule:: bask

.. autosummary::
   :toctree: generated/
   :template: function.rst

    utils.geometric_median
    utils.r2_sequence
    utils.guess_priors
    utils.construct_default_kernel
