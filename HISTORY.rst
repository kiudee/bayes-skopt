=======
History
=======

0.2.0 (2020-02-16)
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
