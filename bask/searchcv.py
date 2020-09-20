try:
    from collections.abc import Iterable, Sized
except ImportError:
    from collections import Sized, Iterable

import logging

import numpy as np
from scipy.stats import rankdata
from sklearn.utils.validation import check_is_fitted
from skopt import BayesSearchCV as BayesSearchCVSK
from skopt.utils import create_result, dimensions_aslist, expected_minimum, point_asdict

from bask.optimizer import Optimizer


class BayesSearchCV(BayesSearchCVSK):
    """Fully Bayesian optimization over hyper parameters.

    Wraps skopt.BayesSearchCV with a fully Bayesian estimation of the
    kernel hyperparameters, making it robust to very noisy target functions.

    BayesSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimator used.
    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.
    In contrast to GridSearchCV, not all parameter values are tried out, but
    rather a fixed number of parameter settings is sampled from the specified
    distributions. The number of parameter settings that are tried is
    given by n_iter.
    Parameters are presented as a list of skopt.space.Dimension objects.
    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each search point.
        This object is assumed to implement the scikit-learn estimator api.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.
    search_spaces : dict, list of dict or list of tuple containing
        (dict, int).
        One of these cases:
        1. dictionary, where keys are parameter names (strings)
        and values are skopt.space.Dimension instances (Real, Integer
        or Categorical) or any other valid value that defines skopt
        dimension (see skopt.Optimizer docs). Represents search space
        over parameters of the provided estimator.
        2. list of dictionaries: a list of dictionaries, where every
        dictionary fits the description given in case 1 above.
        If a list of dictionary objects is given, then the search is
        performed sequentially for every parameter space with maximum
        number of evaluations set to self.n_iter.
        3. list of (dict, int > 0): an extension of case 2 above,
        where first element of every tuple is a dictionary representing
        some search subspace, similarly as in case 2, and second element
        is a number of iterations that will be spent optimizing over
        this subspace.
    n_iter : int, default=50
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution. Consider increasing
        ``n_points`` if you want to try more parameter settings in
        parallel.
    return_policy : string, default='best_setting'
        A string specifying which point should be considered the optimum
        at the end of the optimization. Should be one of

            - 'best_mean': return the point maximizing the mean function
              of the Gaussian process. This is usually the best choice
              when the target function is noisy and a single
              observation might not be representative.
              Note, if the number of iterations ``n_iter`` is low, the
              expected optimum can be still be uncertain.
              Only use this setting when you only have one search space.
            - 'best_setting': return the best setting tried so far.
              This is useful, if the target function is (almost)
              noise-free.
    optimizer_kwargs : dict, optional
        Dict of arguments passed to :class:`Optimizer`.
    scoring : string, callable or None, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If ``None``, the ``score`` method of the estimator is used.
    fit_params : dict, optional
        Parameters to pass to the fit method.
    n_jobs : int, default=1
        Number of jobs to run in parallel. At maximum there are
        ``n_points`` times ``cv`` jobs available during each iteration.
    n_points : int, default=1
        This is not implemented yet. Consider using the original
        skopt.BayesSearchCV for now.
        Number of parameter settings to sample in parallel. If this does
        not align with ``n_iter``, the last iteration will sample less
        points. See also :func:`~Optimizer.ask`
    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
    iid : boolean, default=True
        If True, the data is assumed to be identically distributed across
        the folds, and the loss minimized is the total loss per sample,
        and not the mean loss across the folds.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.
        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.
    refit : boolean, default=True
        Refit the best estimator with the entire dataset.
        If "False", it is impossible to make predictions using
        this RandomizedSearchCV instance after fitting.
    verbose : integer
        Controls the verbosity: the higher, the more messages.
    random_state : int or RandomState
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.
    return_train_score : boolean, default=False
        If ``'True'``, the ``cv_results_`` attribute will include training
        scores.
    Examples
    --------
    >>> from bask import BayesSearchCV
    >>> # parameter ranges are specified by one of below
    >>> from skopt.space import Real, Categorical, Integer
    >>>
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> X, y = load_iris(True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ...                                                     train_size=0.75,
    ...                                                     random_state=0)
    >>>
    >>> # log-uniform: understand as search over p = exp(x) by varying x
    >>> opt = BayesSearchCV(
    ...     SVC(),
    ...     {
    ...         'C': Real(1e-6, 1e+6, prior='log-uniform'),
    ...         'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
    ...         'degree': Integer(1,8),
    ...         'kernel': Categorical(['linear', 'poly', 'rbf']),
    ...     },
    ...     n_iter=32,
    ...     random_state=0
    ... )
    >>>
    >>> # executes bayesian optimization
    >>> _ = opt.fit(X_train, y_train)
    >>>
    >>> # model can be saved, used for predictions or scoring
    >>> print(opt.score(X_test, y_test))
    0.973...
    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.
        For instance the below given table
        +--------------+-------------+-------------------+---+---------------+
        | param_kernel | param_gamma | split0_test_score |...|rank_test_score|
        +==============+=============+===================+===+===============+
        |    'rbf'     |     0.1     |        0.8        |...|       2       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.2     |        0.9        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.3     |        0.7        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        will be represented by a ``cv_results_`` dict of::
            {
            'param_kernel' : masked_array(data = ['rbf', 'rbf', 'rbf'],
                                          mask = False),
            'param_gamma'  : masked_array(data = [0.1 0.2 0.3], mask = False),
            'split0_test_score'  : [0.8, 0.9, 0.7],
            'split1_test_score'  : [0.82, 0.5, 0.7],
            'mean_test_score'    : [0.81, 0.7, 0.7],
            'std_test_score'     : [0.02, 0.2, 0.],
            'rank_test_score'    : [3, 1, 1],
            'split0_train_score' : [0.8, 0.9, 0.7],
            'split1_train_score' : [0.82, 0.5, 0.7],
            'mean_train_score'   : [0.81, 0.7, 0.7],
            'std_train_score'    : [0.03, 0.03, 0.04],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.007, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.001, 0.002, 0.003, 0.005],
            'params' : [{'kernel' : 'rbf', 'gamma' : 0.1}, ...],
            }
        NOTE that the key ``'params'`` is used to store a list of parameter
        settings dict for all the parameter candidates.
        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.
    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if refit=False.
    optimizer_results_ : list of `OptimizeResult`
        Contains a `OptimizeResult` for each search space. The search space
        parameter are sorted by its name.
    best_score_ : float
        Score of best_estimator on the left out data.
    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.
    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.
        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).
    scorer_ : function
        Scorer function used on the held out data to choose the best
        parameters for the model.
    n_splits_ : int
        The number of cross-validation splits (folds/iterations).
    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.
    If `n_jobs` was set to a value higher than one, the data is copied for each
    parameter setting(and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.
    See Also
    --------
    :class:`skopt.BayesSearchCV`:
        This class wraps the original BayesSearchCV in skopt.
    :class:`GridSearchCV`:
        Does exhaustive search over a grid of parameters.
    """

    def __init__(
        self,
        estimator,
        search_spaces,
        optimizer_kwargs=None,
        n_iter=50,
        return_policy="best_setting",
        scoring=None,
        fit_params=None,
        n_jobs=1,
        n_points=1,
        iid=True,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        random_state=None,
        error_score="raise",
        return_train_score=False,
    ):
        super().__init__(
            estimator,
            search_spaces,
            optimizer_kwargs,
            n_iter,
            scoring,
            fit_params,
            n_jobs,
            n_points,
            iid,
            refit,
            cv,
            verbose,
            pre_dispatch,
            random_state,
            error_score,
            return_train_score,
        )
        self.return_policy = return_policy
        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {}
        self.n_samples = self.optimizer_kwargs.get("n_samples", 0)
        self.gp_samples = self.optimizer_kwargs.get("gp_samples", 100)
        self.gp_burnin = self.optimizer_kwargs.get("gp_burnin", 5)
        if "acq_func" not in self.optimizer_kwargs:
            self.optimizer_kwargs["acq_func"] = "pvrs"

    def _make_optimizer(self, params_space):
        """Instantiate bask Optimizer class.

        Parameters
        ----------
        params_space : dict
            Represents parameter search space. The keys are parameter
            names (strings) and values are skopt.space.Dimension instances,
            one of Real, Integer or Categorical.
        Returns
        -------
        optimizer: Instance of the `Optimizer` class used for for search
            in some parameter space.

        """
        kwargs = self.optimizer_kwargs_.copy()
        kwargs["dimensions"] = dimensions_aslist(params_space)
        # Here we replace skopt's Optimizer:
        optimizer = Optimizer(**kwargs)
        for i in range(len(optimizer.space.dimensions)):
            if optimizer.space.dimensions[i].name is not None:
                continue
            optimizer.space.dimensions[i].name = list(sorted(params_space.keys()))[i]

        return optimizer

    @property
    def best_params_(self):
        check_is_fitted(self, "cv_results_")
        if self.return_policy == "best_setting" or len(self.optimizers_) > 1:
            if len(self.optimizers_) > 1:
                logging.warning(
                    "Return policy 'best_mean' is incompatible with multiple search"
                    "spaces. Reverting to 'best_setting'."
                )
            return self.cv_results_["params"][self.best_index_]
        if self.return_policy == "best_mean":
            random_state = self.optimizer_kwargs_["random_state"]
            # We construct a result object manually here, since in skopt versions up to
            # 0.7.4 they were not saved yet:
            opt = self.optimizers_[0]
            result_object = create_result(
                opt.Xi, opt.yi, space=opt.space, rng=random_state, models=[opt.gp]
            )
            point, _ = expected_minimum(
                res=result_object, n_random_starts=100, random_state=random_state,
            )
            dict = point_asdict(self.search_spaces, point)
            return dict

    def _step(self, X, y, search_space, optimizer, groups=None, n_points=1):
        """Generate n_jobs parameters and evaluate them in parallel."""

        # get parameter values to evaluate
        # TODO: Until n_points is supported, we will wrap the return value in a list
        params = [optimizer.ask(n_points=n_points)]

        # convert parameters to python native types
        # in case we have any Iterable parameters, we want to
        # stop numpy from coercing them into an np.array
        def try_convert_to_np(item):
            if isinstance(item, Iterable):
                return item
            try:
                return np.array(item).item()
            except ValueError:
                return item
        params = [[try_convert_to_np(v) for v in p] for p in params]

        # make lists into dictionaries
        params_dict = [point_asdict(search_space, p) for p in params]

        # HACK: self.cv_results_ is reset at every call to _fit, keep current
        all_cv_results = self.cv_results_

        # HACK: this adds compatibility with different versions of sklearn
        refit = self.refit
        self.refit = False
        self._fit(X, y, groups, params_dict)
        self.refit = refit

        # merge existing and new cv_results_
        for k in self.cv_results_:
            all_cv_results[k].extend(self.cv_results_[k])

        all_cv_results["rank_test_score"] = list(
            np.asarray(
                rankdata(-np.array(all_cv_results["mean_test_score"]), method="min"),
                dtype=np.int32,
            )
        )
        if self.return_train_score:
            all_cv_results["rank_train_score"] = list(
                np.asarray(
                    rankdata(
                        -np.array(all_cv_results["mean_train_score"]), method="min"
                    ),
                    dtype=np.int32,
                )
            )
        self.cv_results_ = all_cv_results
        self.best_index_ = np.argmax(self.cv_results_["mean_test_score"])

        # feed the point and objective back into optimizer
        local_results = self.cv_results_["mean_test_score"][-len(params) :]

        # optimizer minimizes objective, hence provide negative score
        return optimizer.tell(
            params,
            [-score for score in local_results],
            n_samples=self.n_samples,
            gp_samples=self.gp_samples,
            gp_burnin=self.gp_burnin,
            progress=False,
        )
