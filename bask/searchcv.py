import numpy as np
from scipy.stats import rankdata
from skopt import BayesSearchCV as BayesSearchCVSK
from skopt.utils import point_asdict, dimensions_aslist
from bask.optimizer import Optimizer


class BayesSearchCV(BayesSearchCVSK):
    def __init__(
        self,
        estimator,
        search_spaces,
        optimizer_kwargs=None,
        n_iter=50,
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
        kwargs['dimensions'] = dimensions_aslist(params_space)
        # Here we replace skopt's Optimizer:
        optimizer = Optimizer(**kwargs)
        for i in range(len(optimizer.space.dimensions)):
            if optimizer.space.dimensions[i].name is not None:
                continue
            optimizer.space.dimensions[i].name = list(sorted(
                params_space.keys()))[i]

        return optimizer

    def _step(self, X, y, search_space, optimizer, groups=None, n_points=1):
        """Generate n_jobs parameters and evaluate them in parallel."""

        # get parameter values to evaluate
        # TODO: Until n_points is supported, we will wrap the return value in a list
        params = [optimizer.ask(n_points=n_points)]

        # convert parameters to python native types
        params = [[np.array(v).item() for v in p] for p in params]

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
        local_results = self.cv_results_["mean_test_score"][-len(params):]

        # optimizer minimizes objective, hence provide negative score
        return optimizer.tell(
            params,
            [-score for score in local_results],
            n_samples=self.n_samples,
            gp_samples=self.gp_samples,
            gp_burnin=self.gp_burnin,
            progress=False,
        )
