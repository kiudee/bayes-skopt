import warnings
import numpy as np
from sklearn.utils import check_random_state
from skopt.utils import create_result, normalize_dimensions, is_listlike, is_2Dlistlike

from . import acquisition
from .bayesgpr import BayesGPR
from .utils import r2_sequence, guess_priors
from bask.acquisition import evaluate_acquisitions

__all__ = ["Optimizer"]

ACQUISITION_FUNC = {
    "mes": acquisition.max_value_search,
    "ei": acquisition.expected_improvement,
    "ttei": acquisition.top_two_ei,
    "mean": acquisition.expectation,
    "lcb": acquisition.lcb,
}


class Optimizer(object):
    def __init__(
        self,
        dimensions,
        n_points=10000,
        n_initial_points=10,
        init_strategy="r2",
        gp_kernel=None,
        gp_kwargs=None,
        gp_priors=None,
        acq_func="mes",
        acq_func_kwargs=None,
        random_state=None,
    ):
        self.rng = check_random_state(random_state)

        if callable(acq_func):
            self.acq_func = acq_func
        else:
            self.acq_func = ACQUISITION_FUNC[acq_func]
        if acq_func_kwargs is None:
            acq_func_kwargs = dict()
        self.acq_func_kwargs = acq_func_kwargs

        self.space = normalize_dimensions(dimensions)
        self._n_initial_points = n_initial_points
        self.n_initial_points_ = n_initial_points
        self.init_strategy = init_strategy
        if self.init_strategy == "r2":
            self._initial_points = self.space.inverse_transform(r2_sequence(n=n_initial_points, d=self.space.n_dims))
        self.n_points = n_points

        # TODO: Maybe a variant of cook estimator?
        # TODO: Construct kernel if None
        if gp_kwargs is None:
            gp_kwargs = dict()
        if gp_priors is None:
            self.gp_priors = guess_priors(n_parameters=self.space.transformed_n_dims)
        self.gp = BayesGPR(kernel=gp_kernel, random_state=self.rng.randint(0, np.iinfo(np.int32).max), **gp_kwargs)

        self.Xi = []
        self.yi = []
        self._next_x = None

    def ask(self):
        if self._n_initial_points > 0:  # TODO: Make sure estimator is trained here always
            if self.init_strategy == "r2":
                return self._initial_points[self._n_initial_points - 1]
            return self.space.rvs()
        else:
            if not self.gp.kernel_:
                raise RuntimeError("Initialization is finished, but no model has been fit.")
            return self._next_x

    def tell(self, x, y, fit=True, n_samples=5):
        # if y isn't a scalar it means we have been handed a batch of points
        if is_listlike(y) and is_2Dlistlike(x):
            self.Xi.extend(x)
            self.yi.extend(y)
            self._n_initial_points -= len(y)
        elif is_listlike(x):
            self.Xi.append(x)
            self.yi.append(y)
            self._n_initial_points -= 1
        else:
            raise ValueError(f"Type of arguments `x` ({type(x)}) and `y` ({type(y)}) " "not compatible.")

        if fit and self._n_initial_points <= 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.gp.fit(self.space.transform(self.Xi), self.yi)

            X = self.space.transform(self.space.rvs(n_samples=self.n_points, random_state=self.rng))
            acq_values = evaluate_acquisitions(
                X=X,
                gpr=self.gp,
                acquisition_functions=(self.acq_func,),
                n_samples=n_samples,
                progress=False,
                random_state=self.rng.randint(0, np.iinfo(np.int32).max),
                **self.acq_func_kwargs,
            ).flatten()

            self._next_x = self.space.inverse_transform(X[np.argmax(acq_values)].reshape((1, -1)))[0]

        return create_result(self.Xi, self.yi, self.space, self.rng, models=[self.gp])

    def run(self, func, n_iter=1):
        for _ in range(n_iter):
            x = self.ask()
            self.tell(x, func(x))

        return create_result(self.Xi, self.yi, self.space, self.rng, models=[self.gp])
