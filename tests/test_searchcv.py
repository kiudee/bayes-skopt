from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from skopt.space import Categorical, Integer, Real

from bask.searchcv import BayesSearchCV


def test_searchcv_run():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=0
    )

    opt = BayesSearchCV(
        SVC(),
        {
            "C": Real(1e-6, 1e6, prior="log-uniform"),
            "gamma": Real(1e-6, 1e1, prior="log-uniform"),
            "degree": Integer(1, 8),
            "kernel": Categorical(["linear", "poly", "rbf"]),
        },
        n_iter=11,
        cv=None,
        random_state=0,
    )

    opt.fit(X_train, y_train)
    assert opt.score(X_test, y_test) > 0.89


def test_searchcv_best_mean():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=0
    )

    opt = BayesSearchCV(
        SVC(),
        {
            "C": Real(1e-6, 1e6, prior="log-uniform"),
            "gamma": Real(1e-6, 1e1, prior="log-uniform"),
            "degree": Integer(1, 8),
            "kernel": Categorical(["linear", "poly", "rbf"]),
        },
        n_iter=11,
        cv=None,
        return_policy="best_mean",
        random_state=0,
    )

    opt.fit(X_train, y_train)
    assert opt.score(X_test, y_test) > 0.89
