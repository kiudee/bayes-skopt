"""Top-level package for Bayes-skopt."""

__author__ = """Karlson Pfannschmidt"""
__email__ = "kiudee@mail.upb.de"
__version__ = "0.8.0-beta.0"

from .acquisition import *
from .bayesgpr import BayesGPR
from .optimizer import Optimizer
from .searchcv import BayesSearchCV
from .utils import guess_priors, optimum_intervals

__all__ = [
    "BayesGPR",
    "Optimizer",
    "BayesSearchCV",
    "optimum_intervals",
    "guess_priors",
    "evaluate_acquisitions",
    "ExpectedImprovement",
    "TopTwoEI",
    "Expectation",
    "LCB",
    "MaxValueSearch",
    "ThompsonSampling",
    "VarianceReduction",
    "PVRS",
]
