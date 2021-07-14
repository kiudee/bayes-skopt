"""Top-level package for Bayes-skopt."""
try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version

__author__ = """Karlson Pfannschmidt"""
__email__ = "kiudee@mail.upb.de"
__version__ = version("bask")

from .acquisition import *
from .bayesgpr import BayesGPR
from .init import r2_sequence, sb_sequence
from .optimizer import Optimizer
from .searchcv import BayesSearchCV
from .utils import guess_priors

__all__ = [
    "BayesGPR",
    "Optimizer",
    "BayesSearchCV",
    "guess_priors",
    "evaluate_acquisitions",
    "ExpectedImprovement",
    "TopTwoEI",
    "Expectation",
    "LCB",
    "MaxValueSearch",
    "r2_sequence",
    "sb_sequence",
    "ThompsonSampling",
    "VarianceReduction",
    "PVRS",
]
