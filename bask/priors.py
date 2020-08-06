import numpy as np
from scipy.integrate import quad

__all__ = ["make_roundflat"]


def make_roundflat(
    lower_bound=0.1,
    upper_bound=0.6,
    lower_steepness=2.0,
    upper_steepness=8.0,
    integration_bounds=(0.0, 10.0),
):
    """Construct a round-flat prior with a certain shape.

    The resulting prior is roughly flat inside the range
    (lower_bound, upper_bound)
    and the density drops smoothly the closer points are to the bounds.

    Parameters
    ----------
    lower_bound : float, default=0.1
        Number which specifies the lower bound of the plausible range.
    upper_bound : float, default=0.6
        Number which specifies the upper bound of the plausible range.
    lower_steepness : float, default=2.0
        Exponent which controls how steep the prior falls off when approaching
        the lower bound. Higher values result in a steeper drop.
    upper_steepness : float, default=8.0
        Exponent which controls how steep the prior falls off when approaching
        the upper bound. Higher values result in a steeper drop.
    integration_bounds : tuple of floats, default=(0.0, 10.0)
        The lower and upper bound of the range in which to integrate the prior.
        This is used to normalize the prior.
    Returns
    -------
    prior : function
        Prior distribution which for a given x returns the log density at that
        point.
    """

    def roundflat(x):
        return -2 * (
            (x / lower_bound) ** (-2 * lower_steepness)
            + (x / upper_bound) ** (2 * upper_steepness)
        )

    value = quad(
        lambda x: np.exp(roundflat(x)), integration_bounds[0], integration_bounds[1]
    )[0]

    def prior(x):
        return roundflat(x) - np.log(value)

    return prior
