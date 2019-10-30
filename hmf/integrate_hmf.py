"""
A supporting module that provides a routine to integrate the differential hmf in a robust manner.
"""
from scipy.interpolate import InterpolatedUnivariateSpline as _spline
import numpy as np
import scipy.integrate as intg

class NaNException(Exception):
    pass

def hmf_integral_gtm(M, dndm, mass_density=False):
    """
    Cumulatively integrate dn/dm.

    Parameters
    ----------
    M : array_like
        Array of masses.

    dndm : array_like
        Array of dn/dm (corresponding to M)

    mass_density : bool, `False`
        Whether to calculate mass density (or number density).

    Returns
    -------
    ngtm : array_like
        Cumulative integral of dndm.

    Examples
    --------
    Using a simple power-law mass function:

    >>> import numpy as np
    >>> m = np.logspace(10,18,500)
    >>> dndm = m**-2
    >>> ngtm = hmf_integral_gtm(m,dndm)
    >>> np.allclose(ngtm,1/m) #1/m is the analytic integral to infinity.
    True

    The function always integrates to m=1e18, and extrapolates with a spline
    if data not provided:

    >>> m = np.logspace(10,12,500)
    >>> dndm = m**-2
    >>> ngtm = hmf_integral_gtm(m,dndm)
    >>> np.allclose(ngtm,1/m) #1/m is the analytic integral to infinity.
    True
    
    """
    # Eliminate NaN's
    m = M[np.logical_not(np.isnan(dndm))]
    dndm = dndm[np.logical_not(np.isnan(dndm))]
    dndlnm = m * dndm

    if len(m) < 4:
        raise NaNException("There are too few real numbers in dndm: len(dndm) = %s, #NaN's = %s" % (len(M), len(M) - len(dndm)))

    # Calculate the mass function (and its integral) from the highest M up to 10**18
    if m[-1] < m[0] * 10 ** 18 / m[3]:
        m_upper = np.arange(np.log(m[-1]), np.log(10 ** 18), np.log(m[-1]) - np.log(m[-2]))
        mf_func = _spline(np.log(m), np.log(dndlnm), k=1)
        mf = mf_func(m_upper)

        if not mass_density:
            int_upper = intg.simps(np.exp(mf), x=m_upper, even='first')
        else:
            int_upper = intg.simps(np.exp(m_upper + mf), dx=m_upper, even='first')
    else:
        int_upper = 0


    # Calculate the cumulative integral (backwards) of [m*]dndm
    if not mass_density:
        # CHANGES: changed dndlnm to dndm since we use explicit
        # mass range instead of fixed integral, hence we don't need
        # the extra factor of m
        # need minus due to flipping integration range
        ngtm = np.concatenate((intg.cumtrapz(dndm[::-1], x=-m[::-1])[::-1],
                               np.zeros(1)))
    else:
        ngtm = np.concatenate((intg.cumtrapz(m[::-1] * dndm[::-1],
                                             x=-m[::-1])[::-1], np.zeros(1)))

    return (ngtm + int_upper)
