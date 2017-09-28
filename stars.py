import numpy as np
import scipy.interpolate as intrp

import halo.tools
import halo.parameters as p

import pdb

ddir = '/Volumes/Data/stijn/Documents/Leiden/MR/code/halo/data/'

def f_interp(m_range, f, m_f):
    '''
    Return fraction interpolated for m_range

    Parameters
    ----------
    m_range : (m,...) array
      mass range to interpolate to

    f : (m',...) array
      fractions from data

    m_f : (m',...) array
      mass associated to f

    Returns
    -------
    f_interp : (m,) array
      interpolated fractions
    '''
    f_interp = intrp.interp1d(m_f, f, axis=0)
    mask = ((m_range < m_f[0]) & (m_range > m_f[-1]))
    f = np.zeros_like(m_range)
    f[~mask] = f_interp(m_range[~mask])

    return f

# ------------------------------------------------------------------------------
# End of f_interp()
# ------------------------------------------------------------------------------

def mh_theory(m_cen):
    '''
    Returns the mean halo mass for fixed central stellar mass m_cen,
    using the iHOD model from Zu & Mandelbaum (2015).

        http://dx.doi.org/10.1093/mnras/stv2062

    Parameters
    ----------
    m_cen : array
      central stellar masses to compute halo mass for

    Returns
    -------
    m_h : array
      corresponding halo masses
    '''
    M_1h = np.power(10, 12.10)
    m = m_cen / (np.power(10, 10.31))
    beta = 0.33
    delta = 0.42
    gamma = 1.21

    m_h = M_1h * m**beta * np.exp(m**delta / (1 + m**(-gamma)) - 0.5)
    return m_h

# ------------------------------------------------------------------------------
# End of mh_theory()
# ------------------------------------------------------------------------------

def mcen_theory(m_h):
    '''
    Returns the central stellar mass m_s for the mean halo mass m_h,
    using iHOD model from Zu & Mandelbaum (2015).

         http://dx.doi.org/10.1093/mnras/stv2062

    Parameters
    ----------
    m_h : array
      halo masses to compute stellar mass for

    Returns
    -------
    m_cen : array
      corresponding stellar masses
    '''
    mh_1 = tools.inverse(mh_theory)
    m_cen = np.ones_like(m_h)
    for idx, val in enumerate(m_h):
        m_cen[idx] = mh_1(val)

    return m_cen

# ------------------------------------------------------------------------------
# End of mcen_theory()
# ------------------------------------------------------------------------------

def mh_fit(m_cen):
    '''
    Returns the mean halo mass for fixed central stellar mass m_cen, using
    the iHOD model from Zu & Mandelbaum (2015), fitting function Eq. 59.

        http://dx.doi.org/10.1093/mnras/stv2062

    Parameters
    ----------
    m_cen : array
      central stellar masses to compute halo mass for

    Returns
    -------
    m_h : array
      corresponding halo masses
    '''
    logm_h = 4.41/(1 + np.exp(-1.82 * (np.log10(m_cen) - 11.18))) + \
             11.12 * np.sin(-0.12 * (np.log10(m_cen) - 23.37))
    m_h = np.power(10, logm_h)
    return m_h

# ------------------------------------------------------------------------------
# End of mh_fit()
# ------------------------------------------------------------------------------

def mcen_fit(m_h):
    '''
    Returns the central stellar mass m_cen for the mean halo mass m_h,
    using the iHOD model from Zu & Mandelbaum (2015), inverse of eq. 59.

         http://dx.doi.org/10.1093/mnras/stv2062

    Parameters
    ----------
    m_h : array
      halo masses to compute stellar mass for

    Returns
    -------
    m_cen : array
      corresponding stellar masses
    '''
    mh_1 = tools.inverse(mh_fit)
    m_cen = np.ones_like(m_h)
    for idx, val in enumerate(m_h):
        m_cen[idx] = mh_1(val)

    return m_cen

# ------------------------------------------------------------------------------
# End of mcen_fit()
# ------------------------------------------------------------------------------

def f_s(m_range):
    '''
    Satellite fraction from Zu & Mandelbaum
    '''
    m_f, f, f_c, f_s = np.loadtxt(ddir +
                                  'data_mccarthy/stars/StellarFraction-Mh.txt',
                                  unpack=True)
    f_s_interp = f_interp(m_range, f_s, m_f)

    return f_s_interp

# ------------------------------------------------------------------------------
# End of f_s()
# ------------------------------------------------------------------------------

def f_c(m_range):
    '''
    Central fraction from Zu & Mandelbaum
    '''
    m_f, f, f_c, f_s = np.loadtxt(ddir +
                                  'data_mccarthy/stars/StellarFraction-Mh.txt',
                                  unpack=True)
    f_c_interp = f_interp(m_range, f_c, m_f)

    return f_c_interp

# ------------------------------------------------------------------------------
# End of f_c()
# ------------------------------------------------------------------------------
