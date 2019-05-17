import time
import inspect
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.interpolate import interp1d
from scipy.special import hyp2f1, factorial
import scipy.optimize as opt
# import commah

import pdb

def _check_iterable(prms):
    '''
    Go through prms and make them iterable. Useful if not sure if input
    is scalar or array

    Parameters
    ----------
    prms : list
      list of parameters to check

    Returns
    -------
    prms_iterable : list
      list of all parameters as lists
    '''
    prms_iterable = []
    for prm in prms:
        prm = np.asarray(prm)
        if prm.shape == ():
            prm = prm.reshape(-1)
        prms_iterable.append(prm)

    return prms_iterable

# ------------------------------------------------------------------------------
# End of _check_iterable()
# ------------------------------------------------------------------------------

def lte(a, b, precision=1e-15):
    '''
    Compare a and b with precision set to 1e-15, since we get some roundoff
    errors with float precision

    Parameters
    ----------
    a : array
    b : array
    precision : float
      precision to which we want to compare

    Returns
    -------
    c : array
      index array where a <= b within 1e-16
    '''
    # these are all the elements that are strictly smaller or equal
    return ((a - b) <= precision)

# ------------------------------------------------------------------------------
# End of lte()
# ------------------------------------------------------------------------------

def gte(a, b, precision=1e-15):
    '''
    Compare a and b with precision set to 1e-15, since we get some roundoff
    errors with float precision

    Parameters
    ----------
    a : array
    b : array
    precision : float
      precision to which we want to compare

    Returns
    -------
    c : array
      index array where a >= b within 1e-16
    '''
    # these are all the elements that are strictly smaller or equal
    return ((a - b) >= -precision)

# ------------------------------------------------------------------------------
# End of lte()
# ------------------------------------------------------------------------------

def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

# ------------------------------------------------------------------------------
# End of merge_dicts()
# ------------------------------------------------------------------------------

def Integrate(y, x, axis=-1): # Simpson integration on fixed spaced data!
	'''
    Integrate array at sample points x over axis using Simpson
	integration

    Parameters
    ----------
    y : array

    '''
	y_new = np.nan_to_num(y)
        # the last interval is computed with trapz
	result = scipy.integrate.simps(y=y_new, x=x, axis=axis, even='first')
	# result = np.trapz(y=y_new, x=x, axis=axis)

	return result

# ------------------------------------------------------------------------------
# End of Integrate()
# ------------------------------------------------------------------------------

def median_slices(data, medians, bins):
    '''
    Return slices for data such that the median of each slice returns medians

    Parameters
    ----------
    data : (n,) array
      array to slice
    medians : (m,) array
      medians to match
    bins : (m,2) array
      maximum allowed bin around median

    Returns
    -------
    slices : (m,2) array
      array containing slices for each median
    '''
    data_sorted = np.sort(data, axis=0)

    # index of elements matching medians
    idx_med = np.argmin(np.abs(data_sorted.reshape(-1,1) - medians.reshape(1,-1)),
                        axis=0)
    # find matching bin indeces
    idx_bin = np.argmin(np.abs(data_sorted.reshape(-1,1,1) - bins.reshape(1,-1,2)),
                        axis=0)
    # get minimum distance from bins to median -> this will be our slice
    min_dist = np.min(np.abs(idx_med.reshape(-1,1) - idx_bin), axis=1)
    slices = np.concatenate([(idx_med - min_dist).reshape(-1,1),
                             (idx_med + min_dist).reshape(-1,1)],
                            axis=1)

    return slices

# ------------------------------------------------------------------------------
# End of median_slices()
# ------------------------------------------------------------------------------

def c_correa(m200c, z_range=0, h=0.7, cosmology='WMAP9'):
    '''
    Returns the mass-concentration relation from Correa et al (2015c)
    through the commah code.

    Parameters
    ----------
    m200c : (m,) array
      array containing masses to compute NFW profile for (mass at z=0)
    z_range : (z,) array
      redshift to evaluate mass-concentration relation at
    cosmology : string or dict for commah
      cosmological parameters for commah

    Returns
    -------
    c : (z,m) array
      array containing concentration for each (m,z)
    '''
    # (z,m) array
    c = commah.run(cosmology=cosmology, Mi=m200c/h, z=z_range, mah=False)['c'].T
    return c

# ------------------------------------------------------------------------------
# End of c_correa()
# ------------------------------------------------------------------------------

# def c200m_correa(m200m, z_range=0, h=0.7):
#     '''
#     Returns the mass-concentration relation from Correa et al (2015c)
#     through the commah code.

#     Parameters
#     ----------
#     m200m : (m,) array
#       array containing masses to compute NFW profile for (mass at z=0)
#     z_range : (z,) array
#       redshift to evaluate mass-concentration relation at
#     cosmology : string or dict for commah
#       cosmological parameters for commah

#     Returns
#     -------
#     c : (z,m) array
#       array containing concentration for each (m,z)
#     '''
#     # (z,m) array
#     c = ct
#     c = commah.run(cosmology=cosmology, Mi=m200c/h, z=z_range, mah=False)['c'].T
#     return c

# # ------------------------------------------------------------------------------
# # End of c_correa()
# # ------------------------------------------------------------------------------

# def c_correa_fit(m_range, z_range=0, h=0.7):
#     '''
#     Returns the mass-concentration relation from Correa et al (2015c)
#     through the commah code.

#     Parameters
#     ----------
#     m_range : (m,) array
#       array containing masses to compute NFW profile for (mass at z=0)
#     z_range : (z,) array
#       redshift to evaluate mass-concentration relation at

#     Returns
#     -------
#     c : (z,m) array
#       array containing concentration for each (m,z)
#     '''
#     m_range, z_range = _check_iterable([m_range, z_range])
#     m = m_range.shape[0]
#     z = z_range.shape[0]

#     # discern high and low z
#     lo_idx = (0. <= z_range) & (z_range <= 4.)
#     hi_idx = (z_range > 4.)

#     # reshape to match our output requirement
#     m_range = m_range.reshape([1,m])
#     z_range = np.array(z_range).reshape([z,1])

#     lo_z = z_range[lo_idx]
#     hi_z = z_range[hi_idx]

#     if lo_z.size > 0:
#         # fit coefficients for lo_z
#         a_lo = 1.7543 - 0.2766 * (1 + lo_z) + 0.02039 * (1 + lo_z)**2
#         b_lo = 0.2753 + 0.00351 * (1 + lo_z) - 0.3038 * (1 + lo_z)**(0.0269)
#         g_lo = -0.01537 + 0.02102 * (1 + lo_z)**(-0.1475)

#         # log10c for lo_z
#         log10c_lo = a_lo + b_lo * np.log10(m_range) * (1 + g_lo * np.log10(m_range)**2)

#     if hi_z.size > 0:
#         # fit coefficients for hi_z
#         a_hi = 1.3081 - 0.1078 * (1 + hi_z) + 0.00398 * (1 + hi_z)**2
#         b_hi = 0.0223 - 0.0944 * (1 + hi_z)**(-0.3907)

#         # log10c for lo_z
#         log10c_hi = a_hi + b_hi * np.log10(m_range)

#     if (lo_z.size > 0) and (hi_z.size > 0):
#         log10c = np.concatenate((log10c_lo, log10c_hi), axis=0)
#     elif (lo_z.size > 0) and (hi_z.size == 0):
#         log10c = log10c_lo
#     elif (lo_z.size == 0) and (hi_z.size > 0):
#         log10c = log10c_hi
#     else:
#         raise ValueError('need to provide positive redshifts')

#     return np.power(10, log10c)

# # ------------------------------------------------------------------------------
# # End of c_correa_fit()
# # ------------------------------------------------------------------------------

def c_duffy(m_range, z_range=0., sigma_lnc=0.):
    '''
    Concentration mass relation of Duffy+08 
    (mean relation for full sample between 0<z<2)
    '''
    m_range, z_range = _check_iterable([m_range, z_range])
    m = m_range.shape[0]
    z = z_range.shape[0]

    m_range = m_range.reshape([1,m])
    z_range = np.array(z_range).reshape([z,1])

    A = 10.14
    B = -0.081
    C = -1.01

    plaw =  A * (m_range/(2e12))**B * (1+z_range)**C * np.e**sigma_lnc
    return plaw

# def c_duffy(m_range, z_range=0., m_pivot=1e14, A=5.05, B=-.101, C=0.):
#     '''
#     Returns the mass-concentration relation from Duffy et al (2008).

#         c = A * (m/m_pivot)^B * (1+z)^C

#     Parameters
#     ----------
#     m_range : (m,) array
#       array containing masses to compute NFW profile for (mass at z=0)
#     z_range : (z,) array
#       redshift to evaluate mass-concentration relation at
#     m_pivot : float (same units as m_range)
#       pivot scale for fit
#     A : float
#       scale parameter for fit
#     B : float
#       mass parameter for fit
#     C : float
#       redshift parameter for fit

#     Returns
#     -------
#     c : (m,z) array
#       array containing concentration for each (m,z)
#     '''
#     m_range, z_range = _check_iterable([m_range, z_range])
#     m = m_range.shape[0]
#     z = z_range.shape[0]

#     m_range = m_range.reshape([m,1])
#     z_range = np.array(z_range).reshape([1,z])

#     c = A * (m_range/m_pivot)**B * (1+z_range)**C
#     return c

# # ------------------------------------------------------------------------------
# # End of c_duffy()
# # ------------------------------------------------------------------------------

def profile_NFW(r_range, m_range, c_x, r_x, rho_mean, z_range=0, Delta=200.):
    '''
    Returns an NFW profile for m_range along axis 0 and r_range along
    axis 1 (with optional z_range along axis 2).

    Parameters
    ----------
    r_range : (m,r) array
      array containing r_range for each M (with r_range[:,-1] = r_vir)
    m_range : (m,) array
      array containing masses to compute NFW profile for (mass at z=0)
    c_x : (m,) or (m,z) array
      array containing mass-concentration relation
    r_x : (m,) or (m,z) array
      array containing r_x to evaluate r_s from r_s = r_x/c_x
    rho_mean : float
      mean dark matter density
    z_range : float/array (default=0)
      redshift range
    Delta : float (default=200.)
      critical overdensity for collapse

    Returns
    -------
    profile : (m,r)
      array containing NFW profile
    or (if z is an array)
    profile : (m,r,z)
      array containing NFW profile
    '''
    m = m_range.shape[0]
    r = r_range.shape[-1]
    # want an empty array for scalar z so we can easily construct shapes
    # later on
    z = np.array(np.array(z_range).shape)

    # (m,z) array
    r_s = r_x.reshape([m] + list(z))/c_x.reshape([m] + list(z))
    rho_s = Delta/3. * rho_mean * c_x**3/(np.log(1+c_x) - c_x/(1+c_x))

    # (m,r,z) array
    x = r_range.reshape([m,r] + list(z/z)) / r_s.reshape([m,1] + list(z))

    profile = rho_s.reshape([m,1] + list(z)) / (x * (1+x)**2)

    return profile

# ------------------------------------------------------------------------------
# End of profile_NFW()
# ------------------------------------------------------------------------------

def m_h(rho, r_range, r_0=None, r_1=None, axis=-1):
    '''
    Calculate the mass of the density profile between over r_range, or between
    r_0 and r_1.

    Parameters
    ----------
    rho : array
      density profile
    r_range : array
      radius
    r_0 : float
      start radius of integration interval
    r_1 : float
      end radius of integration interval

    Returns
    -------
    m_h : float
      mass
    '''
    int_range = r_range
    int_rho = rho

    if r_0 != None:
        idx_0 = np.argmin(np.abs(r_0 - int_range))
        int_range = int_range[...,idx_0:]
        int_rho = int_rho[...,idx_0:]

    if r_1 != None:
        idx_1 = np.argmin(np.abs(r_1 - int_range))
        int_range = int_range[...,:idx_1]
        int_rho = int_rho[...,:idx_1]

    return 4 * np.pi * Integrate(int_rho * int_range**2, int_range, axis=axis)

# ------------------------------------------------------------------------------
# End of m_h()
# ------------------------------------------------------------------------------

def cum_m(rho_r, r_range):
    '''
    Returns the cumulative mass profile

    Parameters
    ----------
    rho_r : (...,r) array
      density profile
    r_range : (...,r) array
      radial range

    Returns
    -------
    cum_m : (...,r-1) array
      cumulative mass profile
    '''
    r = r_range.shape[-1]

    cum_m = np.array([tools.m_h(rho_r[...,:idx],
                                r_range[...,:idx], axis=-1)
                      for idx in np.arange(1, r+1)])

    return cum_m

# ----------------------------------------------------------------------
# End of cum_m()
# ----------------------------------------------------------------------

def m_NFW(r, c_x, r_x, rho_mean, Delta=200):
    '''
    Calculate the mass of the NFW profile with c_x and r_x, relative to
    Delta rho_meanup to r

    Parameters
    ----------
    r : array
      radii to compute mass for
    c_x : float
      concentration of halo
    r_x : float
      array containing r_x to evaluate r_s from r_s = r_x/c_x
    rho_mean : float
      mean dark matter density
    Delta : float (default=200.)
      critical overdensity for collapse

    Returns
    -------
    m_h : float
      mass
    '''
    rho_s = Delta/3. * rho_mean * c_x**3/(np.log(1+c_x) - c_x/(1+c_x))
    r_s = r_x / c_x

    prefactor = 4 * np.pi * rho_s * r_s**3
    c_factor  = np.log((r_s + r) / r_s) - r / (r + r_s)

    mass = prefactor * c_factor

    return mass

# ------------------------------------------------------------------------------
# End of m_NFW()
# ------------------------------------------------------------------------------

def m_beta(r, beta, r_c, mgas_500c, r500c):
    '''
    Return the analytic enclosed mass for the beta profile normalized to
    mgas_500c at r500c

    Parameters
    ----------
    r : float or (r,) array if (m,) array, assume matched
      radii to compute for
    beta : float or (m,) array
      beta slope of the profile
    r_c : float or (m,) array
      !!!physical!!! core radius r_c of the profile
    mgas_500c : float or (m,) array
      gas mass at r500c
    r500c : float or (m,) array
      radius corresponding to halo mass m500c
    '''
    r, beta, r_c, mgas_500c, r500c = _check_iterable([r, beta, r_c,
                                                      mgas_500c, r500c])

    if r.shape != beta.shape:
        # reshape inputs
        r = r.reshape(-1,1)
        beta = beta.reshape(1,-1)
        r_c = r_c.reshape(1,-1)
        mgas_500c = mgas_500c.reshape(1,-1)
        r500c = r500c.reshape(1,-1)

    norm = (4./3 * np.pi * r500c**3 * hyp2f1(3./2, 3 * beta / 2,
                                             5./2, -(r500c / r_c)**2))
    rho_0 = mgas_500c / norm
    m = 4./3 * np.pi * rho_0 * r**3 * hyp2f1(3./2, 3 * beta / 2,
                                             5./2, -(r/r_c)**2)

    return m

# ------------------------------------------------------------------------------
# End of m_beta()
# ------------------------------------------------------------------------------

@np.vectorize
def r_where_m_beta(m, beta, r_c, mgas_500c, r500c):
    '''
    Return the radius where the beta profile mass is m

    Parameters
    ----------
    m : float (m,) array
      masses to get r for
    beta : float or (m,) array
      beta slope of the profile
    r_c : float or (m,) array
      !!!physical!!! core radius r_c of the profile
    mgas_500c : float or (m,) array
      gas mass at r500c
    r500c : float or (m,) array
      radius corresponding to halo mass m500c
    '''
    r = opt.brentq(lambda r, m, beta, r_c, mgas_500c, r500c: \
                       m - m_beta(r, beta, r_c, mgas_500c, r500c),
                   0, 100, args=(m, beta, r_c, mgas_500c, r500c))

    return r

def mass_to_radius(m, mean_dens):
    '''
    Calculate radius of a region of space from its mass.

    Parameters
    ----------
    m : float or array of floats
      Masses
    mean_dens : float
      The mean density of the universe

    Returns
    -------
    r : float or array of floats
      The corresponding radii to m

    Notes
    -----
    The units of r don't matter as long as they are consistent with
    mean_dens.
    '''
    return np.power((3.*m / (4.*np.pi * mean_dens)), 1./3, dtype=float)

# ------------------------------------------------------------------------------
# End of mass_to_radius()
# ------------------------------------------------------------------------------

def radius_to_mass(r, mean_dens):
    '''
    Calculates mass of a region of space from its radius.

    Parameters
    ----------
    r : float or array of floats
      Radii
    mean_dens : float
      The mean density of the universe

    Returns
    -------
    m : float or array of floats
      The corresponding masses in r

    Notes
    -----
    The units of r don't matter as long as they are consistent with
    mean_dens.
    '''
    return 4 * np.pi * r ** 3 * mean_dens / 3.

# ------------------------------------------------------------------------------
# End of radius_to_mass()
# ------------------------------------------------------------------------------

def sigma_from_rho(R, r_range, rho):
    """
    Project a 3-D density profile to a surface density profile.

    Parameters
    ----------
    R : (R,) array
        projected radii
    r_range : (r,) array
        radial range for the density profile
    rho : (r,) array
        density profile at r_range

    Returns
    -------
    sigma : (R,) array
        mass surface density from rho at R
    """
    sigma = np.zeros(R.shape)
    for idx, r in enumerate(R):
        rho_int = interp1d(r_range, rho)
        # can neglect first part of integral since integrand converges to 0 there
        r_int = np.logspace(np.log10(r + 1e-5), np.log10(r_range.max()), 150)
        integrand = 2 * rho_int(r_int) * r_int / np.sqrt(r_int**2 - r**2)
        sigma[idx] = Integrate(integrand, r_int)
        
    return sigma

# ------------------------------------------------------------------------------
# End of sigma_from_rho()
# ------------------------------------------------------------------------------

def m500c_Xray_from_m500c(m500c, cosmo, f_stars):
    '''
    For a given m500c in the DMO case, determine the corresponding observed X-ray
    result.

    Parameters
    ----------
    m500c : array
      DMO halo masses
    cosmo : dict
      cosmological parameters
    f_stars : array
      stellar fraction
    '''

def massdiff_2m5c_correa(m200m, m500c, rhoc, rhom, h, z):
    '''
    Integrate an NFW halo with m200m up to r500c and return the mass difference
    between the integral and m500c
    '''
    # compute radii
    r500c = (m500c / (4./3 * np.pi * 500 * rhoc))**(1./3)
    r200m = mass_to_radius(m200m, 200 * rhom)

    # compute concentration
    m200c = m200m_to_m200c(m200m, rhoc, rhom, h)
    c200c = c_correa(m200c, z, h).reshape(-1)
    r200c = mass_to_radius(m200c, 200 * rhoc)

    # now get analytic mass
    mass = m_NFW(r500c, c200c * r200m / r200c, r200m, rhom, Delta=200.)

    return mass - m500c

@np.vectorize
def m500c_to_m200m_correa(m500c, rhoc, rhom, h, z=0):
    '''
    Give the virial mass for the halo corresponding to m500c

    Parameters
    ----------
    m500c : float
      halo mass at 500 times the universe critical density

    Returns
    -------
    m200m : float
      corresponding halo model halo virial mass
    '''
    # these bounds should be reasonable for m200m < 1e18
    m200m = opt.brentq(massdiff_2m5c_correa, m500c, 10. * m500c,
                       args=(m500c, rhoc, rhom, h, z))

    return m200m

# ------------------------------------------------------------------------------
# End of m500c_to_m200m_correa()
# ------------------------------------------------------------------------------

def massdiff_5c2m_correa(m500c, m200m, m200c, rhoc, rhom, h, z):
    '''
    Integrate an NFW halo with m200m up to r500c and return the mass difference
    between the integral and m500c
    '''
    # compute radii
    r500c = (m500c / (4./3 * np.pi * 500 * rhoc))**(1./3)
    r200m = mass_to_radius(m200m, 200 * rhom)
    r200c = mass_to_radius(m200c, 200 * rhoc)

    # compute concentration
    c200c = c_correa(m200c, z, h).reshape(-1)

    # now get analytic mass
    mass = m_NFW(r500c, c200c * r200m / r200c, r200m, rhom, Delta=200)

    return mass - m500c

@np.vectorize
def m200m_to_m500c_correa(m200m, rhoc, rhom, h, m200c=None, z=0):
    '''
    Give m500c for the an m200m virial mass halo

    Parameters
    ----------
    m200m : float
      halo virial mass

    Returns
    -------
    m500c : float
      halo mass at 500 times the universe critical density
    '''
    # 1e19 Msun is ~maximum for c_correa
    if m200c == None:
        m200c = m200m_to_m200c_correa(m200m, rhoc, rhom, h, z)

    # these bounds should be reasonable for m200m < 1e18
    m500c = opt.brentq(massdiff_5c2m_correa, m200m/10., m200m,
                       args=(m200m, m200c, rhoc, rhom, h, z))

    return m500c

# ------------------------------------------------------------------------------
# End of m200m_to_m500c_correa()
# ------------------------------------------------------------------------------

def massdiff_2m2c_correa(m200m, m200c, rhoc, rhom, h, z):
    '''
    Integrate an NFW halo with m200m up to r200c and return the mass difference
    between the integral and m200c
    '''
    # compute radii
    r200c = (m200c / (4./3 * np.pi * 200 * rhoc))**(1./3)
    r200m = mass_to_radius(m200m, 200 * rhom)

    # compute concentration
    c200c = c_correa(m200c, z, h).reshape(-1)

    # now compute analytic mass
    mass = m_NFW(r200c, c200c * r200m / r200c, r200m, rhom, Delta=200)

    return mass - m200c

@np.vectorize
def m200c_to_m200m_correa(m200c, rhoc, rhom, h, z=0):
    '''
    Give the virial mass for the halo corresponding to m200c

    Parameters
    ----------
    m200c : float
      halo mass at 200 times the universe critical density

    Returns
    -------
    m200m : float
      corresponding halo model halo virial mass
    '''
    # these bounds should be reasonable for m200m < 1e18
    # 1e19 Msun is ~maximum for c_correa
    m200m = opt.brentq(massdiff_2m2c_correa, m200c, 10. * m200c,
                       args=(m200c, rhoc, rhom, h, z))

    return m200m

# ------------------------------------------------------------------------------
# End of m200c_to_m200m_correa()
# ------------------------------------------------------------------------------

def massdiff_2c2m_correa(m200c, m200m, rhoc, rhom, h, z):
    '''
    Integrate an NFW halo with m200m up to r200c and return the mass difference
    between the integral and m200c
    '''
    # compute radii
    r200c = (m200c / (4./3 * np.pi * 200 * rhoc))**(1./3)
    r200m = mass_to_radius(m200m, 200 * rhom)

    # compute concentration
    c200c = c_correa(m200c, z, h).reshape(-1)

    # now get analytic mass
    mass = m_NFW(r200c, c200c * r200m / r200c, r200m, rhom, Delta=200)

    return mass - m200c

@np.vectorize
def m200m_to_m200c_correa(m200m, rhoc, rhom, h, z=0):
    '''
    Give m200c for the an m200m virial mass halo

    Parameters
    ----------
    m200m : float
      halo virial mass

    Returns
    -------
    m200c : float
      halo mass at 200 times the universe critical density
    '''
    # these bounds should be reasonable for m200m < 1e18
    # 1e19 Msun is ~maximum for c_correa
    m200c = opt.brentq(massdiff_2c2m_correa, m200m / 10., m200m,
                       args=(m200m, rhoc, rhom, h, z))

    return m200c

# ------------------------------------------------------------------------------
# End of m200m_to_m200c_correa()
# ------------------------------------------------------------------------------

def massdiff_2c5c_correa(m200c, m500c, rhoc, rhom, h, z):
    '''
    Integrate an NFW halo with m200c up to r500c and return the mass difference
    between the integral and m500c
    '''
    # compute radii
    r500c = (m500c / (4./3 * np.pi * 500 * rhoc))**(1./3)
    r200c = mass_to_radius(m200c, 200 * rhoc)

    # compute concentration
    c200c = c_correa(m200c, z, h).reshape(-1)

    # now get analytic mass
    mass = m_NFW(r500c, c200c, r200c, rhoc, Delta=200.)

    mass_int = m_h(dens, r_range)

    return mass_int - m500c

@np.vectorize
def m500c_to_m200c_correa(m500c, rhoc, rhom, h, z=0):
    '''
    Give m200c for the an m500c virial mass halo

    Parameters
    ----------
    m500c : float [M_sun / h]
      halo mass at 500 times the universe critical density

    Returns
    -------
    m200c : float [M_sun / h]
      halo mass at 200 times the universe critical density
    '''
    # these bounds should be reasonable for m200m < 1e18
    # 1e19 Msun is ~maximum for c_correa
    m200c = opt.brentq(massdiff_2c5c_correa, m500c, 10. * m500c,
                       args=(m500c, rhoc, rhom, h, z))

    return m200c

# ------------------------------------------------------------------------------
# End of m500c_to_m200c_correa()
# ------------------------------------------------------------------------------

def massdiff_2m5c_duffy(m200m, m500c, rhoc, rhom, z):
    '''
    Integrate an NFW halo with m200m up to r500c and return the mass difference
    between the integral and m500c
    '''
    # compute radii
    r500c = (m500c / (4./3 * np.pi * 500 * rhoc))**(1./3)
    r200m = mass_to_radius(m200m, 200 * rhom)

    c200m = c_duffy(m200m, z).reshape(-1)

    # now get analytic mass
    mass = m_NFW(r500c, c200m, r200m, rhom, Delta=200.)

    return mass - m500c

@np.vectorize
def m500c_to_m200m_duffy(m500c, rhoc, rhom, z=0):
    '''
    Give the virial mass for the halo corresponding to m500c

    Parameters
    ----------
    m500c : float
      halo mass at 500 times the universe critical density

    Returns
    -------
    m200m : float
      corresponding halo model halo virial mass
    '''
    # these bounds should be reasonable for m200m < 1e18
    m200m = opt.brentq(massdiff_2m5c_duffy, m500c, 10. * m500c,
                       args=(m500c, rhoc, rhom, z))

    return m200m

# ------------------------------------------------------------------------------
# End of m500c_to_m200m_duffy()
# ------------------------------------------------------------------------------

def massdiff_5c2m_duffy(m500c, m200m, rhoc, rhom, z):
    '''
    Integrate an NFW halo with m200m up to r500c and return the mass difference
    between the integral and m500c
    '''
    # compute radii
    r500c = (m500c / (4./3 * np.pi * 500 * rhoc))**(1./3)
    r200m = mass_to_radius(m200m, 200 * rhom)

    # compute concentration
    c200m = c_duffy(m200m, z).reshape(-1)

    # now get analytic mass
    mass = m_NFW(r500c, c200m, r200m, rhom, Delta=200)

    return mass - m500c

@np.vectorize
def m200m_to_m500c_duffy(m200m, rhoc, rhom, z=0):
    '''
    Give m500c for the an m200m virial mass halo

    Parameters
    ----------
    m200m : float
      halo virial mass

    Returns
    -------
    m500c : float
      halo mass at 500 times the universe critical density
    '''
    # these bounds should be reasonable for m200m < 1e18
    m500c = opt.brentq(massdiff_5c2m_duffy, m200m/10., m200m,
                       args=(m200m, rhoc, rhom, z))

    return m500c

# ------------------------------------------------------------------------------
# End of m200m_to_m500c_duffy()
# ------------------------------------------------------------------------------

def massdiff_2m2c_duffy(m200m, m200c, rhoc, rhom, z):
    '''
    Integrate an NFW halo with m200m up to r200c and return the mass difference
    between the integral and m200c
    '''
    # compute radii
    r200c = (m200c / (4./3 * np.pi * 200 * rhoc))**(1./3)
    r200m = mass_to_radius(m200m, 200 * rhom)

    # compute concentration
    c200m = c_duffy(m200m, z).reshape(-1)

    # now compute analytic mass
    mass = m_NFW(r200c, c200m, r200m, rhom, Delta=200)

    return mass - m200c

@np.vectorize
def m200c_to_m200m_duffy(m200c, rhoc, rhom, z=0):
    '''
    Give the virial mass for the halo corresponding to m200c

    Parameters
    ----------
    m200c : float
      halo mass at 200 times the universe critical density

    Returns
    -------
    m200m : float
      corresponding halo model halo virial mass
    '''
    # these bounds should be reasonable for m200m < 1e18
    # 1e19 Msun is ~maximum for c_duffy
    m200m = opt.brentq(massdiff_2m2c_duffy, m200c, 10. * m200c,
                       args=(m200c, rhoc, rhom, z))

    return m200m

# ------------------------------------------------------------------------------
# End of m200c_to_m200m_duffy()
# ------------------------------------------------------------------------------

def massdiff_2c2m_duffy(m200c, m200m, rhoc, rhom, z):
    '''
    Integrate an NFW halo with m200m up to r200c and return the mass difference
    between the integral and m200c
    '''
    # compute radii
    r200c = (m200c / (4./3 * np.pi * 200 * rhoc))**(1./3)
    r200m = mass_to_radius(m200m, 200 * rhom)

    # compute concentration
    c200m = c_duffy(m200m, z).reshape(-1)

    # now get analytic mass
    mass = m_NFW(r200c, c200m, r200m, rhom, Delta=200)

    return mass - m200c

@np.vectorize
def m200m_to_m200c_duffy(m200m, rhoc, rhom, z=0):
    '''
    Give m200c for the an m200m virial mass halo

    Parameters
    ----------
    m200m : float
      halo virial mass

    Returns
    -------
    m200c : float
      halo mass at 200 times the universe critical density
    '''
    # these bounds should be reasonable for m200m < 1e18
    # 1e19 Msun is ~maximum for c_duffy
    m200c = opt.brentq(massdiff_2c2m_duffy, m200m / 10., m200m,
                       args=(m200m, rhoc, rhom, z))

    return m200c

# ------------------------------------------------------------------------------
# End of m200m_to_m200c_duffy()
# ------------------------------------------------------------------------------

def massdiff_2c5c_duffy(m200c, m500c, rhoc, rhom):
    '''
    Integrate an NFW halo with m200c up to r500c and return the mass difference
    between the integral and m500c
    '''
    # compute radii
    r500c = (m500c / (4./3 * np.pi * 500 * rhoc))**(1./3)
    r200c = mass_to_radius(m200c, 200 * rhoc)

    m200m = m200c_to_m200m_duffy(m200c, rhoc, rhom)
    r200m = mass_to_radius(m200m, 200 * rhom)

    # compute concentration
    c200m = c_duffy(m200m, z).reshape(-1)

    # now get analytic mass
    mass = m_NFW(r500c, c200m, r200m, rhom, Delta=200.)

    mass_int = m_h(dens, r_range)

    return mass_int - m500c

@np.vectorize
def m500c_to_m200c_duffy(m500c, rhoc, rhom, z=0):
    '''
    Give m200c for the an m500c virial mass halo

    Parameters
    ----------
    m500c : float [M_sun / h]
      halo mass at 500 times the universe critical density

    Returns
    -------
    m200c : float [M_sun / h]
      halo mass at 200 times the universe critical density
    '''
    # these bounds should be reasonable for m200m < 1e18
    # 1e19 Msun is ~maximum for c_duffy
    m200c = opt.brentq(massdiff_2c5c_duffy, m500c, 10. * m500c,
                       args=(m500c, rhoc, rhom, z))

    return m200c

# ------------------------------------------------------------------------------
# End of m500c_to_m200c_duffy()
# ------------------------------------------------------------------------------

def find_bounds(f, y, start=1., **f_prms):
    x = start
    while f(x, **f_prms) < y:
        x = x * 2
    lo = 0 if (x == 1) else x/2.
    return lo, x

def binary_search(f, y, lo, hi, delta, **f_prms):
    while lo <= hi:
        x = (lo + hi) / 2.
        if f(x, **f_prms) < y:
            lo = x + delta
        elif f(x, **f_prms) > y:
            hi = x - delta
        else:
            return x;
    return hi if (f(hi, **f_prms) - y < y - f(lo, **f_prms)) else lo

def inverse(f, delta=1/1024., start=1., **f_prms):
    ''''
    Returns the inverse of the monotonic function f, to a precision of delta.

    Parameters
    ----------
    f : function
      function to invert
    delta : float
      precision

    Returns
    -------
    f_1 : function
      inverse of f
    '''
    def f_1(y, **f_prms):
        lo, hi = find_bounds(f, y, start, **f_prms)
        return binary_search(f, y, lo, hi, delta, **f_prms)
    return f_1

# ------------------------------------------------------------------------------
# End of inverse()
# ------------------------------------------------------------------------------

# def extrapolate_func(x_range, func, slope):
#     '''
#     Extrapolate NaN values of func as a powerlaw with slope.

#     Parameters
#     ----------
#     x_range : array
#       range for func
#     func : array
#       function where np.nan will be extrapolated

#     Returns
#     -------
#     func : array
#       function with np.nan extrapolated as power law with slope
#     '''
#     def plaw(x, slope):
#         return x**slope

#     idx_cut = np.argmax(np.isnan(func), axis=-1) - 1
#     if idx_cut != 0:
#         x_fit = x_range[...,idx_cut:].astype(float) / x_range[...,idx_cut]
#         func_fit = func[...,idx_cut:].astype(float) / func[...,idx_cut]

#         fit = func[...,idx_cut] * plaw(x_fit, slope)
#         func[...,idx_cut:] = fit

#     return func

# # ------------------------------------------------------------------------------
# # End of extrapolate_func()
# # ------------------------------------------------------------------------------

def extrapolate_plaw(x_range, func, verbose=False):
    '''
    Extrapolate func NaN values as a powerlaw. Works best if power law behaviour
    is already apparent, extrapolates from largest change/bump in func.

    Parameters
    ----------
    x_range : array
      range for func
    func : array
      function where np.nan will be extrapolated

    Returns
    -------
    func : array
      function with np.nan extrapolated as power law
    '''
    def plaw_pos(x, slope):
        return slope * x

    def plaw_neg(x, slope):
        return np.power(x, slope)

    # find largest change in func, will extrapolate from there
    idx_xs = np.argmin(np.diff(func[~np.isnan(func)], axis=-1))
    idx_nan = np.argmax(np.isnan(func), axis=-1) - 1

    # pdb.set_trace()
    if idx_nan != 0:
        x_fit = x_range[~np.isnan(func)]/x_range[idx_xs]
        func_fit = func[~np.isnan(func)]/func[idx_xs]

        x_fit = x_fit[...,idx_xs:]
        func_fit = func_fit[...,idx_xs:]
        if (func_fit < 0).any():
            slope, cov = opt.curve_fit(plaw_neg,
                                       (x_fit).astype(float),
                                       (func_fit).astype(float))
        else:
            slope, cov = opt.curve_fit(plaw_pos,
                                       np.log10(x_fit).astype(float),
                                       np.log10(func_fit).astype(float))

        func[idx_nan:] = func[idx_nan] * \
                         (x_range[idx_nan:]/x_range[idx_nan])**slope
    if verbose: print('Power law slope: %f'%slope)
    return func

# ------------------------------------------------------------------------------
# End of extrapolate_plaw()
# ------------------------------------------------------------------------------

def mean_density_NFW(r, c_x, rho_mean):
    '''
    Return mean density inside r for halo with concentration c_x

    Parameters
    ----------
    r : float
      radius in units of r_200
    c_x : float
      concentration of halo

    Returns
    -------
    rho : float
      mean enclosed density at radius r
    '''
    rho_s = 200./3. * rho_mean * c_x**3/(np.log(1+c_x) - c_x/(1+c_x))
    rho = 3./r**3 * rho_s * (np.log(1+c_x*r) - c_x*r/(1+c_x*r))/c_x**3

    return rho

# ------------------------------------------------------------------------------
# End of mean_density_NFW()
# ------------------------------------------------------------------------------

def rx_to_r200(x, c_200, rho_mean):
    '''
    Returns the radius at mean overdensity x in units of r_200.

    Parameters
    ----------
    x : float
      overdensity with respect to mean
    c_200 : array
      concentration of halo

    Returns
    -------
    rx_200 : array
      r_x in units of r_200

    Examples
    --------
    Conversion factor gives 1 r_500 in units of r_200, multiplying by r_200
    gives the r_500 for the haloes with r_200
    >>> r_500 = r_200 * rx_to_ry(1., 500, c_200)
    '''
    def dens_diff(r, x, c, rho_mean):
        return mean_density_NFW(r, c, rho_mean) - x * rho_mean

    try:
        rx_200 = np.array([opt.brentq(dens_diff, 1e-6, 10, args=(x, c, rho_mean))
                           for c in c_200])
    except TypeError:
        rx_200 = opt.brentq(dens_diff, 1e-6, 10, args=(x, c_200, rho_mean))

    return rx_200

# ------------------------------------------------------------------------------
# End of rx_to_r200()
# ------------------------------------------------------------------------------

def Mx_to_My(M_x, x, y, c_200, rho_mean):
    '''
    Returns M_x, the enclosed halo mass at x overdensity, in units of M_y for a
    halo with concentration c_200.

    Parameters
    ----------
    M_x : array
      masses in units of M_x
    x : float
      overdensity with respect to mean
    y : float
      overdensity with respect to mean
    c_200 : float
      concentration of halo
    rho_mean : float
      mean matter density of the universe

    Returns
    -------
    M_y : array
      masses in units of M_y

    Examples
    --------
    Returns the mass m_200 in units of m_500
    >>> m200_500 = Mx_to_My(m_200, 200, 500, c_x, p.prms.rho_m)
    '''
    r_y = rx_to_r200(y, c_200, rho_mean)
    r_x = rx_to_r200(x, c_200, rho_mean)
    c_y = r_y * c_200
    c_x = r_x * c_200

    M_xy = M_x *  (np.log(1 + c_x) - c_x / (1 + c_x))/ \
           (np.log(1 + c_y) - c_y / (1 + c_y))

    return M_xy

# ------------------------------------------------------------------------------
# End of Mx_to_My()
# ------------------------------------------------------------------------------

def M_to_c200(M_x, M_y, x, y, rho_mean):
    '''
    Given 2 different overdensity masses, determine the concentration of the halo

    Parameters
    ----------
    M_x : array
      masses at overdensity x
    M_y : array
      masses at overdensity y
    x : array
      overdensities x
    y : array
      overdensities y
    rho_mean : float
      mean matter density of the universe

    Returns
    -------
    c_200 : array
      concentrations of haloes
    '''
    def ratio(c_200, r):
        r_y = rx_to_r200(y, c_200, rho_mean)
        r_x = rx_to_r200(x, c_200, rho_mean)
        c_y = r_y * c_200
        c_x = r_x * c_200

        ratio = ((np.log(1 + c_x) - c_x / (1 + c_x))/
                (np.log(1 + c_y) - c_y / (1 + c_y)))

        return ratio - r

    # we have only limited range of values due to c_min & c_max
    max_cut = Mx_to_My(1., x, y, 0.1, rho_mean)
    min_cut = Mx_to_My(1., x, y, 100, rho_mean)

    quotient = M_x / M_y
    in_range = ((quotient <= max_cut) & (quotient >= min_cut))

    try:
        c_200 = np.array([opt.brentq(ratio, 0.1, 100, args=(r)) if in_range[idx]
                          else -1 for idx, r in enumerate(quotient)])
    except TypeError:
        if in_range:
            c_200 = opt.brentq(ratio, 0.1, 100, args=(quotient))
        else:
            c_200 = -1.

    return c_200

# ------------------------------------------------------------------------------
# End of M_to_c200()
# ------------------------------------------------------------------------------

def r_delta(m_delta, delta, rho_mean):
    '''
    Return radius at overdensity delta for m_delta halo

    Parameters
    ----------
    m_delta : array
      halo mass
    delta : float
      overdensity
    rho_mean : float
      density wrt which m_delta is defined

    Returns
    -------
    r_delta : array
      radius at delta for m_delta
    '''
    r_delta = (3 * m_delta / (4 * np.pi * delta * rho_mean))**(1./3)
    return r_delta

# ------------------------------------------------------------------------------
# End of r_delta()
# ------------------------------------------------------------------------------

def m_delta(r_delta, delta, rho_mean):
    '''
    Return mass at overdensity delta for r_delta

    Parameters
    ----------
    r_delta : array
      radius at delta
    delta : float
      overdensity
    rho_mean : float
      density wrt which m_delta is defined

    Returns
    -------
    m_delta : array
      mass at delta for r_delta
    '''
    m_delta = 4 * np.pi/3 * delta * rho_mean * r_delta**3
    return m_delta

# ------------------------------------------------------------------------------
# End of m_delta()
# ------------------------------------------------------------------------------

def bins2center(bins):
    '''
    Return the center position of bins, with bins along axis -1.
    '''
    return 0.5 * (bins[...,1:] + bins[...,:-1])

# ------------------------------------------------------------------------------
# End of bins2center()
# ------------------------------------------------------------------------------
