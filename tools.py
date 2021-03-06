import halo.input.interpolators as inp_interp

import numpy as np
import scipy.integrate as intg
from scipy.interpolate import interp1d
from scipy.special import hyp2f1
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


def convert_coords_c_correa(c_correa_interp):
    """
    The interpolator accepts arguments differently from our standard
    c(m) function convention, allow conversion to match c_duffy arguments
    """
    def wrapper(*args):
        m200m, z_range = args
        coords = inp_interp.arrays_to_coords(z_range, np.log10(m200m))
        c200m = c_correa_interp(coords)
        return c200m
    return wrapper


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


def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def Integrate(y, x, axis=-1):
    '''
    Integrate array at sample points x over axis using Simpson
    integration

    Parameters
    ----------
    y : array

    '''
    y_new = np.nan_to_num(y)
    # the last interval is computed with trapz
    result = intg.simps(y=y_new, x=x, axis=axis, even='first')
    # result = np.trapz(y=y_new, x=x, axis=axis)

    return result


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
    idx_med = np.argmin(np.abs(data_sorted.reshape(-1, 1) -
                               medians.reshape(1, -1)),
                        axis=0)
    # find matching bin indeces
    idx_bin = np.argmin(np.abs(data_sorted.reshape(-1, 1, 1) -
                               bins.reshape(1, -1, 2)),
                        axis=0)
    # get minimum distance from bins to median -> this will be our slice
    min_dist = np.min(np.abs(idx_med.reshape(-1, 1) - idx_bin), axis=1)
    slices = np.concatenate([(idx_med - min_dist).reshape(-1, 1),
                             (idx_med + min_dist).reshape(-1, 1)],
                            axis=1)

    return slices


def c_duffy(m_range, z_range=0., sigma_lnc=0.):
    '''
    Concentration mass relation of Duffy+08
    (mean relation for full sample between 0<z<2)
    '''
    m_range, z_range = _check_iterable([m_range, z_range])
    m = m_range.shape[0]
    z = z_range.shape[0]

    m_range = m_range.reshape([1, m])
    z_range = np.array(z_range).reshape([z, 1])

    A = 10.14
    B = -0.081
    C = -1.01

    plaw = A * (m_range/(2e12))**B * (1+z_range)**C * np.e**sigma_lnc
    return plaw


def c_correa(m200m, z_range=0., sigma_lnc=0.):
    '''
    Concentration mass relation of Duffy+08
    (mean relation for full sample between 0<z<2)
    '''
    coords = inp_interp.arrays_to_coords(z_range, np.log10(m200m))
    c200m = inp_interp.c200m_interp()(coords)

    return np.squeeze(c200m)


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
    x = r_range.reshape([m, r] + list(z/z)) / r_s.reshape([m, 1] + list(z))

    profile = rho_s.reshape([m, 1] + list(z)) / (x * (1+x)**2)

    return profile


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

    if r_0 is not None:
        idx_0 = np.argmin(np.abs(r_0 - int_range))
        int_range = int_range[..., idx_0:]
        int_rho = int_rho[..., idx_0:]

    if r_1 is not None:
        idx_1 = np.argmin(np.abs(r_1 - int_range))
        int_range = int_range[..., :idx_1]
        int_rho = int_rho[..., :idx_1]

    return 4 * np.pi * Integrate(int_rho * int_range**2, int_range, axis=axis)


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

    cum_m = np.array([m_h(rho_r[..., :idx],
                          r_range[..., :idx], axis=-1)
                      for idx in np.arange(1, r+1)])

    return cum_m


@np.vectorize
def m_NFW(r, m_x, r_x, c_x, **kwargs):
    '''
    Calculate the mass of the NFW profile with c_x and r_x and m_x at r_x

    Parameters
    ----------
    r : float
      radius to compute mass for
    m_x : float
      mass inside r_x
    r_x : float
      r_x to evaluate r_s from r_s = r_x/c_x
    c_x : float
      concentration of halo

    Returns
    -------
    m_h : float
      mass
    '''
    rho_s = m_x / (4. * np.pi * r_x**3) * c_x**3/(np.log(1+c_x) - c_x/(1+c_x))
    r_s = (r_x / c_x)

    prefactor = 4 * np.pi * rho_s * r_s**3
    c_factor = np.log((r_s + r) / r_s) - r / (r + r_s)

    mass = prefactor * c_factor

    return mass


def m_NFW_delta(r, c_x, r_x, rho_mean, Delta=200):
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
    c_factor = np.log((r_s + r) / r_s) - r / (r + r_s)

    mass = prefactor * c_factor

    return mass


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
        r = r.reshape(-1, 1)
        beta = beta.reshape(1, -1)
        r_c = r_c.reshape(1, -1)
        mgas_500c = mgas_500c.reshape(1, -1)
        r500c = r500c.reshape(1, -1)

    norm = (4./3 * np.pi * r500c**3 * hyp2f1(3./2, 3 * beta / 2,
                                             5./2, -(r500c / r_c)**2))
    rho_0 = mgas_500c / norm
    m = 4./3 * np.pi * rho_0 * r**3 * hyp2f1(3./2, 3 * beta / 2,
                                             5./2, -(r/r_c)**2)

    return m


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
    r = opt.brentq(lambda r, m, beta, r_c, mgas_500c, r500c:
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


@np.vectorize
def rx_from_m(m_f, rho_x, **kwargs):
    '''
    For a given cumulative mass profile m_f that takes the radius as its first
    argument, compute the radius where the mean enclosed density is rho_x

    Parameters
    ----------
    m_f : function
        function to compute cumulative mass profile, radius is its first arg
    rho_x : float
        mean enclosed overdensity
    kwargs : dict
        arguments for m_f

    Returns
    -------
    rx : float
        radius where mean enclosed density is rho_x
    '''
    def diff_mx(r):
        mx = 4. / 3 * np.pi * rho_x * r**3
        m_diff = m_f(r, **kwargs) - mx
        return m_diff

    rx = opt.brentq(diff_mx, 1e-4, 100)
    return rx


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
        log10_rho_int = interp1d(r_range, np.log10(rho))
        # can neglect first part of integral since integrand converges to 0
        # there
        # Need a lot of steps to make answer converge to analytic result!
        r_int = np.logspace(np.log10(r + 1e-5),
                            np.log10(r_range.max() - 1e-5),
                            10000)
        integrand = (2 * 10**(log10_rho_int(r_int)) * r_int /
                     np.sqrt(r_int**2 - r**2))
        sigma[idx] = Integrate(integrand, r_int)

    return sigma


def sigma_from_rho_func(R, rho_func, func_args, rmax=None):
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

    for idx, ri in enumerate(R):
        integrand = lambda r, *func_args: (2 * r *
                                           rho_func(r, *func_args) /
                                           np.sqrt(r**2 - ri**2))

        if rmax is not None:
            sigma[idx] = intg.quad(integrand, ri + 1e-4, rmax,
                                   args=func_args)[0]
        else:
            sigma[idx] = intg.quad(integrand, ri + 1e-4, np.inf,
                                   args=func_args)[0]

    return sigma


def sigma_enc_from_sigma(R, sigma):
    """
    Compute the 3D enclosed mass profile for sigma

    Parameters
    ----------
    R : (R, ) array
        projected radii
    sigma : (R, ) array
        surface mass density profile

    Returns
    -------
    sigma_enc : (R, ) array
        enclosed surface mass
    """
    sigma_enc = np.zeros(R.shape)
    r_min = np.min(R)
    r_max = np.max(R)

    for idx, r in enumerate(R):
        sigma_int = interp1d(R, sigma, fill_value="extrapolate")
        r_int = np.logspace(np.log10(r_min) - 1, np.log10(r), 150)

        integrand = 2 * np.pi * sigma_int(r_int) * r_int
        sigma_enc[idx] = Integrate(integrand, r_int)

    return sigma_enc


def sigma_enc(R, R_range, sigma):
    """
    Compute the 3D enclosed mass profile for sigma

    Parameters
    ----------
    R : float
        projected radius to compute enclosed mass at
    R_range : (R, ) array
        projected radii for the mass density profile
    sigma : (R, ) array
        surface mass density profile

    Returns
    -------
    sigma_enc : float
        enclosed surface mass
    """
    r_min = np.min(R_range)
    r_max = R

    sigma_int = interp1d(R_range, sigma, fill_value="extrapolate")
    r_int = np.logspace(np.log10(r_min) - 1, np.log10(r_max), 150)

    integrand = 2 * np.pi * sigma_int(r_int) * r_int
    sigma_enc = Integrate(integrand, r_int)

    return sigma_enc


def sigma_mean_from_sigma(R, R_sigma, sigma, sigma_crit=1,
                          n_int=500, **kwargs):
    """
    Calculate the tangential sigma_mean from Sigma

    Parameters
    ----------
    R : (Rs, ) array
        projected radii
    R_sigma : (R, ) array
        projected radii at which sigma is defined
    sigma : (R, ) array
        surface mass density profile at R_sigma
    sigma_crit : float
        critical surface mass density

    Returns
    -------
    sigma_mean : (Rs, ) array
        sigma_mean
    """
    sigma_mean = np.zeros(R.shape)

    # sigma should go to a constant for small enough Rr.min()
    # interpolate the Sigma(R) profile
    log10_sigma_int = interp1d(np.log10(R_sigma), np.log10(sigma),
                               bounds_error=False,
                               fill_value=(np.log10(sigma[0]), np.nan))

    for idx, r in enumerate(R):
        # Assumes R_sigma.min() is ~ 0
        log10_R_int = np.linspace(-5,
                                  np.log10(r),
                                  n_int)
        R_int = 10**log10_R_int
        integrand = (2. / r**2 * 10**(log10_sigma_int(log10_R_int))
                     * R_int)
        sigma_mean[idx] = Integrate(integrand, R_int)

    return sigma_mean


def shear_from_sigma(R, R_sigma, sigma, sigma_crit=1, n_int=10000,
                     **kwargs):
    """
    Calculate the tangential shear from Sigma

    Parameters
    ----------
    R : (Rs, ) array
        projected radii
    R_sigma : (R, ) array
        projected radii at which sigma is defined
    sigma : (R, ) array
        surface mass density profile at R_sigma
    sigma_crit : float
        critical surface mass density

    Returns
    -------
    shear : (Rs, ) array
        tangential shear
    """
    sigma_mean = np.zeros(R.shape)
    sigma_R = np.zeros(R.shape)

    # sigma should go to a constant for small enough Rr.min()
    # interpolate the Sigma(R) profile
    log10_sigma_int = interp1d(np.log10(R_sigma), np.log10(sigma),
                               bounds_error=False,
                               fill_value=(np.log10(sigma[0]), np.nan))

    for idx, r in enumerate(R):
        # Assumes R_sigma.min() is ~ 0
        log10_R_int = np.linspace(-5,
                                  np.log10(r),
                                  n_int)
        R_int = 10**log10_R_int
        integrand = (2. / r**2 * 10**(log10_sigma_int(log10_R_int))
                     * R_int)
        sigma_mean[idx] = Integrate(integrand, R_int)
        sigma_R[idx] = 10**(log10_sigma_int(np.log10(r)))

    shear = 1. / sigma_crit * (sigma_mean - sigma_R)
    return shear


def shear_from_sigma2(R, R_sigma, sigma, sigma_crit=np.array([1]),
                      n_int=10000):
    """
    Calculate the tangential shear from Sigma

    Parameters
    ----------
    R : (m, Rs) array
        projected radii
    R_sigma : (m, R) array
        projected radii at which sigma is defined
    sigma : (m, R) array
        surface mass density profile at R_sigma
    sigma_crit : (m,) array or float
        critical surface mass density

    Returns
    -------
    shear : (m, Rs) array
        tangential shear
    """
    sigma_mean = np.zeros(R.shape)
    sigma_R = np.zeros(R.shape)

    # interpolate the Sigma(R) profile
    log10_sigma_int = interp1d(np.log10(R_sigma), np.log10(sigma),
                               axis=-1,
                               fill_value="extrapolate")

    for idx_m, rr in enumerate(R):
        for idx_r, r in enumerate(rr):
            # Assumes R_sigma.min() is ~ 0
            log10_R_int = np.linspace(-5,
                                      np.log10(r),
                                      n_int)
            R_int = 10**log10_R_int
            integrand = (2. / r**2 * 10**(log10_sigma_int(log10_R_int))
                         * R_int)
            sigma_mean[idx_m, idx_r] = Integrate(integrand, R_int)
            sigma_R[idx_m, idx_r] = 10**(log10_sigma_int(np.log10(r)))

    shear = 1. / sigma_crit.reshape(-1, 1) * (sigma_mean - sigma_R)
    return shear


def massdiff_2m2c(m200m, m200c, rhoc, rhom, z, c):
    '''
    Integrate an NFW halo with m200m up to r500c and return the mass difference
    between the integral and m500c
    '''
    # compute radii
    r200c = (m200c / (4./3 * np.pi * 200 * rhoc))**(1./3)
    r200m = mass_to_radius(m200m, 200 * rhom)

    c200m = c(m200m, z).reshape(-1)

    # now get analytic mass
    mass = m_NFW_delta(r200c, c200m, r200m, rhom, Delta=200.)

    return mass - m200c


def massdiff_2m5c(m200m, m500c, rhoc, rhom, z, c):
    '''
    Integrate an NFW halo with m200m up to r500c and return the mass difference
    between the integral and m500c
    '''
    # compute radii
    r500c = (m500c / (4./3 * np.pi * 500 * rhoc))**(1./3)
    r200m = mass_to_radius(m200m, 200 * rhom)

    c200m = c(m200m, z).reshape(-1)

    # now get analytic mass
    mass = m_NFW_delta(r500c, c200m, r200m, rhom, Delta=200.)

    return mass - m500c


def massdiff_5c2m(m500c, m200m, rhoc, rhom, z, c):
    '''
    Integrate an NFW halo with m200m up to r500c and return the mass difference
    between the integral and m500c
    '''
    # compute radii
    r500c = (m500c / (4./3 * np.pi * 500 * rhoc))**(1./3)
    r200m = mass_to_radius(m200m, 200 * rhom)

    # compute concentration
    c200m = c(m200m, z).reshape(-1)

    # now get analytic mass
    mass = m_NFW_delta(r500c, c200m, r200m, rhom, Delta=200)

    return mass - m500c


def massdiff_2c2m(m200c, m200m, rhoc, rhom, z, c):
    '''
    Integrate an NFW halo with m200m up to r200c and return the mass difference
    between the integral and m200c
    '''
    # compute radii
    r200c = (m200c / (4./3 * np.pi * 200 * rhoc))**(1./3)
    r200m = mass_to_radius(m200m, 200 * rhom)

    # compute concentration
    c200m = c(m200m, z).reshape(-1)

    # now get analytic mass
    mass = m_NFW_delta(r200c, c200m, r200m, rhom, Delta=200)

    return mass - m200c


@np.vectorize
def m200c_to_m200m_duffy(m200c, rhoc, rhom, z=0):
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
    m200m = opt.brentq(massdiff_2m2c, m200c, 10. * m200c,
                       args=(m200c, rhoc, rhom, z, c_duffy))

    return m200m


@np.vectorize
def m200c_to_m200m_correa(m200c, rhoc, rhom, z=0):
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
    c_correa = convert_coords_c_correa(inp_interp.c200m_interp())
    # these bounds should be reasonable for m200m < 1e18
    m200m = opt.brentq(massdiff_2m2c, m200c, 10. * m200c,
                       args=(m200c, rhoc, rhom, z, c_correa))

    return m200m


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
    m200m = opt.brentq(massdiff_2m5c, m500c, 10. * m500c,
                       args=(m500c, rhoc, rhom, z, c_duffy))

    return m200m


@np.vectorize
def m500c_to_m200m_correa(m500c, rhoc, rhom, z=0):
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
    c_correa = convert_coords_c_correa(inp_interp.c200m_interp())
    # these bounds should be reasonable for m200m < 1e18
    m200m = opt.brentq(massdiff_2m5c, m500c, 10. * m500c,
                       args=(m500c, rhoc, rhom, z, c_correa))

    return m200m


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
    m500c = opt.brentq(massdiff_5c2m, m200m/10., m200m,
                       args=(m200m, rhoc, rhom, z, c_duffy))

    return m500c


@np.vectorize
def m200m_to_m500c_correa(m200m, rhoc, rhom, z=0):
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
    c_correa = convert_coords_c_correa(inp_interp.c200m_interp())
    # these bounds should be reasonable for m200m < 1e18
    m500c = opt.brentq(massdiff_5c2m, m200m/10., m200m,
                       args=(m200m, rhoc, rhom, z, c_correa))

    return m500c


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
    m200c = opt.brentq(massdiff_2c2m, m200m / 10., m200m,
                       args=(m200m, rhoc, rhom, z, c_duffy))

    return m200c


@np.vectorize
def m200m_to_m200c_correa(m200m, rhoc, rhom, z=0):
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
    c_correa = convert_coords_c_correa(inp_interp.c200m_interp())
    # these bounds should be reasonable for m200m < 1e18
    # 1e19 Msun is ~maximum for c_correa
    m200c = opt.brentq(massdiff_2c2m, m200m / 10., m200m,
                       args=(m200m, rhoc, rhom, z, c_correa))

    return m200c


@np.vectorize
def m200m_dmo_from_mx_dmo(mx, rx, rho_m):
    def solve_mass(m200m_dmo):
        c200m_dmo = inp_interp.c200m_emu(m200m=m200m_dmo,
                                         z=np.array([0])).reshape(-1)
        r200m_dmo = mass_to_radius(m200m_dmo, 200 * rho_m)
        m_enc = m_NFW(r=rx, m_x=m200m_dmo, r_x=r200m_dmo, c_x=c200m_dmo)
        return m_enc - mx

    m200m_dmo = opt.brentq(solve_mass, 0.5 * mx, 20 * mx)
    return m200m_dmo


@np.vectorize
def mx_from_my(my, ry, cy, rho_x):
    """
    Convert a halo with mass m_y and concentration c_y at r_y
    to the mass m_x corresponding to mean overdensity rho_x

    Parameters
    ----------
    my : array
        halo masses
    ry : array
        halo radii
    cy : array
        halo concentrations at r_y
    rho_x : array
        mean overdensity to solve for

    Returns
    -------
    mx : array
        masses of haloes my at mean overdensity rho_x
    """
    def solve_mass(rx):
        mx = radius_to_mass(rx, rho_x)
        m_enc = m_NFW(r=rx, m_x=my, r_x=ry, c_x=cy)

        return m_enc - mx

    rx = opt.brentq(solve_mass, 0.1 * ry, 10 * ry)
    mx = radius_to_mass(rx, rho_x)
    return mx


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
            return x
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


def extrapolate_plaw(x_range, func, verbose=False):
    '''
    Extrapolate func NaN values as a powerlaw. Works best if power law
    behaviour is already apparent, extrapolates from largest change/bump in
    func.

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

        x_fit = x_fit[..., idx_xs:]
        func_fit = func_fit[..., idx_xs:]
        if (func_fit < 0).any():
            slope, cov = opt.curve_fit(plaw_neg,
                                       (x_fit).astype(float),
                                       (func_fit).astype(float))
        else:
            slope, cov = opt.curve_fit(plaw_pos,
                                       np.log10(x_fit).astype(float),
                                       np.log10(func_fit).astype(float))

        func[idx_nan:] = (func[idx_nan] *
                          (x_range[idx_nan:]/x_range[idx_nan])**slope)
    if verbose:
        print('Power law slope: {:.f}'.format(slope))
    return func


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
        rx_200 = np.array([opt.brentq(dens_diff, 1e-6, 10, args=(x, c,
                                                                 rho_mean))
                           for c in c_200])
    except TypeError:
        rx_200 = opt.brentq(dens_diff, 1e-6, 10, args=(x, c_200, rho_mean))

    return rx_200


# @np.vectorize
# def rx_from_y(rho_x, m_y, r_y, c_y):
#     '''
#     Returns r_x at mean overdensity rho_x from an NFW profile with c_y.

#     Parameters
#     ----------
#     rho_x : float
#       overdensity to compute r_x for
#     m_y : float
#       mass inside r_y
#     r_y : float
#       r_y to evaluate r_s from r_s = r_y/c_y
#     c_y : float
#       concentration of halo

#     Returns
#     -------
#     r_x : float
#       radius enclosing mean overdensity rho_x

#     Examples
#     --------
#     Returns physical radius at which mean overdensity is rho_x
#     >>> r_500c = rx_from_y(500 * rho_crit, 1e14, 1.1, 3)
#     >>> r_200m = rx_from_y(200 * rho_mean, 1e14, 1.1, 3)
#     '''
#     def m_diff(r, my, ry, cy):
#         return (dp.m_NFW(r, m_x=my, r_x=ry, c_x=cy) -
#                 4./3 * np.pi * rho_x * r**3)

#     r_x = opt.brentq(m_diff, 1e-6, 10, args=(m_y, r_y, c_y))

#     return r_x


def bins2center(bins):
    '''
    Return the center position of bins, with bins along axis -1.
    '''
    return 0.5 * (bins[..., 1:] + bins[..., :-1])
