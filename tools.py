import time
import inspect

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.optimize as opt

import halo.parameters as p

import pdb

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
	result = scipy.integrate.simps(y=y_new, x=x, axis=axis, even='first')

	return result

# ------------------------------------------------------------------------------
# End of Integrate()
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
    return (3.*m / (4.*np.pi * mean_dens)) ** (1. / 3.)

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

def find_bounds(f, y):
    x = 1.
    while f(x) < y:
        x = x * 2
    lo = 0 if (x == 1) else x/2.
    return lo, x

def binary_search(f, y, lo, hi, delta):
    while lo <= hi:
        x = (lo + hi) / 2.
        if f(x) < y:
            lo = x + delta
        elif f(x) > y:
            hi = x - delta
        else:
            return x;
    return hi if (f(hi) - y < y - f(lo)) else lo

def inverse(f, delta=1/1024.):
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
    def f_1(y):
        lo, hi = find_bounds(f, y)
        return binary_search(f, y, lo, hi, delta)
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
    if verbose: print 'Power law slope: %f'%slope
    return func

# ------------------------------------------------------------------------------
# End of extrapolate_plaw()
# ------------------------------------------------------------------------------

def mean_density_NFW(r, c_x, rho_mean=p.prms.rho_m):
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

def rx_to_r200(x, c_x, rho_mean=p.prms.rho_m):
    '''
    Returns the radius at overdensity x in units of r_200.

    Parameters
    ----------
    x : float
      overdensity with respect to mean
    c_x : array
      concentration of halo

    Returns
    -------
    rx_200 : array
      r_x in units of r_200

    Examples
    --------
    Conversion factor gives 1 r_500 in units of r_200, multiplying by r_200
    gives the r_500 for the haloes with r_200
    >>> r_500 = r_200 * rx_to_ry(1., 500, c_x)
    '''
    def dens_diff(r, x, c):
        return mean_density_NFW(r, c, rho_mean) - x * rho_mean

    rx_200 = np.array([opt.brentq(dens_diff, 1e-6, 10, args=(x, c)) for c in c_x])

    return rx_200

# ------------------------------------------------------------------------------
# End of rx_to_r200()
# ------------------------------------------------------------------------------

def Mx_to_My(M_x, x, y, c_200):
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

    Returns
    -------
    M_y : array
      masses in units of M_y

    Examples
    --------
    Conversion factor gives 1 M_500 in units of M_200, multiplying by m_200
    gives the m_500 for the haloes with m_200
    >>> m_500 = m_200 * Mx_to_My(1., 500, 200, c_x)
    '''
    r_x = rx_to_r200(x, c_200)
    r_y = rx_to_r200(y, c_200)
    c_x = r_x * c_200
    c_y = r_y * c_200

    M_xy = M_x * (np.log(1 + c_x) - c_x / (1 + c_x)) / \
           (np.log(1 + c_y) - c_y / (1 + c_y))

    return M_xy

# ------------------------------------------------------------------------------
# End of Mx_to_My()
# ------------------------------------------------------------------------------
