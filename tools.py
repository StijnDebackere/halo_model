import time
import inspect

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.optimize as opt

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

# def mean_slices(data, means, bins):
#     '''
#     Return slices for data such that the mean of each slice returns means

#     Parameters
#     ----------
#     data : (n,) array
#       array to slice
#     means : (m,) array
#       means to match
#     bins : (m,2) array
#       maximum allowed bin around mean

#     Returns
#     -------
#     slices : (m,2) array
#       array containing slices for each mean
#     '''
#     data_sorted = np.sort(data, axis=0)

#     # index of elements matching means
#     idx_mean = np.argmin(np.abs(data_sorted.reshape(-1,1) - means.reshape(1,-1)),
#                         axis=0)

#     # find matching bin indeces
#     idx_bin = np.argmin(np.abs(data_sorted.reshape(-1,1,1) - bins.reshape(1,-1,2)),
#                         axis=0)
#     # get minimum distance from bins to mean -> this will be our slice
#     min_dist = np.min(np.abs(idx_med.reshape(-1,1) - idx_bin), axis=1)
#     slices = np.concatenate([(idx_med - min_dist).reshape(-1,1),
#                              (idx_med + min_dist).reshape(-1,1)],
#                             axis=1)

#     for idx, sl in enumerate(slices):
#         print means[idx]
#         print data[sl[0]:sl[1]].mean()
#         print '---------'

#     return slices

# # ------------------------------------------------------------------------------
# # End of mean_slices()
# # ------------------------------------------------------------------------------

# def running_mean(x, N):
#     cumsum = np.cumsum(np.insert(x, 0, 0))
#     return (cumsum[N:] - cumsum[:-N]) / N

# # ------------------------------------------------------------------------------
# # End of running_mean()
# # ------------------------------------------------------------------------------

# def running_std(x, N):
#     cumsum = np.cumsum(np.insert(x, 0, 0)**2)
#     return np.sqrt((cumsum[N:] - cumsum[:-N]) / N)

# # ------------------------------------------------------------------------------
# # End of running_mean()
# # ------------------------------------------------------------------------------

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
