'''This module implements some different density profiles. All
density profiles should have r_range and m_range as first arguments
for density.Profile to be able to use them.

All Fourier transformed profiles should have k_range and m_range as first
arguments for density.Profile to use them.

The user can add profiles here, following the present profiles as
example.
'''
import numpy as np
import scipy.special as spec
import scipy.optimize as opt
import mpmath as mp
from scipy import integrate as intg
from scipy import interpolate as interp

import halo.tools as tools

import pdb

# def profile_f(profile, k_range, m_range):
#     '''
#     Return the Fourier transform of profile along the final axis

#     Parameters
#     ----------
#     profile : (m, r) array
#       array containing the density profile for each mass
#     k_range : (k,) array
#       array specifying the k_range
#     m_range : (m,) array

#     '''


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

        func[idx_nan:] = func[idx_nan] * \
            (x_range[idx_nan:]/x_range[idx_nan])**slope
    if verbose:
        print('Power law slope: {}'.format(slope))
    return func


def _taylor_expansion_multi(n, r_range, profile, cpus):
    '''
    Computes the Taylor coefficients for the profile expansion for n_range.

        F_n = 1 / (2n+1)! int_r r^(2n+2) * profile[M, r]

    Parameters
    ----------
    n : int
      number of Taylor coefficients to compute
    r_range : (m, r) array
      radius range to integrate over
    profile : array
      density profile with M along axis 0 and r along axis 1
    cpus : int
      number of cpus to use

    Returns
    -------
    taylor_coefs : (m, k, n) array
      array containing Taylor coefficients of Fourier expansion
    '''
    def _taylor_expansion(procn, n_range, r, profile, out_q):
        '''
        Parameters
        ----------
        procn : int
          process id
        n_range : array
          array containing the index of the Taylor coefficients
        r : array
          radius range to integrate over
        profile : array
          density profile with M along axis 0 and r along axis 1
        out_q : queue
          queue to output results
        '''
        # (m, n) array
        F_n = np.empty((profile.shape[0],) + n_range.shape,
                       dtype=np.longdouble)
        r = np.longdouble(r)

        for idx, n in enumerate(n_range):
            prefactor = 1./spec.factorial(2*n+1, exact=True)
            result = prefactor * intg.simps(y=np.power(r, (2.0*n+2)) *
                                            profile,
                                            x=r,
                                            axis=1,
                                            even='first')

            F_n[:, idx] = result

        results = [procn, F_n]
        out_q.put(results)
        return
    # --------------------------------------------------------------------------
    manager = multi.Manager()
    out_q = manager.Queue()

    taylor = np.arange(0, n+1)
    # Split array in number of CPUs
    taylor_split = np.array_split(taylor, cpus)

    # Start the different processes
    procs = []

    for i in range(cpus):
        process = multi.Process(target=_taylor_expansion,
                                args=(i, taylor_split[i],
                                      r_range,
                                      profile,
                                      out_q))
        procs.append(process)
        process.start()

    # Collect all results
    result = []
    for i in range(cpus):
        result.append(out_q.get())

    result.sort()
    taylor_coefs = np.concatenate([item[1] for item in result],
                                  axis=-1)

    # Wait for all worker processes to finish
    for p in procs:
        p.join()

    return taylor_coefs


def profile_f(k_range, r_range, profile, n=84, cpus=4, extrap=True,
              taylor_err=1e-50):
    '''
    Computes the Fourier transform of the density profile, using a Taylor
    expansion of the sin(kr)/(kr) term. We have

        u[M, k] = sum_n (-1)^n F_n[M] k^(2n)

    Returns
    -------
    u : (m, k) array
      Fourier transform of density profile
    '''
    def F_n(r_range, rho_r, n, cpus):
        '''
        Computes the Taylor coefficients in the Fourier expansion:

            F_n[M] = 4 * pi * 1 / (2n+1)! int_r r^(2n+2) * profile[M, r] dr

        Returns
        -------
        F_n : (m, n+1) array
          Taylor coefficients of Fourier expansion
        '''
        # Prefactor only changes along axis 0 (Mass)
        prefactor = (4.0 * np.pi)

        # F_n is (m, n+1) array
        F_n = _taylor_expansion_multi(n=n, r_range=r_range,
                                      profile=rho_r,
                                      cpus=cpus)
        F_n *= prefactor

        return F_n
    # --------------------------------------------------------------------------
    # define shapes for readability
    n_s = n
    m_s = r_range.shape[0]
    k_s = k_range.shape[0]

    Fn = F_n(r_range, rho_r, n, cpus)
    # need (1, n+1) array to match F_n
    n_arr = np.arange(0, n_s+1, dtype=np.longdouble).reshape(1, n_s+1)
    # -> (m, n) array
    c_n = np.power(-1, n_arr) * Fn

    # need (k, n+1) array for exponent
    k_n = np.power(np.tile(np.longdouble(k_range).reshape(k_s, 1),
                           (1, n_s+1)),
                   (2 * n_arr))

    # need to match n terms and sum over them
    # result is (k, m) array -> transpose
    T_n = c_n.reshape(1, m_s, n_s+1) * k_n.reshape(k_s, 1, n_s+1)
    u = np.sum(T_n, axis=-1).T

    # k-values which do not converge anymore will have coefficients
    # that do not converge to zero. Convergence to zero is determined
    # by taylor_err.
    indices = np.argmax((T_n[:, :, -1] > taylor_err), axis=0)
    indices[indices == 0] = k_s
    for idx, idx_max in enumerate(indices):
        u[idx, idx_max:] = np.nan
        # this extrapolation is not really very good...
        if (idx_max != k_s) and extrap:
            u[idx] = extrapolate_plaw(k_range, u[idx])

    # # normalize spectrum so that u[k=0] = 1, otherwise we get a small
    # # systematic offset, while we know that theoretically u[k=0] = 1
    # if (np.abs(u[:, 0]) - 1. > 1.e-2).any():
    #     print('-------------------------------------------------',
    #           '! Density profile mass does not match halo mass !',
    #           '-------------------------------------------------',
    #           sep='\n')

    # nonnil = (u[:, 0] != 0)
    # u[nonnil] = u[nonnil] / u[nonnil, 0].reshape(-1, 1)

    return u


def profile_NFW(r_range, m_x, c_x, r_x, z_range=0):
    '''
    Returns an NFW profile for m_range along axis 0 and r_range along
    axis 1 (with optional z_range along axis 2).

    Parameters
    ----------
    r_range : (m, r) array
      array containing r_range for each m
    m_x : (m,) array
      array containing mass inside r_x
    c_x : (m,) or (m, z) array
      array containing mass-concentration relation
    r_x : (m,) or (m, z) array
      array containing r_x to evaluate r_s from r_s = r_x/c_x
    z_range : float/array (default=0)
      redshift range

    Returns
    -------
    profile : (m, r)
      array containing NFW profile
    or (if z is an array)
    profile : (m, r, z)
      array containing NFW profile
    '''
    m = m_x.shape[0]
    r = r_range.shape[-1]
    # want an empty array for scalar z so we can easily construct shapes
    # later on
    z = np.array(np.array(z_range).shape)

    # (m, z) array
    r_s = r_x.reshape([m] + list(z))/c_x.reshape([m] + list(z))
    rho_s = (m_x / (4. * np.pi * r_x**3) * c_x**3 /
             (np.log(1+c_x) - c_x/(1+c_x)))

    # (m, r, z) array
    x = r_range.reshape([m, r] + list(z/z)) / r_s.reshape([m, 1] + list(z))

    profile = rho_s.reshape([m, 1] + list(z)) / (x * (1+x)**2)

    return profile


@np.vectorize
def sigma_NFW_fit(R, log10_mx, r_s, rho_x):
    """Return the surface mass density profile of an NFW halo with mass
    m_x, radius r_x and concentration c_x

    Parameters
    ----------
    R : array-like
        projected radius [h^-1 Mpc]
    log10_mx : float
        log_10 halo mass inside r_x [h^-1 M_sun]
    r_s : float
        physical scale radius [h^-1 Mpc]
    rho_x : float
        mean overdensity at r_x [h^2 M_sun/Mpc^3]

    Returns
    -------
    sigma_NFW : array
        surface mass density of NFW profile at projected radius R
    """
    m_x = 10**log10_mx
    r_x = tools.mass_to_radius(m_x, rho_x)
    c_x = r_x / r_s
    rho_s = m_x / (4 * np.pi * r_x**3) * c_x**3 / (np.log(1 + c_x) -
                                                   c_x / (1 + c_x))

    prefactor = 2 * r_s * rho_s

    if R == r_s:
        return 1. / 3 * prefactor
    elif R < r_s:
        x = R / r_s
        prefactor *= 1. / (x**2 - 1)
        sigma = prefactor * (1 - 2 / np.sqrt(1 - x**2) *
                             np.arctanh(np.sqrt((1 - x) / (1 + x))))
        return sigma
    else:
        x = R / r_s
        prefactor *= 1. / (x**2 - 1)
        sigma = prefactor * (1 - 2 / np.sqrt(x**2 - 1) *
                             np.arctan(np.sqrt((x - 1) / (x + 1))))
        return sigma


@np.vectorize
def sigma_NFW(R, m_x, r_x, c_x):
    """Return the surface mass density profile of an NFW halo with mass
    m_x, radius r_x and concentration c_x

    Parameters
    ----------
    R : array-like
        projected radius [h^-1 Mpc]
    m_x : (m,) array
      array containing mass inside r_x
    c_x : (m,) or (m, z) array
      array containing mass-concentration relation
    r_x : (m,) or (m, z) array
      array containing r_x to evaluate r_s from r_s = r_x/c_x

    Returns
    -------
    sigma_NFW : array
        surface mass density of NFW profile at projected radius R
    """
    r_s = r_x / c_x
    rho_s = m_x / (4 * np.pi * r_x**3) * c_x**3 / (np.log(1 + c_x) -
                                                   c_x / (1 + c_x))

    prefactor = 2 * r_s * rho_s

    if R == r_s:
        return 1. / 3 * prefactor
    elif R < r_s:
        x = R / r_s
        prefactor *= 1. / (x**2 - 1)
        sigma = prefactor * (1 - 2 / np.sqrt(1 - x**2) *
                             np.arctanh(np.sqrt((1 - x) / (1 + x))))
        return sigma
    else:
        x = R / r_s
        prefactor *= 1. / (x**2 - 1)
        sigma = prefactor * (1 - 2 / np.sqrt(x**2 - 1) *
                             np.arctan(np.sqrt((x - 1) / (x + 1))))
        return sigma


@np.vectorize
def shear_NFW_fit(R, log10_mx, r_s, rho_x, sigma_crit=1):
    """Return the surface mass density profile of an NFW halo with mass
    m_x, radius r_x and concentration c_x

    Parameters
    ----------
    R : array-like
        projected radius [h^-1 Mpc]
    log10_mx : float
        log_10 halo mass inside r_x [h^-1 M_sun]
    r_s : float
        physical scale radius [h^-1 Mpc]
    rho_x : float
        mean overdensity at r_x [h^2 M_sun/Mpc^3]

    Returns
    -------
    shear_NFW : array
        surface mass density of NFW profile at projected radius R
    """
    m_x = 10**log10_mx
    r_x = tools.mass_to_radius(m_x, rho_x)
    c_x = r_x / r_s
    rho_s = m_x / (4 * np.pi * r_x**3) * c_x**3 / (np.log(1 + c_x) -
                                                   c_x / (1 + c_x))

    prefactor = r_s * rho_s / sigma_crit

    if R == r_s:
        return prefactor * (10. / 3 + 4 * np.log(0.5))
    elif R < r_s:
        x = R / r_s
        f_atanh = np.arctanh(np.sqrt((1 - x) / (1 + x)))
        g = ((8 * f_atanh / (x**2 * np.sqrt(1 - x**2)))
             + 4. / x**2 * np.log(0.5 * x)
             - 2. / (x**2 - 1)
             + 4 * f_atanh / ((x**2 - 1) * np.sqrt(1 - x**2)))
        shear = prefactor * g
        return shear
    else:
        x = R / r_s
        f_atan = np.arctan(np.sqrt((x - 1) / (x + 1)))
        g = ((8 * f_atan / (x**2 * np.sqrt(x**2 - 1)))
             + 4. / x**2 * np.log(0.5 * x)
             - 2. / (x**2 - 1)
             + 4 * f_atan / ((x**2 - 1)**(1.5)))
        shear = prefactor * g
        return shear


@np.vectorize
def shear_NFW(R, m_x, r_x, c_x, sigma_crit=1):
    """Return the shear profile of an NFW halo with mass
    m_x, radius r_x and concentration c_x

    Parameters
    ----------
    R : array-like
        projected radius [h^-1 Mpc]
    m_x : (m,) array
        array containing mass inside r_x
    c_x : (m,) or (m, z) array
        array containing mass-concentration relation
    r_x : (m,) or (m, z) array
        array containing r_x to evaluate r_s from r_s = r_x/c_x
    sigma_crit : (m,) array or (m, z) array or float
        critical surface mass density of the observed systems
    
    Returns
    -------
    shear_NFW : array
        shear of NFW profile at projected radius R
    """
    r_s = r_x / c_x
    rho_s = m_x / (4 * np.pi * r_x**3) * c_x**3 / (np.log(1 + c_x) -
                                                   c_x / (1 + c_x))

    prefactor = r_s * rho_s / sigma_crit

    if R == r_s:
        return prefactor * (10. / 3 + 4 * np.log(0.5))
    elif R < r_s:
        x = R / r_s
        f_atanh = np.arctanh(np.sqrt((1 - x) / (1 + x)))
        g = ((8 * f_atanh / (x**2 * np.sqrt(1 - x**2)))
             + 4. / x**2 * np.log(0.5 * x)
             - 2. / (x**2 - 1)
             + 4 * f_atanh / ((x**2 - 1) * np.sqrt(1 - x**2)))
        shear = prefactor * g
        return shear
    else:
        x = R / r_s
        f_atan = np.arctan(np.sqrt((x - 1) / (x + 1)))
        g = ((8 * f_atan / (x**2 * np.sqrt(x**2 - 1)))
             + 4. / x**2 * np.log(0.5 * x)
             - 2. / (x**2 - 1)
             + 4 * f_atan / ((x**2 - 1)**(1.5)))
        shear = prefactor * g
        return shear


@np.vectorize
def sigma_mean_NFW(R, m_x, r_x, c_x):
    """Return the mean surface mass density  profile of an NFW halo with mass
    m_x, radius r_x and concentration c_x within R

    Parameters
    ----------
    R : array-like
        projected radius [h^-1 Mpc]
    m_x : (m,) array
        array containing mass inside r_x
    c_x : (m,) or (m, z) array
        array containing mass-concentration relation
    r_x : (m,) or (m, z) array
        array containing r_x to evaluate r_s from r_s = r_x/c_x

    Returns
    -------
    sigma_mean : array
        mean enclosed surface mass NFW profile at projected radius R
    """
    r_s = r_x / c_x
    rho_s = m_x / (4 * np.pi * r_x**3) * c_x**3 / (np.log(1 + c_x) -
                                                   c_x / (1 + c_x))

    prefactor = 4 * r_s * rho_s

    if R == r_s:
        return prefactor * (1. + np.log(0.5))
    elif R < r_s:
        x = R / r_s
        prefactor *= 1. / x**2

        f_atanh = np.arctanh(np.sqrt((1 - x) / (1 + x)))
        s_m = ((2 * f_atanh / np.sqrt(1 - x**2))
               + np.log(0.5 * x))
        sigma_mean = prefactor * s_m
        return sigma_mean
    else:
        x = R / r_s
        prefactor *= 1. / x**2

        f_atan = np.arctan(np.sqrt((x - 1) / (x + 1)))
        s_m = ((2 * f_atan / np.sqrt(x**2 - 1))
               + np.log(0.5 * x))
        sigma_mean = prefactor * s_m
        return sigma_mean


@np.vectorize
def r200m_m_NFW(m_x, r_x, c_x, cosmo, z_range=0):
    '''
    Calculate r200m for the NFW profile with c_x and r_x and m_x at r_x

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
    def mass_diff(r):
        m_r = m_NFW(r, m_x, r_x, c_x)
        m200m = 4. / 3 * np.pi * 200 * cosmo.rho_m * r**3
        return m200m - m_r
    r = opt.brentq(mass_diff, 1e-4, 100)
    return r


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


def profile_NFW_f(k_range, m_h, r_h, c_h, z_range=0):
    '''
    Returns the analytic Fourier transform of the NFW profile for
    m_range along axis 0 and k_range along axis 1 (and optional
    z_range along axis 2).

    Parameters
    ----------
    k_range : (k,) array
      array containing k_range for profile
    m_h : (m,) array
      array containing mass inside r_h
    r_h : (m, z) array
      array containing r_h to evaluate r_s from r_s = r_h/c_h
    c_h : (m, z) array
      array containing mass-concentration relation
    z_range : float/array (default=0)
      redshift to evaluate mass-concentration relation at

    Returns
    -------
    profile_f : (m, k) array
      array containing Fourier transform of NFW profile
    or (if z is an array)
    profile_f : (m, k, z) array
      array containing Fourier transform of NFW profile

    '''
    m = m_h.shape[0]
    k = k_range.shape[0]
    # want an empty array for scalar z so we can easily construct shapes
    # later on
    z = np.array(np.array(z_range).shape)

    # (m, z) array
    r_s = r_h.reshape([m] + list(z))/c_h.reshape([m] + list(z))
    # rho_s = Delta/3. * rho_mean * c_h**3/(np.log(1+c_h) - c_h/(1+c_h))

    # reshape to match up with k range
    r_s = r_s.reshape([m, 1] + list(z))
    # rho_s = rho_s.reshape([m, 1] + list(z))
    c_h = c_h.reshape([m, 1] + list(z))
    m_h = m_h.reshape([m, 1] + list(z))
    # (m, 1, z) array
    new_shape = [m, 1] + list(z/z)

    # prefactor = 4 * np.pi * rho_s * r_s**3 / m_range.reshape(new_shape)
    prefactor = m_h / (np.log(1+c_h) - c_h/(1+c_h))
    # (1, k, 1) array
    k_range = k_range.reshape([1, k] + list(z/z))
    K = k_range * r_s

    # (1, k, 1) array
    Si, Ci = spec.sici(K)
    # (m, k, z) array
    Si_c, Ci_c = spec.sici((1+c_h) * K)

    gamma_s = Si_c - Si
    gamma_c = Ci_c - Ci

    # (m, k, z) array
    profile_f = prefactor * (np.sin(K) * gamma_s + np.cos(K) * gamma_c -
                             np.sin(c_h*K) / (K * (1+c_h)))

    return profile_f


def profile_beta(r_range, m_x, r_x, r_c, beta):
    '''
    Return a beta profile with mass m_x inside r_range <= r_x

        rho[r] =  rho_c[m_range, r_c, r_x] / (1 + ((r/r_x)/r_c)^2)^(beta / 2)

    rho_c is determined by the mass of the profile.

    Parameters
    ----------
    r_range : (m, r) array
      array containing r_range for each m
    m_x : (m,) array
      array containing masses to match at r_x
    r_x : (m,) array
      x overdensity radius to match m_x at, in units of r_range
    beta : (m,) array
      power law slope of profile
    r_c : (m,) array
      physical core radius of beta profile in as a fraction

    Returns
    -------
    profile : (m, r) array
      array containing beta profile
    '''
    m = np.shape(m_x)

    # analytic enclosed mass inside r_x gives normalization rho_0
    rho_0 = m_x / (4./3 * np.pi * r_x**3 * spec.hyp2f1(3./2, 3. * beta / 2,
                                                       5./2, -(r_x / r_c)**2))

    r_c = np.reshape(r_c, (m + (1,)))
    beta = np.reshape(beta, (m + (1,)))
    r_x = np.reshape(r_x, (m + (1,)))
    m_x = np.reshape(m_x, (m + (1,)))
    rho_0 = np.reshape(rho_0, (m + (1,)))
    r_range = np.reshape(r_range, (m + (-1,)))

    profile = rho_0 / (1 + (r_range / r_c)**2)**(3*beta/2)

    return profile


@np.vectorize
def m_beta(r, m_x, r_x, r_c, beta, **kwargs):
    '''
    Return the analytic enclosed mass for the beta profile normalized to
    m_x at r_x

    Parameters
    ----------
    r : float
      radius to compute for
    m_x : float
      gas mass at r500c
    r_x : float
      physical radius corresponding to halo mass m500c
    r_c : float
      physical core radius r_c of the profile
    beta : float
      beta slope of the profile
    '''
    # analytic enclosed mass inside r_x gives normalization rho_0
    rho_0 = m_x / (4./3 * np.pi * r_x**3 * spec.hyp2f1(3./2, 3 * beta / 2,
                                                       5./2, -(r_x / r_c)**2))

    m = 4./3 * np.pi * rho_0 * r**3 * spec.hyp2f1(3./2, 3 * beta / 2,
                                                  5./2, -(r/r_c)**2)

    return m


@np.vectorize
def r_where_m_beta(m, m_x, r_x, r_c, beta):
    '''
    Return the radius where the beta profile mass is m

    Parameters
    ----------
    m : float (m,) array
      masses to get r for
    m_x : float
      gas mass at r500c
    r_x : float
      physical radius corresponding to halo mass m500c
    r_c : float
      physical core radius r_c of the profile
    beta : float
      beta slope of the profile
    '''
    r = opt.brentq(lambda r, m, m_x, r_x, r_c, beta:
                   m - m_beta(r, m_x, r_x, r_c, beta),
                   0, 100, args=(m, m_x, r_x, r_c, beta))

    return r


def sigma_beta(R, m_x, r_x, r_c, beta):
    '''
    Return a beta profile with mass m_x inside r_range <= r_x

    Parameters
    ----------
    r_range : (m, r) array
      array containing r_range for each m
    m_x : (m,) array
      array containing masses to match at r_x
    r_x : (m,) array
      x overdensity radius to match m_x at, in units of r_range
    beta : (m,) array
      power law slope of profile
    r_c : (m,) array
      physical core radius of beta profile in as a fraction

    Returns
    -------
    profile : (m, r) array
      array containing beta profile
    '''
    # m = np.shape(m_x)

    # analytic enclosed mass inside r_x gives normalization rho_0
    rho_0 = m_x / (4./3 * np.pi * r_x**3 * spec.hyp2f1(3./2, 3. * beta / 2,
                                                       5./2, -(r_x / r_c)**2))

    # r_c = np.reshape(r_c, (m + (1,)))
    # beta = np.reshape(beta, (m + (1,)))
    # r_x = np.reshape(r_x, (m + (1,)))
    # m_x = np.reshape(m_x, (m + (1,)))
    # rho_0 = np.reshape(rho_0, (m + (1,)))
    # R = np.reshape(R, (m + (-1,)))

    prefactor = np.pi**0.5 * r_c * rho_0
    sigma = prefactor * ((((R/r_c)**2 + 1)**(0.5 - 3 * beta / 2) *
                          spec.gamma(3 * beta / 2 - 0.5)) /
                         spec.gamma(3 * beta / 2))

    return sigma


@np.vectorize
def shear_beta(R, m_x, r_x, r_c, beta, sigma_crit=1,
               **kwargs):
    '''
    Return a beta profile with mass m_x inside r_range <= r_x

    Parameters
    ----------
    r_range : (m, r) array
      array containing r_range for each m
    m_x : (m,) array
      array containing masses to match at r_x
    r_x : (m,) array
      x overdensity radius to match m_x at, in units of r_range
    beta : (m,) array
      power law slope of profile
    r_c : (m,) array
      physical core radius of beta profile in as a fraction
    sigma_crit : (m,) array or (m, z) array or float
        critical surface mass density of the observed systems

    Returns
    -------
    shear : (m, r) array
        shear for a beta profile
    '''
    # analytic enclosed mass inside r_x gives normalization rho_0
    rho_0 = m_x / (4./3 * np.pi * r_x**3 * spec.hyp2f1(1.5, 1.5 * beta,
                                                       2.5, -(r_x / r_c)**2))

    prefactor = (np.pi**0.5 * r_c * rho_0 / sigma_crit)
    x2 = (R / r_c)**2

    if beta != 1:
        prefactor *= (spec.gamma(1.5 * beta - 0.5) /
                      spec.gamma(1.5 * beta))
        f = (1. / (1.5 - 1.5 * beta) * 1. / x2 *
             ((1 + x2)**(1.5 - 1.5 * beta) - 1) -
             (1 + x2)**(0.5 - 1.5 * beta))

        shear = prefactor * f
        return shear

    else:
        prefactor *= 2. / np.pi**0.5
        f = 1. / x2 * np.log(1 + x2) - 1. / (1 + x2)

        shear = prefactor * f
        return shear


@np.vectorize
def sigma_mean_beta(R, m_x, r_x, r_c, beta, **kwargs):
    '''Return a mean enclosed surface mass density of a beta profile with
    mass m_x inside r_range <= r_x

    Parameters
    ----------
    R : array-like
        projected radius [h^-1 Mpc]
    m_x : (m,) array
      array containing masses to match at r_x
    r_x : (m,) array
      x overdensity radius to match m_x at, in units of r_range
    beta : (m,) array
      power law slope of profile
    r_c : (m,) array
      physical core radius of beta profile in as a fraction

    Returns
    -------
    sigma_mean : (m, r) array
        mean enclosed surface mass density for a beta profile

    '''
    # analytic enclosed mass inside r_x gives normalization rho_0
    rho_0 = m_x / (4./3 * np.pi * r_x**3 * spec.hyp2f1(1.5, 1.5 * beta,
                                                       2.5, -(r_x / r_c)**2))

    x2 = (R / r_c)**2

    if beta != 1:
        prefactor = (2 * np.pi**0.5 * r_c * rho_0 * 1. / x2 *
                     spec.gamma(1.5 * beta - 0.5) /
                     spec.gamma(1.5 * beta))
        f = (1. / (1.5 - 1.5 * beta) * ((1 + x2)**(1.5 - 1.5 * beta) - 1))

        sigma_mean = prefactor * f
        return sigma_mean

    else:
        prefactor = (2 * r_c * rho_0 * 1. / x2)
        f = np.log(1 + x2)

        sigma_mean = prefactor * f
        return sigma_mean


def profile_plaw(r_range, rho_x, r_x, gamma):
    '''
    Return a power law profile with density rho_x at r_x that decays with a
    power law slope of gamma

        rho(r|m) = rho_x(m) (r_range / r_x)**(-gamma)

    Parameters
    ----------
    r_range : (m, r) array
      array containing r_range for each m
    rho_x : (m,) array
      array containing density to match at r_x
    r_x : (m,) array
      radius to match rho_x at, in units of r_range
    gamma : float
      power law slope of profile

    Returns
    -------
    profile : (m, r) array
      array containing beta profile
    '''
    m = r_x.shape[0]

    gamma = gamma.reshape(m, 1)
    r_x = r_x.reshape(m, 1)
    rho_x = rho_x.reshape(m, 1)

    profile = rho_x * (r_range / r_x)**(-gamma)
    profile[(r_range < r_x)] = 0.

    return profile


@np.vectorize
def m_plaw(r, rho_x, r_x, gamma, **kwargs):
    '''
    Return the analytic enclosed mass for the power law profile

    Parameters
    ----------
    r : float
      radius to compute for
    rho_x : float
      density to match at r_x
    r_x : float
      radius to match rho_x at, in units of r_range
    gamma : float
      power law slope of profile
    '''
    if r < r_x:
        return 0.

    else:
        prefactor = 4 * np.pi * rho_x * r_x**3
        if gamma == 3:
            return prefactor * np.log(r / r_x)
        else:
            return prefactor * ((r/r_x)**(3 - gamma) - 1) / (3 - gamma)


def profile_plaw_f(k_range, rho_x, r_x, r_y, gamma):
    '''
    Return the analytic 3D radially symmetric FT of a power law between
    between r_x & r_y

    Parameters
    ----------
    k_range : (k,) array
      array containing r_range for each m
    rho_x : (m,) array
      array containing densities to match at r_x
    r_x : (m,) array
      x overdensity radius to match rho_x at, in units of r_range
    r_y : (m,) array
      y radius, in units of r_range
    gamma : (m,) array
      power law slope of profile

    Returns
    -------
    profile : (m, r) array
      array containing power law profile
    '''
    k_range = k_range.reshape(1, -1)
    rho_x = rho_x.reshape(-1, 1)
    r_x = r_x.reshape(-1, 1)
    r_y = r_y.reshape(-1, 1)
    gamma = gamma.reshape(-1, 1)

    prefactor = 4 * np.pi * rho_x * r_x**3 / (2 * k_range**2)

    ikry = 1j * k_range * r_y
    ikrx = 1j * k_range * r_x
    gm2 = 2 - gamma
    ryxmg = (r_y / r_x)**(-gamma)

    gammainc = np.vectorize(mp.gammainc)
    upper_bound = (-1j * ryxmg * ((-ikry)**gamma * gammainc(gm2, -ikry) -
                                  (ikry)**gamma * gammainc(gm2, ikry)))
    lower_bound = (-1j * ((-ikrx)**gamma * gammainc(gm2, -ikrx) -
                          (ikrx)**gamma * gammainc(gm2, ikrx)))

    return prefactor * (upper_bound - lower_bound)


def profile_beta_plaw(r_range, m_x, r_x, r_c, beta, gamma, rho_x=None):
    '''
    Return a beta profile with mass m_x inside r_range <= r_x

        rho[r] =  rho_c[m_range, r_c, r_x] / (1 + ((r/r_x)/r_c)^2)^(beta / 2)

    and a power law outside

        rho[r] = rho_x (r/r_x)^(-gamma)

    rho_c is determined by the mass of the profile.

    Parameters
    ----------
    r_range : (m, r) array
      array containing r_range for each m
    m_x : (m,) array
      array containing masses to match at r_x
    r_x : (m,) array
      x overdensity radius to match m_x at, in units of r_range
    beta : (m,) array
      power law slope of profile
    r_c : (m,) array
      physical core radius of beta profile
    gamma : (m,) array
      power law index

    Returns
    -------
    profile : (m, r) array
      array containing beta profile
    '''
    m = m_x.shape[0]

    # analytic enclosed mass inside r_x gives normalization rho_0
    rho_0 = m_x / (4./3 * np.pi * r_x**3 * spec.hyp2f1(3./2, 3 * beta / 2,
                                                       5./2, -(r_x / r_c)**2))

    if rho_x is None:
        rho_x = rho_0 / (1 + (r_x / r_c)**2)**(3 * beta / 2)

    r_c = r_c.reshape(m, 1)
    beta = beta.reshape(m, 1)
    r_x = r_x.reshape(m, 1)
    m_x = m_x.reshape(m, 1)
    rho_0 = rho_0.reshape(m, 1)
    rho_x = rho_x.reshape(m, 1)

    profile = np.zeros_like(r_range)
    for idx, r in enumerate(r_range):
        sl_beta = (r <= r_x[idx])
        sl_plaw = (r > r_x[idx])
        profile[idx][sl_beta] = (rho_0[idx] / np.power(1 + (r[sl_beta] /
                                                            r_c[idx])**2,
                                                       3*beta[idx]/2))
        profile[idx][sl_plaw] = (rho_x[idx] * np.power(r[sl_plaw]/r_x[idx],
                                                       -gamma[idx]))

    return profile


@np.vectorize
def sigma_beta_plaw(R, m_x, r_x, r_c, beta, gamma, **kwargs):
    '''Return the surface mass density of a beta profile with mass m_x
    inside r_range <= r_x

        rho[r] =  rho_c[m_range, r_c, r_x] / (1 + ((r/r_x)/r_c)^2)^(beta / 2)

    and a power law outside

        rho[r] = rho_x (r/r_x)^(-gamma)

    rho_c is determined by the mass of the profile.

    Parameters
    ----------
    R : (m, r) array
      array containing R for each m
    m_x : float
      mass inside r_x
    r_x : float
      radius to match rho_x at, in units of r_range
    r_c : float
      physical core radius r_c of the profile
    beta : float
      beta slope of the profile
    gamma : float
      power law slope of profile

    Returns
    -------
    sigma : array
        surface mass density of beta plaw profile

    '''
    # analytic enclosed mass inside r_x gives normalization rho_0
    rho_0 = m_x / (4./3 * np.pi * r_x**3 * spec.hyp2f1(1.5, 1.5 * beta,
                                                       2.5, -(r_x / r_c)**2))
    rho_x = rho_0 / (1 + (r_x / r_c)**2)**(1.5 * beta)

    if (gamma <= 1):
        raise ValueError("for gamma <= 1 the profile diverges.")

    elif gamma == 2:
        if R == 0:
            sigma = (np.pi**0.5 * rho_0 * r_c * (spec.gamma(1.5 * beta - 0.5)
                                                 / spec.gamma(1.5 * beta)) *
                     (1 - (1 + (r_x / r_c)**2)**(0.5 - 1.5 * beta))
                     + 2 * rho_x * r_x)
        elif R <= r_x:
            sigma = (np.pi**0.5 * rho_0 * r_c * (spec.gamma(1.5 * beta - 0.5)
                                                 / spec.gamma(1.5 * beta)) *
                     ((1 + (R / r_c)**2)**(0.5 - 1.5 * beta) -
                      (1 + (r_x / r_c)**2)**(0.5 - 1.5 * beta))
                     + 2 * rho_x * r_x * np.arcsin(R / r_x) / (R / r_x))
        else:
            sigma = np.pi * rho_x * r_x * (R / r_x)**(-1)

    else:
        if R == 0:
            sigma = (np.pi**0.5 * rho_0 * r_c * (spec.gamma(1.5 * beta - 0.5)
                                                 / spec.gamma(1.5 * beta)) *
                     (1 - (1 + (r_x / r_c)**2)**(0.5 - 1.5 * beta))
                     + 2 * rho_x * r_x / (gamma - 1))

        elif R < r_x:
            sigma_beta = (np.pi**0.5 * rho_0 * r_c *
                          (spec.gamma(1.5 * beta - 0.5)
                           / spec.gamma(1.5 * beta)) *
                          ((1 + (R / r_c)**2)**(0.5 - 1.5 * beta) -
                           (1 + (r_x / r_c)**2)**(0.5 - 1.5 * beta)))

            x = R / r_x
            # Mathematica solution for rho_gamma integral
            sigma_gamma = ((-1j * x)**(1-gamma) * mp.gamma(1 - 0.5*gamma)
                           * mp.gamma(0.5 * gamma - 0.5) /
                           (2 * mp.pi**0.5) +
                           1j * mp.hyp2f1(0.5, 1 - 0.5 * gamma,
                                          2 - 0.5 * gamma, x**(-2))
                           / (x * (gamma - 2))) * 2 * rho_x * r_x
            # sigma_gamma should be a real number for all values
            # 1 < gamma <= 3
            # the imaginary part is usually ~1e-15 due to machine error
            sigma = sigma_beta + float(sigma_gamma.real)

        else:
            sigma = (np.pi**0.5 * rho_x * r_x * (spec.gamma(0.5 * gamma - 0.5)
                                                 / spec.gamma(0.5 * gamma))
                     * (R / r_x)**(1 - gamma))

    return sigma


@np.vectorize
def shear_beta_plaw(R, m_x, r_x, r_c, beta, gamma, sigma_crit=1,
                    n_int=1000,
                    **kwargs):
    '''Return the shear of a beta profile with mass m_x inside r_range <=
    r_x

        rho[r] =  rho_c[m_range, r_c, r_x] / (1 + ((r/r_x)/r_c)^2)^(beta / 2)

    and a power law outside

        rho[r] = rho_x (r/r_x)^(-gamma)

    rho_c is determined by the mass of the profile.

    Parameters
    ----------
    R : (m, r) array
      array containing R for each m
    m_x : float
      mass inside r_x
    r_x : float
      radius to match rho_x at, in units of r_range
    r_c : float
      physical core radius r_c of the profile
    beta : float
      beta slope of the profile
    gamma : float
      power law slope of profile
    sigma_crit : (m,) array or (m, z) array or float
        critical surface mass density of the observed systems
    n_int : int
        number of steps to interpolate for integral

    Returns
    -------
    shear : array
        shear of beta plaw profile

    '''
    sigma = sigma_beta_plaw(R=R, m_x=m_x, r_x=r_x, r_c=r_c, beta=beta,
                            gamma=gamma, **kwargs)

    sigma_mean = np.empty_like(sigma)
    for idx_m, Rr in enumerate(R):
        # sigma should go to a constant for small enough Rr.min()
        log10_sigma_int = interp.interp1d(np.log10(Rr), np.log10(sigma[idx_m]),
                                          fill_value=np.log10(sigma[idx_m, 0]))
        for idx_r, r in enumerate(Rr):
            R_int = np.logspace(-5, np.log10(r), n_int)
            log10_R_int = np.linspace(-5, np.log10(r), n_int)

            integrand = 10**log10_sigma_int(log10_R_int) * R_int
            sigma_mean[idx_m, idx_r] = 2. / r**2 * intg.simps(integrand,
                                                              x=R_int)

    shear = 1. / sigma_crit * (sigma_mean - sigma)
    return shear


def sigma_mean_beta_plaw(R, m_x, r_x, r_c, beta, gamma, **kwargs):
    '''Return the surface mass density of a beta profile with mass m_x
    inside r_range <= r_x

        rho[r] =  rho_c[m_range, r_c, r_x] / (1 + ((r/r_x)/r_c)^2)^(beta / 2)

    and a power law outside

        rho[r] = rho_x (r/r_x)^(-gamma)

    rho_c is determined by the mass of the profile.

    Parameters
    ----------
    R : (m, r) array
      array containing R for each m
    m_x : (m, 1) array
      mass inside r_x
    r_x : (m, 1) array
      radius to match rho_x at, in units of r_range
    r_c : (m, 1) array
      physical core radius r_c of the profile
    beta : (m, 1) array
      beta slope of the profile
    gamma : (m, 1) array
      power law slope of profile

    Returns
    -------
    sigma_mean : array
        mean surface mass density of beta plaw profile

    '''
    if "sigma" not in kwargs:
        sigma = sigma_beta_plaw(R=R, m_x=m_x, r_x=r_x, r_c=r_c, beta=beta,
                                gamma=gamma, **kwargs)
    else:
        sigma = kwargs["sigma"]

    sigma_mean = np.empty_like(sigma)
    for idx_m, Rr in enumerate(R):
        # sigma should go to a constant for small enough Rr.min()
        log10_sigma_int = interp.interp1d(np.log10(Rr), np.log10(sigma[idx_m]),
                                          bounds_error=False,
                                          fill_value=(np.log10(sigma[idx_m, 0]),
                                                      np.nan))
        for idx_r, r in enumerate(Rr):
            R_int = np.logspace(-5, np.log10(r), 1000)
            log10_R_int = np.linspace(-5, np.log10(r), 1000)

            integrand = 10**log10_sigma_int(log10_R_int) * R_int
            sigma_mean[idx_m, idx_r] = 2. / r**2 * intg.simps(integrand,
                                                              x=R_int)

    return sigma_mean


@np.vectorize
def m_beta_plaw(r, m_x, r_x, r_c, beta, gamma, rho_x=None, **kwargs):
    '''Return the mean enclosed mass of a beta profile
    with mass m_x inside r_range <= r_x

        rho[r] =  rho_c[m_range, r_c, r_x] / (1 + ((r/r_x)/r_c)^2)^(beta / 2)

    and a power law outside

        rho[r] = rho_x (r/r_x)^(-gamma)

    rho_c is determined by the mass of the profile.

    Parameters
    ----------
    r : float
      radius to compute for
    m_x : float
      mass inside r_x
    r_x : float
      radius to match rho_x at, in units of r_range
    r_c : float
      physical core radius r_c of the profile
    beta : float
      beta slope of the profile
    gamma : float
      power law slope of profile
    rho_x : density at r_x
    '''
    if r <= r_x:
        return m_beta(r=r, m_x=m_x, r_x=r_x, r_c=r_c, beta=beta)
    else:
        if rho_x is None:
            rho_x = profile_beta(np.array([r_x]).reshape(-1, 1),
                                 m_x=np.array([m_x]).reshape(-1, 1),
                                 r_x=np.array([r_x]).reshape(-1, 1),
                                 r_c=np.array([r_c]).reshape(-1, 1),
                                 beta=np.array([beta]).reshape(-1, 1))
            rho_x = rho_x.reshape(-1)
        return (m_x + m_plaw(r=r, rho_x=rho_x, r_x=r_x, gamma=gamma))


@np.vectorize
def r_where_m_beta_plaw(m, m_x, r_x, r_c, beta, gamma, rho_x):
    '''
    Return the radius where the beta profile mass is m

    Parameters
    ----------
    m : float
      masses to get r for
      radius to compute for
    m_x : float
      mass inside r_x
    rho_x : float
      density to match at r_x
    r_x : float
      radius to match rho_x at, in units of r_range
    gamma : float
      power law slope of profile
    '''
    try:
        r = opt.brentq(lambda r, m_x, r_x, r_c, beta, gamma, rho_x:
                       m - m_beta_plaw(r, m_x, r_x, r_c, beta, gamma, rho_x),
                       r_x, 15 * r_x, args=(m_x, r_x, r_c, beta, gamma, rho_x))
    # in case of ValueError we will have r >> r_y, so might as well be
    # infinite in our case
    except ValueError:
        r = np.inf

    return r


def profile_beta_plaw_uni(r_range, m_x, r_x, r_c, beta, r_y, gamma,
                          rho_x=None):
    '''
    Return a beta profile with mass m_x inside r_range <= r_x

        rho[r] =  rho_c[m_range, r_c, r_x] / (1 + ((r/r_x)/r_c)^2)^(beta / 2)

    and a power law outside

        rho[r] = rho_x (r/r_x)^(-gamma)

    rho_c is determined by the mass of the profile.

    Parameters
    ----------
    r_range : (m, r) array
      array containing r_range for each m
    m_x : (m,) array
      array containing masses to match at r_x
    r_x : (m,) array
      x overdensity radius to match m_x at, in units of r_range
    r_c : (m,) array
      physical core radius of beta profile in as a fraction
    beta : (m,) array
      power law slope of profile
    r_y : (m,) array
      radius out to which power law holds
    gamma : (m,) array
      power law index

    Returns
    -------
    profile : (m, r) array
      array containing beta profile
    '''
    m = m_x.shape[0]

    # analytic enclosed mass inside r_x gives normalization rho_0
    rho_0 = m_x / (4./3 * np.pi * r_x**3 * spec.hyp2f1(3./2, 3 * beta / 2,
                                                       5./2, -(r_x / r_c)**2))

    if rho_x is None:
        rho_x = rho_0 / (1 + (r_x / r_c)**2)**(3*beta/2)

    r_c = r_c.reshape(m, 1)
    beta = beta.reshape(m, 1)
    r_x = r_x.reshape(m, 1)
    m_x = m_x.reshape(m, 1)
    r_y = r_y.reshape(m, 1)
    rho_0 = rho_0.reshape(m, 1)
    rho_x = rho_x.reshape(m, 1)

    rho_x = rho_x.reshape(m, 1)
    profile = np.zeros_like(r_range)
    for idx, r in enumerate(r_range):
        # create slices for the different profiles
        sl_beta = (r <= r_x[idx])
        sl_plaw = ((r > r_x[idx]) & (r <= r_y[idx]))
        sl_uni = (r > r_y[idx])
        profile[idx][sl_beta] = (rho_0[idx] / np.power(1 + (r[sl_beta] /
                                                            r_c[idx])**2,
                                                       3*beta[idx]/2))
        profile[idx][sl_plaw] = rho_x[idx] * np.power(r[sl_plaw]/r_x[idx],
                                                      -gamma[idx])
        profile[idx][sl_uni] = rho_x[idx] * np.power(r_y[idx] / r_x[idx],
                                                     -gamma[idx])

    return profile


@np.vectorize
def m_beta_plaw_uni(r, m_x, r_x, r_c, beta, r_y, gamma, rho_x=None, **kwargs):
    '''
    Return the analytic enclosed mass inside r for a beta profile upto
    r_x and a power law outside

    Parameters
    ----------
    r : float
      radius to compute for
    m_x : float
      mass inside r_x
    r_x : float
      radius to match rho_x at, in units of r_range
    r_c : float
      physical core radius r_c of the profile
    beta : float
      beta slope of the profile
    r_y : float
      radius to extend power law to
    gamma : float
      power law slope of profile
    rho_x : density at r_x
    '''
    if r <= r_x:
        return m_beta(r=r, m_x=m_x, r_x=r_x, r_c=r_c, beta=beta)
    else:
        if rho_x is None:
            rho_x = profile_beta(np.array([r_x]).reshape(-1, 1),
                                 m_x=np.array([m_x]).reshape(-1, 1),
                                 r_x=np.array([r_x]).reshape(-1, 1),
                                 r_c=np.array([r_c]).reshape(-1, 1),
                                 beta=np.array([beta]).reshape(-1, 1))
            rho_x = rho_x.reshape(-1)
        if r <= r_y:
            return (m_x + m_plaw(r=r, rho_x=rho_x, r_x=r_x, gamma=gamma))
        else:
            rho_y = rho_x * (r_y / r_x)**(-gamma)
            m_uni = 4./3 * np.pi * rho_y * (r**3 - r_y**3)
            return (m_x + m_plaw(r=r_y, rho_x=rho_x, r_x=r_x, gamma=gamma) +
                    m_uni)


@np.vectorize
def r_where_m_beta_plaw_uni(m, m_x, r_x, r_c, beta, r_y, gamma, rho_x):
    '''
    Return the radius where the profile mass is m

    Parameters
    ----------
    m : float
      masses to get r for
      radius to compute for
    m_x : float
      mass inside r_x
    rho_x : float
      density to match at r_x
    r_x : float
      radius to match rho_x at, in units of r_range
    r_y : float
      radius to extend power law profile to
    gamma : float
      power law slope of profile
    '''
    try:
        r = opt.brentq(lambda r, m_x, r_x, r_c, beta, r_y, gamma, rho_x:
                       m - m_beta_plaw_uni(r=r, m_x=m_x, r_x=r_x, r_c=r_c,
                                           beta=beta, r_y=r_y, gamma=gamma,
                                           rho_x=rho_x),
                       r_x, 15 * r_x, args=(m_x, r_x, r_c, beta, r_y, gamma,
                                            rho_x))
    # in case of ValueError we will have r >> r_y, so might as well be
    # infinite in our case
    except ValueError:
        r = np.inf

    return r


def profile_beta_gamma_plaw(r_range, m_x, r_x, r_c, beta, r_y, gamma,
                            delta=3, rho_x=None):
    '''
    Return a beta profile with mass m_x inside r_range <= r_x

        rho[r] =  rho_c[m_range, r_c, r_x] / (1 + ((r/r_x)/r_c)^2)^(beta / 2)

    and a power-law between r_x and r_y

        rho[r] = rho_x (r/r_x)^(-gamma)

    and a power-law with slope -delta outside r_y.

    Parameters
    ----------
    r_range : (m, r) array
      array containing r_range for each m
    m_x : (m,) array
      array containing masses to match at r_x
    r_x : (m,) array
      x overdensity radius to match m_x at, in units of r_range
    r_c : (m,) array
      physical core radius of beta profile in as a fraction
    beta : (m,) array
      power law slope of profile
    r_y : (m,) array
      radius out to which power law holds
    gamma : (m,) array
      power law index
    delta : (m,) array
      power law index

    Returns
    -------
    profile : (m, r) array
      array containing beta profile
    '''
    m = m_x.shape[0]

    # analytic enclosed mass inside r_x gives normalization rho_0
    rho_0 = m_x / (4./3 * np.pi * r_x**3 * spec.hyp2f1(3./2, 3 * beta / 2,
                                                       5./2, -(r_x / r_c)**2))

    if rho_x is None:
        rho_x = rho_0 / (1 + (r_x / r_c)**2)**(3*beta/2)

    r_c = r_c.reshape(m, 1)
    beta = beta.reshape(m, 1)
    r_x = r_x.reshape(m, 1)
    m_x = m_x.reshape(m, 1)
    r_y = r_y.reshape(m, 1)
    rho_0 = rho_0.reshape(m, 1)
    rho_x = rho_x.reshape(m, 1)

    rho_x = rho_x.reshape(m, 1)
    profile = np.zeros_like(r_range)
    for idx, r in enumerate(r_range):
        # create slices for the different profiles
        sl_beta = (r <= r_x[idx])
        sl_gamma = ((r > r_x[idx]) & (r <= r_y[idx]))
        sl_delta = (r > r_y[idx])
        profile[idx][sl_beta] = (rho_0[idx] / np.power(1 + (r[sl_beta] /
                                                            r_c[idx])**2,
                                                       3*beta[idx]/2))
        profile[idx][sl_gamma] = rho_x[idx] * np.power(r[sl_gamma]/r_x[idx],
                                                       -gamma[idx])
        profile[idx][sl_delta] = (rho_x[idx] * np.power(r_y[idx] / r_x[idx],
                                                        -gamma[idx]) *
                                  np.power(r[sl_delta] / r_y[idx],
                                           -delta[idx]))

    return profile


@np.vectorize
def m_beta_gamma_plaw(r, m_x, r_x, r_c, beta, r_y, gamma, delta=3,
                      rho_x=None, **kwargs):
    '''
    Return the analytic enclosed mass inside r for a beta profile upto
    r_x and a power law outside

    Parameters
    ----------
    r : float
      radius to compute for
    m_x : float
      mass inside r_x
    r_x : float
      radius to match rho_x at, in units of r_range
    r_c : float
      physical core radius r_c of the profile
    beta : float
      beta slope of the profile
    r_y : float
      radius to extend power law to
    gamma : float
      power law slope of profile
    delta : float
      asymptotic power-law slope
    rho_x : density at r_x
    '''
    if r <= r_x:
        return m_beta(r=r, m_x=m_x, r_x=r_x, r_c=r_c, beta=beta)
    else:
        if rho_x is None:
            rho_x = profile_beta(np.array([r_x]).reshape(-1, 1),
                                 m_x=np.array([m_x]).reshape(-1, 1),
                                 r_x=np.array([r_x]).reshape(-1, 1),
                                 r_c=np.array([r_c]).reshape(-1, 1),
                                 beta=np.array([beta]).reshape(-1, 1))
            rho_x = rho_x.reshape(-1)
        if r <= r_y:
            return (m_x + m_plaw(r=r, rho_x=rho_x, r_x=r_x, gamma=gamma))
        else:
            rho_y = rho_x * (r_y / r_x)**(-gamma)
            return (m_x + m_plaw(r=r_y, rho_x=rho_x, r_x=r_x, gamma=gamma) +
                    m_plaw(r=r, rho_x=rho_y, r_x=r_y, gamma=delta))


def profile_uniform(r_range, m_y, r_x, r_y):
    '''
    Return a uniform, spherically symmetric profile between r_x and r_y

    !!! NOTE -> currently only works for float values of m !!!

    Parameters
    ----------
    r_range : (r,) array
      array containing r_range for each m
    m_y : float
      mass inside r_y
    r_x : float
      inner radius
    r_x : float
      outer radius

    Returns
    -------
    profile : (m, r)
      array containing profile
    '''
    idx_x = np.where(tools.gte(r_range, r_x))[0][0]
    idx_y = np.where(tools.lte(r_range, r_y))[0][-1]

    r_x = r_range[idx_x]
    r_y = r_range[idx_y]

    profile = np.zeros_like(r_range)
    profile[idx_x:idx_y+1] = 1.

    mass = tools.m_h(profile, r_range)
    profile = m_y * profile / mass

    return profile


@np.vectorize
def m_uniform(r, m_y, r_x, r_y, **kwargs):
    '''
    Return a uniform, spherically symmetric profile between r_x and r_y

    !!! NOTE -> currently only works for float values of m !!!

    Parameters
    ----------
    r : float
      array containing r_range for each m
    m_y : float
      mass inside the r_y
    r_x : float
      inner radius
    r_y : float
      outer radius

    Returns
    -------
    m_uni : float
      mass in uniform profile for r
    '''
    if r < r_x:
        return 0.
    elif (r >= r_x) and (r <= r_x):
        return m_y * (r**3 - r_x**3) / (r_y**3 - r_x**3)
    else:
        return m_y


def profile_uniform_f(k_range, m_y, r_x, r_y):
    '''
    Return the FT of a uniform, spherically symmetric
    profile between r_x and r_y

    !!! NOTE -> currently not adjusted for array values of r_x & r_y

    Parameters
    ----------
    k_range : (k,) array
      array containing the k_range
    m_y : (m,) array
      array containing mass inside r_y
    r_x : (m,) array
      array containing inner radius for each m
    r_x : (m,) array
      array containing outer radius for each m

    Returns
    -------
    profile_f : (m, k)
      array containing profile_f

    '''
    m = m_y.shape[0]
    k = k_range.shape[0]

    m_range = m_y.reshape(m, 1)
    k_range = k_range.reshape(1, k)

    kx = k_range * r_x.reshape(m, 1)
    ky = k_range * r_y.reshape(m, 1)

    prefactor = 3. * m_range / (ky**3 - kx**3)
    profile_f = (np.sin(ky) - np.sin(kx) + kx * np.cos(kx) - ky * np.cos(ky))
    profile_f = prefactor * profile_f

    return profile_f


def profile_delta(r_range, m_x):
    '''
    Returns a delta function profile
    '''
    profile = np.zeros_like(r_range, dtype=float)

    # mass is computed using Simpson integration, we need to take out this
    # factor
    h = np.diff(r_range)
    h0 = h[:, 0]
    h1 = h[:, 1]
    hsum = h0 + h1
    h0divh1 = h0 / h1

    simps_factor = (6. / (hsum * (2 - 1.0 / h0divh1)) *
                    1./(4 * np.pi * r_range[:, 0]**2))

    profile[..., 0] = 1.
    profile *= m_x.reshape(-1, 1) * simps_factor.reshape(-1, 1)

    return profile


@np.vectorize
def m_delta(r, m_x, **kwargs):
    '''
    Returns a delta function mass
    '''
    return m_x


def profile_delta_f(k_range, m_x):
    '''
    Returns the normalized analytic Fourier transform of the delta profile for
    m_range along axis 0 and k_range along axis 1.

    Parameters
    ----------
    k_range : (k,) array
      array containing k_range for profile
    m_x : (m,) array
      array containing each M for which we compute profile

    Returns
    -------
    profile_f : (m, k) array
      array containing Fourier transform of delta profile
    '''
    m = m_x.shape[0]
    profile = m_x.reshape(m, 1) * np.ones(m_x.shape + k_range.shape,
                                              dtype=float)

    return profile


def r200m_from_m(m_f, cosmo, **kwargs):
    '''
    For a given cumulative mass profile m_f that takes the radius as its first
    argument, compute the radius where the mean enclosed density is 200 rho_m

    Parameters
    ----------
    m_f : function
      function to compute cumulative mass profile, radius is its first arg
    kwargs : dict
      arguments for m_f

    Returns
    -------
    r200m : float
      radius where mean enclosed density is 200 rho_m
    '''
    def diff_m200m(r):
        m200m = 4. / 3 * np.pi * 200 * cosmo.rho_m * r**3
        m_diff = m_f(r, **kwargs) - m200m
        return m_diff

    r200m = opt.brentq(diff_m200m, 0.1, 100)
    return r200m


def r_fb_from_f(f_b, cosmo, **kwargs):
    '''
    For a given cumulative mass profile m_f that takes the radius as its first
    argument, compute the radius where the mean enclosed density is 200 rho_m

    Parameters
    ----------
    m_f : function
      function to compute cumulative mass profile, radius is its first arg
    kwargs : dict
      arguments for m_f

    Returns
    -------
    r200m : float
      radius where mean enclosed density is 200 rho_m
    '''
    def diff_fb(r):
        fb = cosmo.omegab / cosmo.omegam
        f_diff = f_b(r, **kwargs) - fb
        return f_diff

    if 'r500c' in kwargs:
        r500c = kwargs['r500c']

    try:
        r_fb = opt.brentq(diff_fb, 0.5 * r500c, 100 * r500c)
    except ValueError:
        r_fb = np.inf

    return r_fb
