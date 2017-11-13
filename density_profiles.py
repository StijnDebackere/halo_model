'''This module implements some different density profiles. All
density profiles should have r_range and m_range as first arguments
for density.Profile to be able to use them.

All Fourier transformed profiles should have k_range and m_range as first
arguments for density.Profile to use them.

The user can add profiles here, following the present profiles as
example.
'''
import numpy as np
import scipy.special
import scipy.optimize as opt

import matplotlib.pyplot as plt

import halo.tools as tools
import halo.sersic as sersic
import halo.model.density as dens

import pdb

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

def profile_NFW_f(k_range, m_range, c_x, r_x, rho_mean, z_range=0, Delta=200.):
    '''
    Returns the analytic Fourier transform of the NFW profile for m_range along
    axis 0 and k_range along axis 1 (and optional z_range along axis 2).

    Parameters
    ----------
    k_range : (k,) array
      array containing k_range for profile
    m_range : (m,) array
      array containing each M for which we compute profile
    c_x : (m,z) array
      array containing mass-concentration relation
    r_x : (m,z) array
      array containing r_x to evaluate r_s from r_s = r_x/c_x
    rho_mean : float
      mean dark matter density
    z_range : float/array (default=0)
      redshift to evaluate mass-concentration relation at
    Delta : float (default=200.)
      critical overdensity for collapse

    Returns
    -------
    profile_f : (m,k) array
      array containing Fourier transform of NFW profile
    or (if z is an array)
    profile_f : (m,k,z) array
      array containing Fourier transform of NFW profile
    '''
    m = m_range.shape[0]
    k = k_range.shape[0]
    # want an empty array for scalar z so we can easily construct shapes
    # later on
    z = np.array(np.array(z_range).shape)

    # (m,z) array
    r_s = r_x.reshape([m] + list(z))/c_x.reshape([m] + list(z))
    # rho_s = Delta/3. * rho_mean * c_x**3/(np.log(1+c_x) - c_x/(1+c_x))

    # reshape to match up with k range
    r_s = r_s.reshape([m,1] + list(z))
    # rho_s = rho_s.reshape([m,1] + list(z))
    c_x = c_x.reshape([m,1] + list(z))
    # (m,1,z) array
    new_shape = [m,1] + list(z/z)

    # prefactor = 4 * np.pi * rho_s * r_s**3 / m_range.reshape(new_shape)
    prefactor = 1./(np.log(1+c_x) - c_x/(1+c_x))
    # (1,k,1) array
    k_range = k_range.reshape([1,k] + list(z/z))
    K = k_range * r_s

    # (1,k,1) array
    Si, Ci = scipy.special.sici(K)
    # (m,k,z) array
    Si_c, Ci_c = scipy.special.sici((1+c_x) * K)

    gamma_s = Si_c - Si
    gamma_c = Ci_c - Ci

    # (m,k,z) array
    profile_f = prefactor * (np.sin(K) * gamma_s + np.cos(K) * gamma_c -
                             np.sin(c_x*K) / (K * (1+c_x)))

    # normalize spectrum so that u[k=0] = 1, otherwise we get a small
    # systematic offset, while we know that theoretically u[k=0] = 1
    # profile_f = profile_f / profile_f[:,0].reshape(m,1)

    return profile_f

# ------------------------------------------------------------------------------
# End of profile_NFW_f()
# ------------------------------------------------------------------------------

def profile_Schaller(r_range, r_s, d_s, r_i, d_i, rho_crit):
    '''
    Generalized matter profile defined by Schaller et al (2014). Outer profile
    is NFW, inner profile is modified NFW characterized by d_i corresponding to
    additional mass due to baryons.

    Parameters
    ----------
    r_range : (m,r) array
      array containing r_range for each M (with r_range[:,-1] = r_vir)
    r_s : float (same units as r_range)
      value for r_s in Schaller profile (~constant)
    d_s : (m,) array
      value for d_s in Schaller profile
    r_i : float (same units as r_range)
      value for r_i in Schaller profile (~constant)
    d_i : (m,) array
      value for d_i in Schaller profile
    m_i : (m,) array
      value for m_i in Schaller profile
    rho_crit : float
      critical density of universe
    Returns
    -------
    profile : (m,r)
      array containing profile
    '''
    m = r_range.shape[0]
    r = r_range.shape[1]

    d_s = d_s.reshape(m,1)
    r_s = r_s.reshape(m,1)
    d_i = d_i.reshape(m,1)
    r_i = r_i.reshape(m,1)

    rs = r_range/r_s
    rho_NFW = rho_crit * d_s * 1/(rs * (1+rs)**2)

    ri = r_range/r_i
    rho_bar = rho_crit * d_i * 1/(ri * (1 + ri**2))

    return rho_NFW + rho_bar

# ------------------------------------------------------------------------------
# End of profile_Schaller()
# ------------------------------------------------------------------------------

def profile_gNFW(r_range, c_x, alpha, r_x, m_s):
    '''
    Returns a gNFW profile for m_range along axis 0 and r_range along
    axis 1.

        rho[r] = delta_c * rho_crit * (r/r_s)^-alpha * (1 + r/r_s)^(alpha-3)

    Parameters
    ----------
    r_range : (m,r) array
      array containing r_range for each M (with r_range[:,-1] = r_vir)
    m_range : (m,) array
      array containing masses to compute NFW profile for (mass at z=0)
    c_x : (m,)
      array containing mass-concentration relation
    alpha : float
      exponent in the gNFW profile
    m_s : (m,) array
      equivalent halo mass of the stellar component

    Returns
    -------
    profile : (m,r)
      array containing gNFW profile
    '''
    # pdb.set_trace()
    m = r_range.shape[0]
    r = r_range.shape[-1]

    # (m,) array
    r_s = r_x/c_x

    # (m,r) array
    x = r_range.reshape(m,r) / r_s.reshape(m,1)

    profile = 1. / (x**alpha * (1+x)**(3-alpha))
    m_s_prof = 4 * np.pi * tools.Integrate(y=profile * r_range**2,
                                           x=r_range,
                                           axis=1)
    profile *= m_s.reshape(m,1) / m_s_prof.reshape(m,1)

    return profile

# ------------------------------------------------------------------------------
# End of profile_gNFW()
# ------------------------------------------------------------------------------

def profile_plaws(r_range, m_x, r_x, a, b, r_s):
    x = r_range / r_s
    idx_a = (x <= 1.)
    profile = x**(-a)
    profile[~idx_a] = x[~idx_a]**(-b)

    # profile needs to have mass m_x at r_x
    x_idx = np.argmin(np.abs(r_range - r_x))
    norm =  m_x/tools.m_h(profile[:x_idx+1], r_range[:x_idx+1])

    profile *= norm
    return profile

def fit_profile_plaws(r_range, m_x, r_x, profile, err=None):
    '''
    Fit a gNFW profile to profile, optimize fit for alpha and r_s

    Parameters
    ----------
    r_range : array
      radius corresponding to profile density
    m_x : array
      mass enclosed at x overdensity
    r_x : array
      x overdensity radius of halo
    profile : array
      data to fit
    err : array
      error on data

    Returns
    -------
    fit_prms : (m,2) array
      (beta, r_c) for each fit
    fit : array
      gnfw function fit to profile
    '''
    popt, pcov = opt.curve_fit(lambda r_range, a, b, r_s: \
                               profile_plaws(r_range, m_x,
                                             r_x, a, b, r_s),
                               r_range, profile,
                               bounds=([0, 0, 0], [5, 5, r_x]),
                               sigma=err)

    fit_prms = {'a': popt[0],
                'b': popt[1],
                'r_s' : popt[2]}
    fit = profile_plaws(r_range, m_x, r_x, **fit_prms)

    return fit_prms, pcov, fit

# ------------------------------------------------------------------------------
# End of fit_profile_gnfw()
# ------------------------------------------------------------------------------

def profile_BCG(r_range, m_range, r_half):
    '''
    Returns the BCG profile for m_range along axis 0 and r_range along
    axis 1.

        rho[r] = m / (4 * pi^1.5 * r_half * r^2) * exp(-(r/r_half)^2)


    Parameters
    ----------
    r_range : (m,r) array
      array containing r_range for each M (with r_range[:,-1] = r_vir)
    m_range : (m,) array
      array containing masses to compute profile for (mass at z=0)
    r_half : (m,) array
      half-light radius of BCG (standard relation r_half = 0.015 r_vir)

    Returns
    -------
    profile : (m,r)
      array containing BCG profile
    '''
    m = r_range.shape[0]
    r = r_range.shape[-1]

    r_half = r_half.reshape(m,1)

    profile = m_range.reshape(m,1) / (4*np.pi**1.5 * r_half * r_range**2) \
              * np.exp(-(r_range/(2 * r_half))**2)
    profile[profile < 1.e-20] = 0.

    return profile

# ------------------------------------------------------------------------------
# End of profile_BCG()
# ------------------------------------------------------------------------------

def profile_ICL(r_range, m_range, r_half, n):
    '''
    Returns the BCG profile for m_range along axis 0 and r_range along
    axis 1.

                 | m / (8 * pi * r_half * r^2)                   for r < 2r_half
        rho[r] = |
                 | m / (32 * pi * r_half^3) * (r/(2r_half))^(-n) for r > 2r_half

    Parameters
    ----------
    r_range : (m,r) array
      array containing r_range for each M (with r_range[:,-1] = r_vir)
    m_range : (m,) array
      array containing masses to compute profile for (mass at z=0)
    r_half : (m,) array
      half-light radius of BCG (standard relation r_half = 0.015 r_vir)
    n : float
      power law slope of ICL component

    Returns
    -------
    profile : (m,r)
      array containing BCG profile
    '''
    m = r_range.shape[0]
    r = r_range.shape[-1]

    r_half = r_half.reshape(m,1)
    m_range = m_range.reshape(m,1)

    profile = m_range / (8*np.pi * r_half * r_range**2)
    profile_2 = m_range / (32 * np.pi * r_half**3) * (r_range/(2*r_half))**(-n)
    # do it the lazy way, easier because of shapes...
    profile[r_range > 2*r_half] = profile_2[r_range > 2*r_half]

    return profile

# ------------------------------------------------------------------------------
# End of profile_ICL()
# ------------------------------------------------------------------------------

def profile_beta(r_range, m_x, r_x, beta, r_c):
    '''
    Returns a beta profile for m_range along axis 0 and r_range along
    axis 1.

        rho[r] =  rho_c[m_range, r_c, r_x] / (1 + (r/r_c)^2)^(beta / 2)

    rho_c is determined by the mass of the profile.

    Parameters
    ----------
    r_range : (m,r) array
      array containing r_range for each M (with r_range[:,-1] = r_vir)
    m_x : (m,) array
      array containing masses to match at r_x
    r_x : (m,) array
      x overdensity radius to match m_x at, in units of r_range
    beta : (m,) array
      power law slope of profile
    r_c : (m,) array
      core radius of beta profile

    Returns
    -------
    profile : (m,r) array
      array containing beta profile
    '''
    m = m_x.shape[0]
    r = r_range.shape[-1]

    r_c = r_c.reshape(m,1)
    beta = beta.reshape(m,1)
    r_x = r_x.reshape(m,1)
    m_x = m_x.reshape(m,1)

    profile = 1. / (1 + (r_range/r_c)**2)**(beta/2)
    x_idx = np.argmin(np.abs(r_range - r_x), axis=-1)
    masses = np.array([tools.m_h(profile[i,:idx+1], r_range[i,:idx+1])
                       for i, idx in enumerate(x_idx)]).reshape(m,1)
    norm =  m_x/masses
    profile *= norm

    return profile

# ------------------------------------------------------------------------------
# End of profile_beta()
# ------------------------------------------------------------------------------

def profile_beta_gamma(r_range, m_x, r_x, beta, gamma, r_c):
    '''
    Return modified beta profile with transition depending on gamma for m_range
    along axis 0 and r_range along axis 1.

        rho[r] =  rho_c[m_range, r_c, r_x] / (1 + (r/r_c)^gamma)^(beta / gamma)

    rho_c is determined by the mass of the profile.

    Parameters
    ----------
    r_range : (m,r) array
      array containing r_range for each M (with r_range[:,-1] = r_vir)
    m_x : (m,) array
      array containing masses to match at r_x
    r_x : (m,) array
      x overdensity radius to match m_x at, in units of r_range
    beta : (m,) array
      power law slope of profile
    gamma : (m,) array
      strength of transition of core to power law
    r_c : (m,) array
      core radius of beta profile

    Returns
    -------
    profile : (m,r) array
      array containing beta profile
    '''
    m = m_x.shape[0]
    r = r_range.shape[-1]

    r_c = r_c.reshape(1,m)
    beta = beta.reshape(1,m)
    gamma = gamma.reshape(1,m)
    r_x = r_x.reshape(1,m)
    m_x = m_x.reshape(1,m)

    profile = (1 + (r_range/r_c)**gamma)**(-beta/gamma)

    x_idx = np.argmin(np.abs(r_range - r_x), axis=-1)
    norm =  m_x/tools.m_h(profile[:,:x_idx+1], r_range[:,:x_idx+1])
    profile *= norm

    return profile

# ------------------------------------------------------------------------------
# End of profile_beta()
# ------------------------------------------------------------------------------

def profile_beta_plaw(r_range, m_x, r_x, beta, gamma, r_c):
    '''
    Return beta profile with power-law behaviour towards center.
    Return modified beta profile with transition depending on gamma for m_range
    along axis 0 and r_range along axis 1.

        rho[r] =  rho_c[m_range, r_c, r_x] / (1 + (r/r_c)^2)^(beta/2) *
                  (r/r_c)**(-gamma)

    rho_c is determined by the mass of the profile.

    Based on Vikhlinin, Markevitch & Murray (2005).

    Parameters
    ----------
    r_range : (m,r) array
      array containing r_range for each M (with r_range[:,-1] = r_vir)
    m_x : (m,) array
      array containing masses to match at r_x
    r_x : (m,) array
      x overdensity radius to match m_x at, in units of r_range
    beta : (m,) array
      power law slope of profile
    gamma : (m,) array
      power law slope of inner profile
    r_c : (m,) array
      core radius of beta profile

    Returns
    -------
    profile : (m,r) array
      array containing beta profile
    '''
    m = m_x.shape[0]
    r = r_range.shape[-1]

    r_c = r_c.reshape(1,m)
    beta = beta.reshape(1,m)
    gamma = gamma.reshape(1,m)
    r_x = r_x.reshape(1,m)
    m_x = m_x.reshape(1,m)

    profile = (1 + (r_range/r_c)**2)**(-beta/2) * (r_range/r_c)**(-gamma)

    x_idx = np.argmin(np.abs(r_range - r_x), axis=-1)
    norm =  m_x/tools.m_h(profile[:,:x_idx+1], r_range[:,:x_idx+1])
    profile *= norm

    return profile

# ------------------------------------------------------------------------------
# End of profile_beta()
# ------------------------------------------------------------------------------

def profile_delta(r_range, m_range):
    '''
    Returns a delta function profile
    '''
    profile = np.zeros_like(r_range, dtype=float)

    profile[:,0] = 1.
    profile *= m_range.reshape(-1,1)

    return profile

# ------------------------------------------------------------------------------
# End of profile_delta()
# ------------------------------------------------------------------------------

def profile_delta_f(k_range, m_range):
    '''
    Returns the analytic Fourier transform of the delta profile for m_range along
    axis 0 and k_range along axis 1 (and optional z_range along axis 2).

    Parameters
    ----------
    k_range : (k,) array
      array containing k_range for profile
    m_range : (m,) array
      array containing each M for which we compute profile

    Returns
    -------
    profile_f : (m,k) array
      array containing Fourier transform of delta profile
    '''
    profile = np.ones(m_range.shape + k_range.shape, dtype=float)
    # profile /= m_range.reshape(-1,1)

    return profile

# ------------------------------------------------------------------------------
# End of profile_delta_f()
# ------------------------------------------------------------------------------

def profile_sersic(r_range, m_range, r_eff, p, q=1):
    '''
    Returns a sersic profile for m_range along axis 0 and r_range along
    axis 1.

        rho[r] = Gamma * nu[r]

    Gamma is determined by the mass of the profile. nu[r] is deprojected
    luminosity density for a sersic profile with sersic index n=p/q as defined
    by Baes & Gentile (2010).

    Parameters
    ----------
    r_range : (m,r) array
      array containing r_range for each m
    m : (m,) array
      mass to compute profile for (mass at z=0)
    r_eff : (m,) array
      effective radius of galaxy
    p : int
      numerator of sersic index n = p/q
    q : int
      denominator of sersic index n = p/q

    Returns
    -------
    profile : (r,) or (m,r) array
      array containing beta profile

    '''
    m = m_range.shape[0]

    lum = sersic.luminosity(p, q)
    rho = np.ones_like(r_range)

    s = r_range / r_eff.reshape(m,1)

    # nu = lum(s[-1])

    # luminosity density nu(r), following Baes & Gentile (2010)
    nu = np.empty_like(s)
    for idx, si in enumerate(s):
        nu[idx] = lum(si)

    for idx, r in enumerate(r_range):
        nu2rho = m_range[idx] / (4*np.pi * tools.Integrate(nu * r**2, r))
        rho[idx] = nu2rho * nu

    return rho

# ------------------------------------------------------------------------------
# End of profile_sersic()
# ------------------------------------------------------------------------------

def profile_plaw(r_range, m_x, r_x, r_eff, a, b):
    r = r_range / r_eff
    profile = r**(-a) * np.exp(-r**b)

    x_idx = np.argmin(np.abs(r_range - r_x))
    norm =  m_x/tools.m_h(profile[:x_idx+1], r_range[:x_idx+1])

    profile *= norm

    return profile

# ------------------------------------------------------------------------------
# End of profile_plaw()
# ------------------------------------------------------------------------------

def fit_profile_plaw(r_range, m_x, r_x, profile, err=None):
    '''
    Fit an ... profile to profile, optimize fit for r_eff and a

    Parameters
    ----------
    r_range : array
      radius corresponding to profile density
    m_x : array
      mass enclosed at x overdensity
    r_x : array
      x overdensity radius of halo
    profile : array
      data to fit
    err : array
      error on data

    Returns
    -------
    fit_prms : (m,2) array
      (r_eff, a) for each fit
    fit : array
      beta function fit to profile
    '''
    popt, pcov = opt.curve_fit(lambda r_range, r_eff, a, b: \
                               profile_plaw(r_range, m_x,
                                               r_x, r_eff, a, b),
                               r_range, profile,
                               bounds=([0, 0, 0], [r_x, 3, 5]),
                               sigma=err)

    fit_prms = {'r_eff': popt[0],
                'a': popt[1],
                'b': popt[2]}
    fit = profile_plaw(r_range, m_x, r_x, **fit_prms)

    return fit_prms, pcov, fit

# ------------------------------------------------------------------------------
# End of fit_profile_plaw()
# ------------------------------------------------------------------------------

def profile_b(r_range, m_x, r_x, beta, r_c):
    profile = 1 / (1 + (r_range/r_c)**2)**(beta/2)
    x_idx = np.argmin(np.abs(r_range - r_x))
    norm =  m_x/tools.m_h(profile[:x_idx+1], r_range[:x_idx+1])
    profile *= norm

    return profile

def fit_profile_beta(r_range, m_x, r_x, profile, err=None):
    '''
    Fit a beta profile to profile, optimize fit for beta and r_c

    Parameters
    ----------
    r_range : array
      radius corresponding to profile density
    m_x : array
      mass enclosed at x overdensity
    r_x : array
      x overdensity radius of halo
    profile : array
      data to fit
    err : array
      error on data

    Returns
    -------
    fit_prms : (m,2) array
      (beta, r_c) for each fit
    fit : array
      beta function fit to profile
    '''
    popt, pcov = opt.curve_fit(lambda r_range, beta, r_c: \
                               profile_b(r_range, m_x,
                                         r_x, beta,r_c),
                               r_range, profile,
                               bounds=([0, 0], [5, r_x]),
                               sigma=err)

    fit_prms = {'beta': popt[0],
                'r_c' : popt[1]}
    fit = profile_b(r_range, m_x, r_x, **fit_prms)

    return fit_prms, pcov, fit

# ------------------------------------------------------------------------------
# End of fit_profile_beta()
# ------------------------------------------------------------------------------

def profile_b_plaw(r_range, m_x, r_x, beta, gamma, r_c):
    profile = (1 + (r_range/r_c)**2)**(-beta/2) * (r_range/r_c)**(-gamma)
    x_idx = np.argmin(np.abs(r_range - r_x))
    norm =  m_x / tools.m_h(profile[:x_idx+1], r_range[:x_idx+1])
    profile *= norm

    return profile

def fit_profile_beta_plaw(r_range, m_x, r_x, profile, err=None):
    '''
    Fit a beta profile to profile, optimize fit for beta and r_c

    Parameters
    ----------
    r_range : array
      radius corresponding to profile density
    r_x : float
      x overdensity radius
    profile : array
      data to fit
    err : array
      error on data

    Returns
    -------
    fit_prms : (m,3) array
      (beta, gamma, r_c) for each fit
    fit : array
      beta function fit to profile
    '''
    popt, pcov = opt.curve_fit(lambda r_range, beta, gamma, r_c: \
                               profile_b_plaw(r_range, m_x, r_x,
                                              beta, gamma, r_c),
                               r_range, profile,
                               bounds=([0, 0, 0],[100, 100, r_x]),
                               sigma=err)
    # popt, pcov = opt.curve_fit(lambda r_range, beta, gamma: \
    #                            profile_b_plaw(r_range, m_x, r_x,
    #                                              beta, gamma, r_x),
    #                            r_range, profile,
    #                            bounds=([0, 0],[100, 100]),
    #                            sigma=err)

    fit_prms = {'beta' : popt[0],
                'gamma': popt[1],
                'r_c'  : popt[2]}
    # fit_prms = {'beta' : popt[0],
    #             'gamma': popt[1],
    #             'r_c'  : r_x}

    fit = profile_b_plaw(r_range, m_x, r_x, **fit_prms)

    return fit_prms, pcov, fit

# ------------------------------------------------------------------------------
# End of fit_profile_beta_plaw()
# ------------------------------------------------------------------------------

def profile_b_gamma(r_range, m_x, r_x, beta, gamma, r_c):
    profile = (1 + (r_range/r_c)**gamma)**(-beta/gamma)
    x_idx = np.argmin(np.abs(r_range - r_x))
    norm =  m_x/tools.m_h(profile[:x_idx+1], r_range[:x_idx+1])
    profile *= norm

    return profile

def fit_profile_beta_gamma(r_range, m_x, r_x, profile, err=None):
    '''
    Fit a beta profile to profile, optimize fit for beta and r_c

    Parameters
    ----------
    r_range : array
      radius corresponding to profile density
    r_x : float
      x overdensity radius
    profile : array
      data to fit
    err : array
      error on data

    Returns
    -------
    fit_prms : (m,4) array
      (norm, beta, gamma, r_c) for each fit
    fit : array
      beta function fit to profile
    '''
    popt, pcov = opt.curve_fit(lambda r_range, beta, gamma, r_c: \
                               profile_b_gamma(r_range, m_x, r_x,
                                               beta, gamma, r_c),
                               r_range, profile,
                               bounds=([0, 0, 0], [5, 5, r_x]),
                               sigma=err)

    fit_prms = {'beta' : popt[0],
                'gamma': popt[1],
                'r_c'  : popt[2]}
    fit = profile_b_gamma(r_range, m_x, r_x, **fit_prms)

    return fit_prms, pcov, fit

# ------------------------------------------------------------------------------
# End of fit_profile_beta_gamma()
# ------------------------------------------------------------------------------

def profile_beta_extra(r_range, prof, r_x, a):
    x_idx = np.argmin(np.abs(r_range - r_x.reshape(-1,1)), axis=-1)

    for idx, p in enumerate(prof):
        tail = p[x_idx[idx]] * (r_range[idx,x_idx[idx]:] /
                                r_range[idx, x_idx[idx]])**(-a)
        prof[idx, x_idx[idx]:] = tail

    return prof
