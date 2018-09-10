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
import pyfftw

import matplotlib.pyplot as plt

import halo.tools as tools

import pdb

# def profile_f(profile, k_range, m_range):
#     '''
#     Return the Fourier transform of profile along the final axis

#     Parameters
#     ----------
#     profile : (m,r) array
#       array containing the density profile for each mass
#     k_range : (k,) array
#       array specifying the k_range
#     m_range : (m,) array

#     '''


def profile_NFW(r_range, m_x, c_x, r_x, z_range=0):
    '''
    Returns an NFW profile for m_range along axis 0 and r_range along
    axis 1 (with optional z_range along axis 2).

    Parameters
    ----------
    r_range : (m,r) array
      array containing r_range for each m
    m_x : (m,) array
      array containing mass inside r_x
    c_x : (m,) or (m,z) array
      array containing mass-concentration relation
    r_x : (m,) or (m,z) array
      array containing r_x to evaluate r_s from r_s = r_x/c_x
    z_range : float/array (default=0)
      redshift range

    Returns
    -------
    profile : (m,r)
      array containing NFW profile
    or (if z is an array)
    profile : (m,r,z)
      array containing NFW profile
    '''
    m = m_x.shape[0]
    r = r_range.shape[-1]
    # want an empty array for scalar z so we can easily construct shapes
    # later on
    z = np.array(np.array(z_range).shape)

    # (m,z) array
    r_s = r_x.reshape([m] + list(z))/c_x.reshape([m] + list(z))
    rho_s = m_x / (4. * np.pi * r_x**3) * c_x**3 / (np.log(1+c_x) - c_x/(1+c_x))

    # (m,r,z) array
    x = r_range.reshape([m,r] + list(z/z)) / r_s.reshape([m,1] + list(z))

    profile = rho_s.reshape([m,1] + list(z)) / (x * (1+x)**2)

    return profile

# ------------------------------------------------------------------------------
# End of profile_NFW()
# ------------------------------------------------------------------------------

@np.vectorize
def m_NFW(r, m_x, r_x, c_x):
    '''
    Calculate the mass of the NFW profile with c_x and r_x, relative to
    Delta rho_meanup to r

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
    c_factor  = np.log((r_s + r) / r_s) - r / (r + r_s)

    mass = prefactor * c_factor

    return mass

# ------------------------------------------------------------------------------
# End of m_NFW()
# ------------------------------------------------------------------------------

def profile_NFW_f(k_range, m_x, c_x, r_x, z_range=0):
    '''
    Returns the analytic Fourier transform of the NFW profile for
    m_range along axis 0 and k_range along axis 1 (and optional
    z_range along axis 2).

    Parameters
    ----------
    k_range : (k,) array
      array containing k_range for profile
    m_x : (m,) array
      array containing mass inside r_x
    c_x : (m,z) array
      array containing mass-concentration relation
    r_x : (m,z) array
      array containing r_x to evaluate r_s from r_s = r_x/c_x
    z_range : float/array (default=0)
      redshift to evaluate mass-concentration relation at

    Returns
    -------
    profile_f : (m,k) array
      array containing Fourier transform of NFW profile
    or (if z is an array)
    profile_f : (m,k,z) array
      array containing Fourier transform of NFW profile

    '''
    m = m_x.shape[0]
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
    m_x = m_x.reshape([m,1] + list(z))
    # (m,1,z) array
    new_shape = [m,1] + list(z/z)

    # prefactor = 4 * np.pi * rho_s * r_s**3 / m_range.reshape(new_shape)
    prefactor = m_x / (np.log(1+c_x) - c_x/(1+c_x))
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

    return profile_f

# ------------------------------------------------------------------------------
# End of profile_NFW_f()
# ------------------------------------------------------------------------------

def profile_beta(r_range, m_x, r_x, rc, beta):
    '''
    Return a beta profile with mass m_x inside r_range <= r_x

        rho[r] =  rho_c[m_range, rc, r_x] / (1 + ((r/r_x)/rc)^2)^(beta / 2)

    rho_c is determined by the mass of the profile.

    Parameters
    ----------
    r_range : (m,r) array
      array containing r_range for each m
    m_x : (m,) array
      array containing masses to match at r_x
    r_x : (m,) array
      x overdensity radius to match m_x at, in units of r_range
    beta : (m,) array
      power law slope of profile
    rc : (m,) array
      core radius of beta profile

    Returns
    -------
    profile : (m,r) array
      array containing beta profile
    '''
    m = m_x.shape[0]
    r = r_range.shape[-1]

    rc = rc.reshape(m,1)
    beta = beta.reshape(m,1)
    r_x = r_x.reshape(m,1)
    m_x = m_x.reshape(m,1)

    profile = 1. / (1 + ((r_range / r500c) / rc)**2)**(-3*beta/2)
    masses = np.array([tools.m_h(profile[idx][tools.lte(r, r_x[idx])],
                                 r[tools.lte(r, r_x[idx])])
                       for idx, r in enumerate(r_range)]).reshape(m,1)
    norm =  m_x/masses
    profile *= norm

    return profile

# ------------------------------------------------------------------------------
# End of profile_beta()
# ------------------------------------------------------------------------------

@np.vectorize
def m_beta(r, mgas_500c, r500c, rc, beta):
    '''
    Return the analytic enclosed mass for the beta profile normalized to
    mgas_500c at r500c

    Parameters
    ----------
    r : float
      radius to compute for
    mgas_500c : float
      gas mass at r500c
    r500c : float
      radius corresponding to halo mass m500c
    rc : float
      core radius r_c of the profile
    beta : float
      beta slope of the profile
    '''
    norm = (4./3 * np.pi * r500c**3 * hyp2f1(3./2, 3 * beta / 2,
                                             5./2, -(r500c / rc)**2))
    rho_0 = mgas_500c / norm
    m = 4./3 * np.pi * rho_0 * r**3 * hyp2f1(3./2, 3 * beta / 2,
                                             5./2, -(r/rc)**2)

    return m

# ------------------------------------------------------------------------------
# End of m_beta()
# ------------------------------------------------------------------------------

def profile_uniform(r_range, m_range, r_x, r_y):
    '''
    Return a uniform, spherically symmetric profile between r_x and r_y

    !!! NOTE -> currently only works for float values of m !!!

    Parameters
    ----------
    r_range : (r,) array
      array containing r_range for each m
    m_range : float
      mass inside the profile
    r_x : float
      inner radius
    r_x : float
      outer radius

    Returns
    -------
    profile : (m,r)
      array containing profile
    '''
    idx_x = np.where(tools.gte(r_range, r_x))[0][0]
    idx_y = np.where(tools.lte(r_range, r_y))[0][-1]

    r_x = r_range[idx_x]
    r_y = r_range[idx_y]

    profile = np.zeros_like(r_range)
    profile[idx_x:idx_y+1] = 1.

    mass = tools.m_h(profile, r_range)
    profile = m_range * profile / mass

    return profile

# ------------------------------------------------------------------------------
# End of profile_uniform()
# ------------------------------------------------------------------------------

@np.vectorize
def m_uniform(r, m_y, r_y, r_x):
    '''
    Return a uniform, spherically symmetric profile between r_x and r_y

    !!! NOTE -> currently only works for float values of m !!!

    Parameters
    ----------
    r : float
      array containing r_range for each m
    m_y : float
      mass inside the r_y
    r_y : float
      outer radius
    r_x : float
      inner radius

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

# ------------------------------------------------------------------------------
# End of m_uniform()
# ------------------------------------------------------------------------------

def profile_uniform_f(k_range, m_range, r_x, r_y):
    '''
    Return the FT of a uniform, spherically symmetric
    profile between r_x and r_y

    !!! NOTE -> currently not adjusted for array values of r_x & r_y

    Parameters
    ----------
    k_range : (k,) array
      array containing the k_range
    m_range : (m,) array
      array containing the mass range
    r_x : (m,) array
      array containing inner radius for each m
    r_x : (m,) array
      array containing outer radius for each m

    Returns
    -------
    profile_f : (m,k)
      array containing profile_f

    '''
    m = m_range.shape[0]
    k = k_range.shape[0]

    m_range = m_range.reshape(m,1)
    k_range = k_range.reshape(1,k)

    kx = k_range * r_x.reshape(m,1)
    ky = k_range * r_y.reshape(m,1)

    prefactor = 3. * m_range / (ky**3 - kx**3)
    profile_f = (np.sin(ky) - np.sin(kx) + kx * np.cos(kx) - ky * np.cos(ky))
    profile_f = prefactor * profile_f

    return profile_f

# ------------------------------------------------------------------------------
# End of profile_uniform_f()
# ------------------------------------------------------------------------------

def profile_delta(r_range, m_range):
    '''
    Returns a delta function profile
    '''
    profile = np.zeros_like(r_range, dtype=float)

    # mass is computed using Simpson integration, we need to take out this factor
    h = np.diff(r_range)
    h0 = h[:,0]
    h1 = h[:,1]
    hsum = h0 + h1
    h0divh1 = h0 / h1

    simps_factor = (6. / (hsum * (2 - 1.0 / h0divh1)) *
                    1./(4 * np.pi * r_range[:,0]**2))

    profile[...,0] = 1.
    profile *= m_range.reshape(-1,1) * simps_factor.reshape(-1,1)

    return profile

# ------------------------------------------------------------------------------
# End of profile_delta()
# ------------------------------------------------------------------------------

@np.vectorize
def m_delta(r, m):
    '''
    Returns a delta function mass
    '''
    return m

# ------------------------------------------------------------------------------
# End of m_delta()
# ------------------------------------------------------------------------------

def profile_delta_f(k_range, m_range):
    '''
    Returns the normalized analytic Fourier transform of the delta profile for
    m_range along axis 0 and k_range along axis 1.

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
    m = m_range.shape[0]
    profile = m_range.reshape(m,1) * np.ones(m_range.shape + k_range.shape, dtype=float)

    return profile

# ------------------------------------------------------------------------------
# End of profile_delta_f()
# ------------------------------------------------------------------------------





