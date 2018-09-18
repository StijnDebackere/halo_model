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
def m_NFW(r, m_x, r_x, c_x, z_range=0):
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
    c_factor  = np.log((r_s + r) / r_s) - r / (r + r_s)

    mass = prefactor * c_factor

    return mass

# ------------------------------------------------------------------------------
# End of m_NFW()
# ------------------------------------------------------------------------------

def profile_NFW_f(k_range, m_x, r_x, c_x, z_range=0):
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
    r_x : (m,z) array
      array containing r_x to evaluate r_s from r_s = r_x/c_x
    c_x : (m,z) array
      array containing mass-concentration relation
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
    Si, Ci = spec.sici(K)
    # (m,k,z) array
    Si_c, Ci_c = spec.sici((1+c_x) * K)

    gamma_s = Si_c - Si
    gamma_c = Ci_c - Ci

    # (m,k,z) array
    profile_f = prefactor * (np.sin(K) * gamma_s + np.cos(K) * gamma_c -
                             np.sin(c_x*K) / (K * (1+c_x)))

    return profile_f

# ------------------------------------------------------------------------------
# End of profile_NFW_f()
# ------------------------------------------------------------------------------

def profile_beta_ana(r_range, m_x, r_x, rc, beta):
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
      physical core radius of beta profile in as a fraction

    Returns
    -------
    profile : (m,r) array
      array containing beta profile
    '''
    m = m_x.shape[0]

    # analytic enclosed mass inside r_x gives normalization rho_0
    rho_0 = m_x / (4./3 * np.pi * r_x**3 * spec.hyp2f1(3./2, 3 * beta / 2,
                                                       5./2, -(r_x / rc)**2))

    rc = rc.reshape(m,1)
    beta = beta.reshape(m,1)
    r_x = r_x.reshape(m,1)
    m_x = m_x.reshape(m,1)
    rho_0 = rho_0.reshape(m,1)

    profile = rho_0 / (1 + (r_range / rc)**2)**(3*beta/2)

    return profile

# ------------------------------------------------------------------------------
# End of profile_beta_ana()
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
      physical core radius of beta profile in as a fraction

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

    profile = 1. / (1 + (r_range / rc)**2)**(3*beta/2)
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
def m_beta(r, m_x, r_x, rc, beta):
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
    rc : float
      core radius r_c of the profile
    beta : float
      beta slope of the profile
    '''
    norm = (4./3 * np.pi * r_x**3 * spec.hyp2f1(3./2, 3 * beta / 2,
                                                5./2, -(r_x / rc)**2))
    rho_0 = m_x / norm
    m = 4./3 * np.pi * rho_0 * r**3 * spec.hyp2f1(3./2, 3 * beta / 2,
                                             5./2, -(r/rc)**2)

    return m

# ------------------------------------------------------------------------------
# End of m_beta()
# ------------------------------------------------------------------------------

@np.vectorize
def r_where_m_beta(m, m_x, r_x, rc, beta):
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
    rc : float
      core radius r_c of the profile
    beta : float
      beta slope of the profile
    '''
    r = opt.brentq(lambda r, m, m_x, r_x, rc, beta: \
                       m - m_beta(r, m_x, r_x, rc, beta),
                   0, 100, args=(m, m_x, r_x, rc, beta))

    return r

# ----------------------------------------------------------------------
# End of r_where_m_beta()
# ----------------------------------------------------------------------

def profile_plaw(r_range, rho_x, r_x, r_y, gamma):
    '''
    Return a power law profile with density rho_x at r_x that decays with a
    power law slope of gamma upto r_y

        rho(r|m) = rho_x(m) (r_range / r_x)**(-gamma) for r_range < r_y

    Parameters
    ----------
    r_range : (m,r) array
      array containing r_range for each m
    rho_x : (m,) array
      array containing density to match at r_x
    r_x : (m,) array
      radius to match rho_x at, in units of r_range
    r_y : (m,) array
      radius top extend profile to
    gamma : (m,) array
      power law slope of profile

    Returns
    -------
    profile : (m,r) array
      array containing beta profile
    '''
    profile_plaw = np.zeros_like(r_range)
    for idx, r in enumerate(r_range):
        sl = (tools.lte(r, r_y[idx]) & (tools.gte(r, r_x[idx])))
        profile_plaw[idx][sl] = rho_x[idx] * (r[sl]/r_x[idx])**(-gamma[idx])

    return profile_plaw

# ----------------------------------------------------------------------
# End of profile_plaw()
# ----------------------------------------------------------------------

@np.vectorize
def m_plaw(r, rho_x, r_x, r_y, gamma):
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
    r_y : float
      radius to extend profile to
    gamma : float
      power law slope of profile
    '''
    if r < r_x:
        raise ValueError('we need r_x > r')

    prefactor = 4 * np.pi * rho_x * r_x**3
    if r >= r_y:
        if gamma == 3:
            return prefactor * np.log(r_y / r_x)
        else:
            return prefactor * ((r_y/r_x)**(3 - gamma) - 1) / (3 - gamma)
    else:
        if gamma == 3:
            return prefactor * np.log(r / r_x)
        else:
            return prefactor * ((r/r_x)**(3 - gamma) - 1)/ (3 - gamma)

# ----------------------------------------------------------------------
# End of m_plaw()
# ----------------------------------------------------------------------

@np.vectorize
def m_beta_plaw(r, m_x, rho_x, r_x, r_y, gamma):
    '''
    Return the analytic enclosed mass inside r for a beta profile upto
    r_x and a power law outside

    Parameters
    ----------
    r : float
      radius to compute for
    m_x : float
      mass inside r_x
    rho_x : float
      density to match at r_x
    r_x : float
      radius to match rho_x at, in units of r_range
    r_y : float
      radius to extend profile to
    gamma : float
      power law slope of profile
    '''
    return m_x + m_plaw(r=r, rho_x=rho_x, r_x=r_x, r_y=r_y, gamma=gamma)

# ----------------------------------------------------------------------
# End of m_beta_plaw()
# ----------------------------------------------------------------------

@np.vectorize
def r_where_m_beta_plaw(m, m_x, rho_x, r_x, r_y, gamma):
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
    r_y : float
      radius to extend profile to
    gamma : float
      power law slope of profile
    '''
    try:
        r = opt.brentq(lambda r, m_x, rho_x, r_x, r_y, gamma: \
                       m - m_beta_plaw(r, m_x, rho_x, r_x, r_y, gamma),
                       r_x, r_y, args=(m_x, rho_x, r_x, r_y, gamma))
    # in case of ValueError we will have r >> r_y, so might as well be
    # infinite in our case
    except ValueError:
        r = np.inf

    return r

# ----------------------------------------------------------------------
# End of r_where_m_beta_plaw()
# ----------------------------------------------------------------------

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
    profile = m_y * profile / mass

    return profile

# ------------------------------------------------------------------------------
# End of profile_uniform()
# ------------------------------------------------------------------------------

@np.vectorize
def m_uniform(r, m_y, r_x, r_y):
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

# ------------------------------------------------------------------------------
# End of m_uniform()
# ------------------------------------------------------------------------------

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
    profile_f : (m,k)
      array containing profile_f

    '''
    m = m_y.shape[0]
    k = k_range.shape[0]

    m_range = m_y.reshape(m,1)
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
def m_delta(r, m_range):
    '''
    Returns a delta function mass
    '''
    return m_range

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



