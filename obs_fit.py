import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import sys
import pickle
from copy import deepcopy

# allow import of plot
sys.path.append('~/Documents/Leiden/MR/code')
import plot as pl

import halo.hmf as hmf
import halo.parameters as p
import halo.tools as tools
import halo.density_profiles as profs
import halo.tools as tools
import halo.model.density as dens
import halo.model.power as power
import halo.data.data as d

import pdb

# ------------------------------------------------------------------------------
# Definition of different matter components
# ------------------------------------------------------------------------------

def load_dm_dmo(prms):
    '''
    Dark matter profile with NFW profile and f_dm = 1
    '''
    f_dm = prms.cosmo.omegac / prms.cosmo.omegam
    c200m = tools.c_duffy(prms.m200m).reshape(-1)
    # c200m = prms.c_correa

    dm_kwargs = {'cosmo': prms.cosmo,
                 'r_h': prms.r200m,
                 'r200m_inf': prms.r200m,
                 'm200m_inf': prms.m200m,
                 'm_h': prms.m200m,
                 'profile': profs.profile_NFW,
                 'profile_args': {'m_x': prms.m200m,
                                  'c_x': c200m,
                                  'r_x': prms.r200m},
                 'profile_mass': profs.m_NFW,
                 'profile_f': profs.profile_NFW_f,
                 'profile_f_args': {'m_x': prms.m200m,
                                    'c_x': c200m,
                                    'r_x': prms.r200m}}

    dens_dm = dens.Profile(**dm_kwargs)
    return dens_dm

# ------------------------------------------------------------------------------
# End of load_dm_dmo()
# ------------------------------------------------------------------------------

def load_dm(prms, m200m_obs=p.prms.m200m):
    '''
    Dark matter profile with NFW profile and f_dm = 1 - f_b
    '''
    f_dm = prms.cosmo.omegac / prms.cosmo.omegam
    c200m = tools.c_duffy(m200m_obs).reshape(-1)
    # c200m = prms.c_correa

    dm_kwargs = {'cosmo': prms.cosmo,
                 'r_h': prms.r200m,
                 'r200m_inf': prms.r200m,
                 'm200m_inf': prms.m200m,
                 # only f_dm of halo mass in dm
                 'm_h': prms.m200m * f_dm,
                 'profile': profs.profile_NFW,
                 'profile_args': {'m_x': f_dm * prms.m200m,
                                  'c_x': c200m,
                                  'r_x': prms.r200m},
                 'profile_mass': profs.m_NFW,
                 'profile_f': profs.profile_NFW_f,
                 'profile_f_args': {'m_x': f_dm * prms.m200m,
                                    'c_x': c200m,
                                    'r_x': prms.r200m}}

    dens_dm = dens.Profile(**dm_kwargs)
    return dens_dm

# ------------------------------------------------------------------------------
# End of load_dm()
# ------------------------------------------------------------------------------

def load_dm_dmo_rmax(prms, r_max):
    '''
    Pure dark matter only component with NFW profile and f_dm = 1, the profile
    goes up to 5r500c, but the NFW profile goes to 0 for r > r200m
    '''
    r_range = np.array([np.logspace(prms.r_min, np.log10(rm), prms.r_bins)
                        for i,rm in enumerate(r_max)])
    rx = r_range / prms.r200m.reshape(-1,1)

    # concentration
    c200m = tools.c_duffy(prms.m200m).reshape(-1)
    # c200m = prms.c_correa

    prof_dm = np.zeros_like(rx)
    for idx, prof in enumerate(prof_dm):
        sl = (rx[idx] <= 1.)
        prof_dm[idx][sl] = profs.profile_NFW(r_range[idx][sl].reshape(1,-1),
                                             prms.m200m[idx].reshape(1),
                                             c200m[idx].reshape(1),
                                             prms.r200m[idx].reshape(1)).reshape(-1)

    dm_kwargs = {'cosmo': prms.cosmo,
                 'r_h': r_max,
                 'r200m_inf': prms.r200m,
                 'm200m_inf': prms.m200m,
                 'm_h': prms.m200m,
                 'profile': prof_dm,
                 'profile_args': {'m_x': prms.m200m,
                                  'c_x': c200m,
                                  'r_x': prms.r200m},
                 'profile_mass': profs.m_NFW,
                 'profile_f': profs.profile_NFW_f,
                 'profile_f_args': {'m_x': prms.m200m,
                                    'c_x': c200m,
                                    'r_x': prms.r200m}}

    dens_dm = dens.Profile(**dm_kwargs)
    return dens_dm

# ------------------------------------------------------------------------------
# End of load_dm_dmo_rmax()
# ------------------------------------------------------------------------------

def load_dm_rmax(prms, r_max, m200m_obs=p.prms.m200m):
    '''
    Return NFW profiles with up to r200m and 0 up to 5r500c
    '''
    r_range = np.array([np.logspace(prms.r_min, np.log10(rm), prms.r_bins)
                        for i,rm in enumerate(r_max)])
    rx = r_range / prms.r200m.reshape(-1,1)

    # halo dark matter mass fraction
    f_dm = prms.cosmo.omegac / prms.cosmo.omegam
    c200m = tools.c_duffy(m200m_obs).reshape(-1)
    # c200m = prms.c_correa

    prof_dm = np.zeros_like(rx)
    for idx, prof in enumerate(prof_dm):
        sl = (rx[idx] <= 1.)
        prof_dm[idx][sl] = profs.profile_NFW(r_range[idx][sl].reshape(1,-1),
                                             (f_dm * prms.m200m[idx]).reshape(1),
                                             c200m[idx].reshape(1),
                                             prms.r200m[idx].reshape(1)).reshape(-1)

    dm_kwargs = {'cosmo': prms.cosmo,
                 'r_h': r_max,
                 'r200m_inf': prms.r200m,
                 'm200m_inf': prms.m200m,
                 # only f_dm of halo mass in dm
                 'm_h': f_dm * prms.m200m,
                 'profile': prof_dm,
                 'profile_args': {'m_x': f_dm * prms.m200m,
                                  'c_x': c200m,
                                  'r_x': prms.r200m},
                 'profile_mass': profs.m_NFW,
                 'profile_f': profs.profile_NFW_f,
                 'profile_f_args': {'m_x': f_dm * prms.m200m,
                                    'c_x': c200m,
                                    'r_x': prms.r200m}}

    dens_dm = dens.Profile(**dm_kwargs)
    return dens_dm

# ------------------------------------------------------------------------------
# End of load_dm_rmax()
# ------------------------------------------------------------------------------

def prof_beta(x, sl, rc, beta, m_sl, r500):
    '''
    Return a beta profile with mass m_sl inside x[sl]

    Parameters
    ----------
    x : (r,) array
      halo radius normalized by r500c
    sl : (r,) boolean array
      slice to normalize the mass for
    rc : float
      core radius, in units of r500c
    beta : float
      beta slope
    m_sl : float
      mass to normalize profile to at r500c
    r500 : float [Mpc/h]
      halo r500c in physical units

    Returns
    -------
    profile : (r,) array
      beta profile with m_sl inside sl
    '''
    profile = (1 + (x/rc)**2)**(-3*beta/2)
    mass = tools.m_h(profile[sl], x[sl] * r500)
    profile *= m_sl/mass

    return profile

def load_gas(prms, f_stars, q_f=50, q_rc=50, q_beta=50):
    '''
    Return beta profiles with fgas_500c = f_obs, extrapolated to r200m

    Parameters
    ----------
    f_stars : (m,) array
      stellar fraction for each halo mass
    q_f : float
      percentile for fgas-m500 relation fit
    q_rc : float
      percentile for rc-m500 relation fit
    q_beta : float
      percentile for beta-m500 relation fit
    '''
    # radius in terms of r500c
    r_range = np.array([np.logspace(prms.r_min, np.log10(rm), prms.r_bins)
                        for i,rm in enumerate(prms.r200m)])
    rx = r_range / prms.r500c.reshape(-1,1)

    # ---------------------------------------------------------------- #
    # Our halo model assumes simulation variables with Hubble units    #
    # The observations have assumed h=0.7 and get different h scalings #
    # than the theoretical model                                       #
    # ---------------------------------------------------------------- #
    # FOR THE FITS WE NEED TO CONVERT ALL OF OUR HALO MODEL RESULTS TO
    # h=0.7!!!
    # Since r_c and beta are the same for all halo masses, it doesn't
    # matter here
    rc, beta = d.fit_prms(x=500, q_rc=q_rc, q_beta=q_beta)

    # gas fractions
    # h=0.7 needs to be converted here
    f_prms = d.f_gas_prms(prms.cosmo, q=q_f)
    f_gas500 = d.f_gas(prms.m500c / 0.7, cosmo=prms.cosmo, **f_prms)

    # determine the radius at which the beta profile mass exceeds the
    # baryon fraction
    # h=0.7 needs to be converted here
    r_cut = tools.r_where_m_beta(m=(prms.cosmo.omegab/prms.cosmo.omegam - f_stars) *
                                 prms.m200m,
                                 beta=beta, r_c=rc*prms.r500c, mgas_500c=f_gas500 * prms.m500c,
                                 r500c=prms.r500c)

    # r_cut should always be smaller than r200m
    r_cut = np.concatenate([prms.r200m[r_cut >= prms.r200m],
                            r_cut[r_cut < prms.r200m]])

    prof_gas = np.zeros_like(rx)
    prof_mass = np.zeros_like(r_cut)
    for idx, prof in enumerate(prof_gas):
        sl_500 = (rx[idx] <= 1.)
        sl_fb = (rx[idx] >= r_cut[idx] / prms.r500c[idx])
        # this profile corresponds to the physically measured one
        # assuming h=0.7, but our theoretical rho ~ h^2
        prof_gas[idx] = prof_beta(rx[idx], sl_500, rc, beta,
                                  f_gas500[idx] * prms.m500c[idx] / 0.7,
                                  prms.r500c[idx] / 0.7) * (0.7)**(-2.)
        prof_gas[idx][sl_fb] = 0.
        prof_mass[idx] = profs.m_beta(r=r_cut[idx],
                                      m_x=f_gas500[idx] * prms.m500c[idx],
                                      r_x=prms.r500c[idx], beta=beta,
                                      rc=rc * prms.r500c[idx])

    gas_kwargs = {'cosmo': prms.cosmo,
                  'r_h': prms.r200m,
                  'r200m_inf': prms.r200m,
                  'm200m_inf': prms.m200m,
                  'm_h': tools.m_h(prof_gas, r_range),
                  'profile': prof_gas,
                  'profile_mass': prof_mass,
                  # compute FT in dens
                  'profile_f': None}

    dens_gas = dens.Profile(**gas_kwargs)
    return dens_gas

# ------------------------------------------------------------------------------
# End of load_gas()
# ------------------------------------------------------------------------------

def load_gas_obs(prms, q_f=50, q_rc=50, q_beta=50):
    '''
    Return beta profiles with fgas_500c = f_obs which only reach up until r500c

    This one is to be used with a profile that matches f_b at r200m, since we
    do not correct from m_bar to m_dmo

    Parameters
    ----------
    q_f : float
      percentile for fgas-m500 relation fit
    q_rc : float
      percentile for rc-m500 relation fit
    q_beta : float
      percentile for beta-m500 relation fit
    '''
    # radius in terms of r500c
    r_range = np.array([np.logspace(prms.r_min, np.log10(rm), prms.r_bins)
                        for i,rm in enumerate(prms.r200m)])
    rx = r_range / prms.r500c.reshape(-1,1)

    # ---------------------------------------------------------------- #
    # Our halo model assumes simulation variables with Hubble units    #
    # The observations have assumed h=0.7 and get different h scalings #
    # than the theoretical model                                       #
    # ---------------------------------------------------------------- #
    # FOR THE FITS WE NEED TO CONVERT ALL OF OUR HALO MODEL RESULTS TO
    # h=0.7!!!
    # Since r_c and beta are the same for all halo masses, it doesn't
    # matter here
    rc, beta = d.fit_prms(x=500, q_rc=q_rc, q_beta=q_beta)

    # gas fractions
    # h=0.7 needs to be converted here
    f_prms = d.f_gas_prms(prms.cosmo, q=q_f)
    f_gas500 = d.f_gas(prms.m500c / 0.7, cosmo=prms.cosmo, **f_prms)

    prof_gas = np.zeros_like(rx)
    prof_mass = np.zeros_like(prms.m200m)
    for idx, prof in enumerate(prof_gas):
        sl_500 = (rx[idx] <= 1.)
        # this profile corresponds to the physically measured one
        # assuming h=0.7, but our theoretical rho ~ h^2
        prof_gas[idx][sl_500] = prof_beta(rx[idx][sl_500], sl_500[sl_500],
                                          rc, beta,
                                          f_gas500[idx] * prms.m500c[idx] / 0.7,
                                          prms.r500c[idx] / 0.7) * (0.7)**(-2.)
        prof_mass[idx] = profs.m_beta(r=prms.r500c[idx],
                                      m_x=f_gas500[idx] * prms.m500c[idx],
                                      r_x=prms.r500c[idx], rc=rc * prms.r500c[idx],
                                      beta=beta)

    gas_kwargs = {'cosmo': prms.cosmo,
                  'r_h': prms.r200m,
                  'r200m_inf': prms.r200m,
                  'm200m_inf': prms.m200m,
                  'm_h': tools.m_h(prof_gas, r_range),
                  'profile': prof_gas,
                  'profile_args': {'rc': rc,
                                   'beta': beta},
                  'profile_mass': prof_mass,
                  # compute FT in dens
                  'profile_f': None}

    dens_gas = dens.Profile(**gas_kwargs)
    return dens_gas

# ------------------------------------------------------------------------------
# End of load_gas_obs()
# ------------------------------------------------------------------------------

def load_gas_beta_r500c_rmax(prms, r_max, q_f=50, q_rc=50, q_beta=50):
    '''
    Return beta profiles with fgas_500c = f_obs which only reach up until r500c

    This one is to be used with a profile that matches f_b at r200m, since we
    do not correct from m_bar to m_dmo

    Parameters
    ----------
    q_f : float
      percentile for fgas-m500 relation fit
    q_rc : float
      percentile for rc-m500 relation fit
    q_beta : float
      percentile for beta-m500 relation fit
    '''
    # radius in terms of r500c
    r_range = np.array([np.logspace(prms.r_min, np.log10(rm), prms.r_bins)
                        for i,rm in enumerate(r_max)])
    rx = r_range / prms.r500c.reshape(-1,1)

    # ---------------------------------------------------------------- #
    # Our halo model assumes simulation variables with Hubble units    #
    # The observations have assumed h=0.7 and get different h scalings #
    # than the theoretical model                                       #
    # ---------------------------------------------------------------- #
    # FOR THE FITS WE NEED TO CONVERT ALL OF OUR HALO MODEL RESULTS TO
    # h=0.7!!!
    # Since r_c and beta are the same for all halo masses, it doesn't
    # matter here
    rc, beta = d.fit_prms(x=500, q_rc=q_rc, q_beta=q_beta)

    # gas fractions
    # h=0.7 needs to be converted here
    f_prms = d.f_gas_prms(prms.cosmo, q=q_f)
    f_gas500 = d.f_gas(prms.m500c / 0.7, cosmo=prms.cosmo, **f_prms)

    prof_gas = np.zeros_like(rx)
    prof_mass = np.zeros_like(prms.m200m)
    for idx, prof in enumerate(prof_gas):
        sl_500 = (rx[idx] <= 1.)
        # this profile corresponds to the physically measured one
        # assuming h=0.7, but our theoretical rho ~ h^2
        prof_gas[idx][sl_500] = prof_beta(rx[idx][sl_500], sl_500[sl_500],
                                          rc, beta,
                                          f_gas500[idx] * prms.m500c[idx] / 0.7,
                                          prms.r500c[idx] / 0.7) * (0.7)**(-2.)
        prof_mass[idx] = profs.m_beta(r=prms.r500c[idx],
                                      m_x=f_gas500[idx] * prms.m500c[idx],
                                      r_x=prms.r500c[idx], rc=rc * prms.r500c[idx],
                                      beta=beta)

    gas_kwargs = {'cosmo': prms.cosmo,
                  'r_h': r_max,
                  'r200m_inf': prms.r200m,
                  'm200m_inf': prms.m200m,
                  'm_h': tools.m_h(prof_gas, r_range),
                  'profile': prof_gas,
                  'profile_args': {'rc': rc,
                                   'beta': beta},
                  'profile_mass': prof_mass,
                  # compute FT in dens
                  'profile_f': None}

    dens_gas = dens.Profile(**gas_kwargs)
    return dens_gas

# ------------------------------------------------------------------------------
# End of load_gas_beta_r500c_rmax()
# ------------------------------------------------------------------------------

def load_gas_smooth_r500c_r200m(prms, fgas_200, f_stars):
    '''
    Return uniform profiles with fgas_500c = 0 and fgas_200m = f_b - f_obs

    Parameters
    ----------
    prms : p.Parameters object
      contains relevant model info
    fgas_200 : (m,) array
      missing gas fraction from model to be joined
    '''
    # will need to fill exact r500c values in r_range for uniform profile to match
    r500_in_range = np.zeros_like(prms.m200m)

    # radius in terms of r500c
    r_range = np.array([np.logspace(prms.r_min, np.log10(rm), prms.r_bins)
                        for i,rm in enumerate(prms.r200m)])
    rx = r_range / prms.r500c.reshape(-1,1)

    # gas fractions
    f_b = prms.cosmo.omegab / prms.cosmo.omegam

    # relative position of virial radius
    x200m = prms.r200m / prms.r500c

    prof_gas = np.zeros_like(rx)
    prof_mass = np.zeros_like(prms.m200m)
    for idx, prof in enumerate(prof_gas):
        sl = (rx[idx] >= 1.)
        r500_in_range[idx] = r_range[idx][sl.nonzero()[0][0]]
        prof_gas[idx] = profs.profile_uniform(r_range[idx],
                                              (f_b - fgas_200[idx] - f_stars[idx]) *
                                              prms.m200m[idx],
                                              r500_in_range[idx],
                                              prms.r200m[idx])
        prof_mass[idx] = (f_b - fgas_200[idx] - f_stars[idx]) * prms.m200m[idx]


    prof_gas_f = profs.profile_uniform_f(prms.k_range, ((f_b - fgas_200 - f_stars) *
                                                        prms.m200m),
                                         r500_in_range, prms.r200m)

    gas_kwargs = {'cosmo': prms.cosmo,
                  'r_h': prms.r200m,
                  'r200m_inf': prms.r200m,
                  'm200m_inf': prms.m200m,
                  'm_h': tools.m_h(prof_gas, r_range),
                  'profile': prof_gas,
                  'profile_mass': prof_mass,
                  'profile_f': prof_gas_f}

    dens_gas = dens.Profile(**gas_kwargs)
    return dens_gas

# ------------------------------------------------------------------------------
# End of load_gas_smooth_r500c_r200m()
# ------------------------------------------------------------------------------

def load_gas_rmax(prms, r_max, f_stars, q_f=50, q_rc=50, q_beta=50):
    '''
    Return beta profiles with fgas_200m = f_obs_extrapolated = fgas_rmax

    Parameters
    ----------
    f_stars : (m,) array
      stellar fraction for each halo mass
    prms : p.Parameters object
      contains relevant model info
    q_f : float
      percentile for fgas-m500 relation fit
    q_rc : float
      percentile for rc-m500 relation fit
    q_beta : float
      percentile for beta-m500 relation fit
    bar2dmo : bool
      specifies whether to carry out hmf conversion for missing m200m
    '''
    # radius in terms of r500c
    r_range = np.array([np.logspace(prms.r_min, np.log10(rm), prms.r_bins)
                        for i,rm in enumerate(r_max)])
    rx = r_range / prms.r500c.reshape(-1,1)

    # ---------------------------------------------------------------- #
    # Our halo model assumes simulation variables with Hubble units    #
    # The observations have assumed h=0.7 and get different h scalings #
    # than the theoretical model                                       #
    # ---------------------------------------------------------------- #
    # FOR THE FITS WE NEED TO CONVERT ALL OF OUR HALO MODEL RESULTS TO
    # h=0.7!!!
    # Since r_c and beta are the same for all halo masses, it doesn't
    # matter here
    rc, beta = d.fit_prms(x=500, q_rc=q_rc, q_beta=q_beta)

    # gas fractions
    # h=0.7 needs to be converted here
    f_prms = d.f_gas_prms(prms.cosmo, q=q_f)
    f_gas500 = d.f_gas(prms.m500c / 0.7, cosmo=prms.cosmo, **f_prms)

    # determine the radius at which the beta profile mass exceeds the
    # baryon fraction
    # h=0.7 needs to be converted here
    r_cut = tools.r_where_m_beta(m=(prms.cosmo.omegab/prms.cosmo.omegam - f_stars) *
                                 prms.m200m,
                                 beta=beta, r_c=rc * prms.r500c,
                                 mgas_500c=f_gas500 * prms.m500c, r500c=prms.r500c)

    # r_cut should always be smaller than r200m
    r_cut = np.concatenate([prms.r200m[r_cut >= prms.r200m],
                            r_cut[r_cut < prms.r200m]])

    # relative position of virial radius
    x200m = prms.r200m / prms.r500c

    prof_gas = np.zeros_like(rx)
    prof_mass = np.zeros_like(r_cut)
    for idx, prof in enumerate(prof_gas):
        sl = (rx[idx] <= x200m[idx])
        sl_500 = (rx[idx] <= 1.)
        sl_fb = (rx[idx] >= r_cut[idx] / prms.r500c[idx])
        # this profile corresponds to the physically measured one
        # assuming h=0.7, but our theoretical rho ~ h^2
        prof_gas[idx][sl] = prof_beta(rx[idx], sl_500, rc, beta,
                                      f_gas500[idx] * prms.m500c[idx] / 0.7,
                                      prms.r500c[idx] / 0.7)[sl] * (0.7)**(-2.)
        prof_gas[idx][sl_fb] = 0.
        prof_mass[idx] = profs.m_beta(r=r_cut[idx],
                                      m_x=f_gas500[idx] * prms.m500c[idx],
                                      r_x=prms.r500c[idx],
                                      rc=rc * prms.r500c[idx], beta=beta)

    gas_kwargs = {'cosmo': prms.cosmo,
                  'r_h': r_max,
                  'r200m_inf': prms.r200m,
                  'm200m_inf': prms.m200m,
                  'm_h': tools.m_h(prof_gas, r_range),
                  'profile': prof_gas,
                  'profile_mass': prof_mass,
                  # compute FT in dens
                  'profile_f': None}

    dens_gas = dens.Profile(**gas_kwargs)
    return dens_gas

# ------------------------------------------------------------------------------
# End of load_gas_rmax()
# ------------------------------------------------------------------------------

# def load_gas_r500c_r200m_rmax(prms, r_max, f_stars, q_f=50, q_rc=50, q_beta=50):
#     '''
#     Return beta profiles + smooth with 0 from r200m
#     to 5r500c

#     Parameters
#     ----------
#     r_max : (m,) array
#       maximum radius for each halo mass to compute profile up to
#     f_stars : (m,) array
#       stellar fraction for each halo mass
#     prms : p.Parameters object
#       contains relevant model info
#     q_f : float
#       percentile for fgas-m500 relation fit
#     q_rc : float
#       percentile for rc-m500 relation fit
#     q_beta : float
#       percentile for beta-m500 relation fit
#     bar2dmo : bool
#       specifies whether to carry out hmf conversion for missing m200m
#     '''
#     # radius in terms of r500c
#     r_range = np.array([np.logspace(prms.r_min, np.log10(rm), prms.r_bins)
#                         for i,rm in enumerate(r_max)])
#     rx = r_range / prms.r500c.reshape(-1,1)

#     # ---------------------------------------------------------------- #
#     # Our halo model assumes simulation variables with Hubble units    #
#     # The observations have assumed h=0.7 and get different h scalings #
#     # than the theoretical model                                       #
#     # ---------------------------------------------------------------- #
#     # FOR THE FITS WE NEED TO CONVERT ALL OF OUR HALO MODEL RESULTS TO
#     # h=0.7!!!
#     # Since r_c and beta are the same for all halo masses, it doesn't
#     # matter here
#     rc, beta = d.fit_prms(x=500, q_rc=q_rc, q_beta=q_beta)

#     # gas fractions
#     # h=0.7 needs to be converted here
#     f_prms = d.f_gas_prms(prms.cosmo, q=q_f)
#     f_gas500 = d.f_gas(prms.m500c / 0.7, cosmo=prms.cosmo, **f_prms)
#     f_b = prms.cosmo.omegab / prms.cosmo.omegam

#     # determine the radius at which the beta profile mass exceeds the
#     # baryon fraction
#     # h=0.7 needs to be converted here
#     r_cut = tools.r_where_m_beta((f_b - f_stars) * prms.m200m,
#                                  beta, rc,
#                                  f_gas500 * prms.m500c, prms.r500c)

#     # r_cut should always be smaller than r200m
#     r_cut = np.concatenate([prms.r200m[r_cut >= prms.r200m],
#                             r_cut[r_cut < prms.r200m]])

#     # relative position of virial radius
#     x200m = prms.r200m / prms.r500c

#     prof_gas = np.zeros_like(rx)
#     prof_mass = np.zeros_like(r_cut)
#     for idx, prof in enumerate(prof_gas):
#         # radial range between r500c and r200m
#         sl_500_200 = ((rx[idx] >= 1.) & (rx[idx] <= x200m[idx]))
#         sl_gt200 = (rx[idx] >= x200m[idx])
#         sl_gt500 = (rx[idx] >= 1.)
#         # radial range up to r500c
#         sl_500 = (rx[idx] <= 1.)
#         sl_fb = (rx[idx] >= r_cut[idx] / prms.r500c[idx])

#         # use beta profile up to r500c

#         # this profile corresponds to the physically measured one
#         # assuming h=0.7, but our theoretical rho ~ h^2
#         prof_gas[idx][sl_500] = prof_beta(rx[idx][sl_500], sl_500[sl_500],
#                                           rc, beta,
#                                           f_gas500[idx] * prms.m500c[idx] / 0.7,
#                                           prms.r500c[idx] / 0.7) * (0.7)**(-2.)

#         m_gas500 = tools.m_h(prof_gas[idx], r_range[idx])

#         # put remaining mass in smooth component outside r500c up to r200m
#         prof_gas[idx][sl_gt500] = 1.
#         mass_gt500 = tools.m_h(prof_gas[idx][sl_gt500], r_range[idx][sl_gt500])

#         prof_gas[idx][sl_gt500] *= ((f_b - f_stars[idx] - m_gas500 / prms.m200m[idx]) *
#                                     prms.m200m[idx] / mass_gt500)

#         prof_gas[idx][sl_gt200] = 0.
#         prof_gas[idx][sl_fb] = 0.

#     gas_kwargs = {'cosmo': prms.cosmo,
#                   'r_h': r_max,
#                   'r200m_inf': prms.r200m,
#                   'm200m_inf': prms.m200m,
#                   # can integrate entire profile, since rho = 0 for r > r200m
#                   'm_h': tools.m_h(prof_gas, r_range),
#                   'profile': prof_gas,
#                   # compute FT in dens
#                   'profile_f': None}

#     dens_gas = dens.Profile(**gas_kwargs)
#     return dens_gas

# # ------------------------------------------------------------------------------
# # End of load_gas_r500c_r200m_rmax()
# # ------------------------------------------------------------------------------

def load_gas_smooth_r200m_rmax(prms, r_max, fgas_200, f_stars):
    '''
    Return uniform profiles with fgas_200m = 0 and fgas_5r500c = f_b - fgas_200

    Parameters
    ----------
    m_dmo : (m,) array
      halo mass at r200m for model to be joined, used to calculate hmf
    prms : p.Parameters object
      contains relevant model info
    fgas_200: (m,) array
      gas fraction at r200m for the model to be joined
    bar2dmo : bool
      specifies whether to carry out hmf conversion for missing m200m
    '''
    # radius in terms of r500c
    r_range = np.array([np.logspace(prms.r_min, np.log10(rm), prms.r_bins)
                        for i,rm in enumerate(r_max)])
    rx = r_range / prms.r500c.reshape(-1,1)

    # gas fractions
    f_b = prms.cosmo.omegab / prms.cosmo.omegam

    # relative position of virial radius
    x200m = prms.r200m / prms.r500c

    prof_gas = np.zeros_like(rx)
    prof_gas_f = np.zeros_like(rx)
    prof_mass = np.zeros_like(prms.m200m)
    for idx, prof in enumerate(prof_gas):
        prof_gas[idx] = profs.profile_uniform(r_range[idx],
                                              (f_b - fgas_200[idx] - f_stars[idx]) *
                                              prms.m200m[idx],
                                              prms.r200m[idx], r_max[idx])
        # this is the mass inside r200m!!
        prof_mass[idx] = 0.

    prof_gas_f = profs.profile_uniform_f(prms.k_range, ((f_b - fgas_200 - f_stars) *
                                                        prms.m200m),
                                         prms.r200m, r_max)

    gas_kwargs = {'cosmo': prms.cosmo,
                  'r_h': r_max,
                  'r200m_inf': prms.r200m,
                  'm200m_inf': prms.m200m,
                  'm_h': tools.m_h(prof_gas, r_range),
                  'profile': prof_gas,
                  'profile_mass': prof_mass,
                  'profile_f': prof_gas_f}

    dens_gas = dens.Profile(**gas_kwargs)
    return dens_gas

# ------------------------------------------------------------------------------
# End of load_gas_smooth_r200m_rmax()
# ------------------------------------------------------------------------------

def load_gas_plaw_r500c_rmax(prms, f_stars, rho_500c, r_max, gamma, q_f=50,
                             q_rc=50, q_beta=50):
    '''
    Return a beta profile upto r500c and a power law with index gamma upto r_max,
    or wherever the baryon fraction is reached.
    '''
    # radius in terms of r500c
    r_range = np.array([np.logspace(prms.r_min, np.log10(rm), prms.r_bins)
                        for i,rm in enumerate(r_max)])
    rx = r_range / prms.r500c.reshape(-1,1)

    # ---------------------------------------------------------------- #
    # Our halo model assumes simulation variables with Hubble units    #
    # The observations have assumed h=0.7 and get different h scalings #
    # than the theoretical model                                       #
    # ---------------------------------------------------------------- #
    # FOR THE FITS WE NEED TO CONVERT ALL OF OUR HALO MODEL RESULTS TO
    # h=0.7!!!
    # Since r_c and beta are the same for all halo masses, it doesn't
    # matter here
    rc, beta = d.fit_prms(x=500, q_rc=q_rc, q_beta=q_beta)

    # gas fractions
    # h=0.7 needs to be converted here
    f_prms = d.f_gas_prms(prms.cosmo, q=q_f)
    f_gas500 = d.f_gas(prms.m500c / 0.7, cosmo=prms.cosmo, **f_prms)

    # determine the radius at which the beta profile mass exceeds the
    # baryon fraction, we take high r_y to make sure we get convergence
    r_cut = profs.r_where_m_beta_plaw(m=(prms.cosmo.omegab/prms.cosmo.omegam - f_stars) *
                                      prms.m200m, m_x=f_gas500 * prms.m500c,
                                      rho_x=rho_500c, r_x=prms.r500c,
                                      r_y=1000 * r_max, gamma=gamma)

    r_cut = np.concatenate([r_max[r_cut >= r_max], r_cut[r_cut < r_max]])

    prof_gas = np.zeros_like(rx)
    prof_mass = np.zeros_like(r_cut)
    for idx, prof in enumerate(prof_gas):
        sl_500 = tools.lte(rx[idx], 1.)
        sl_fb = tools.gte(rx[idx], r_cut[idx] / prms.r500c[idx])
        # this profile corresponds to the physically measured one
        # assuming h=0.7, but our theoretical rho ~ h^2
        prof_gas[idx][sl_500] = prof_beta(rx[idx], sl_500, rc, beta,
                                          f_gas500[idx] * prms.m500c[idx] / 0.7,
                                          prms.r500c[idx] / 0.7)[sl_500] * (0.7)**(-2.)

        rho_x = prof_gas[idx][sl_500][-1]
        prof_gas[idx][~sl_500] = (profs.profile_plaw((rx[idx] * prms.r500c[idx]).reshape(1,-1),
                                                     rho_x=np.array([rho_x]),
                                                     r_x=np.array([prms.r500c[idx]]),
                                                     r_y=np.array([r_max[idx]]),
                                                     gamma=np.array([gamma])).reshape(-1))[~sl_500]
        prof_gas[idx][sl_fb] = 0.

        # mass inside r200m!!
        prof_mass[idx] = profs.m_beta_plaw(r=prms.r200m[idx],
                                           m_x=f_gas500[idx] * prms.m500c[idx],
                                           rho_x=rho_500c[idx],
                                           r_x=prms.r500c[idx],
                                           r_y=r_cut[idx],
                                           gamma=gamma)

    gas_kwargs = {'cosmo': prms.cosmo,
                  'r_h': r_max,
                  'r200m_inf': prms.r200m,
                  'm200m_inf': prms.m200m,
                  'm_h': tools.m_h(prof_gas, r_range),
                  'profile': prof_gas,
                  'profile_mass': prof_mass,
                  # compute FT in dens
                  'profile_f': None}

    dens_gas = dens.Profile(**gas_kwargs)
    return dens_gas

# ----------------------------------------------------------------------
# End of load_gas_plaw_rmax()
# ----------------------------------------------------------------------

def load_centrals(prms, f_comp='cen'):
    '''
    Return delta profiles with fstars_500c = f_obs

    Parameters
    ----------
    prms : p.Parameters object
      contains relevant model info
    bar2dmo : bool
      specifies whether to carry out hmf conversion for missing m200m
    '''
    # stellar fraction
    f_cen = d.f_stars(prms.m200m / 0.7, comp=f_comp)

    cen_kwargs = {'cosmo': prms.cosmo,
                  'r_h': prms.r200m,
                  'r200m_inf': prms.r200m,
                  'm200m_inf': prms.m200m,
                  # only f_cen of halo mass in dm
                  'm_h': prms.m200m * f_cen,
                  'profile': profs.profile_delta,
                  'profile_args': {'m_range': f_cen * prms.m200m},
                  'profile_mass': profs.m_delta,
                  'profile_f': profs.profile_delta_f,
                  'profile_f_args': {'m_range': f_cen * prms.m200m}}

    dens_cen = dens.Profile(**cen_kwargs)
    return dens_cen

# ------------------------------------------------------------------------------
# End of load_centrals()
# ------------------------------------------------------------------------------

def load_satellites(prms, f_c=0.86):
    '''
    Return NFW profiles with fstars_500c = f_obs

    Parameters
    ----------
    f_c : float
      ratio between c_sat(m) and c_dm(m) used in iHOD
    prms : p.Parameters object
      contains relevant model info
    bar2dmo : bool
      specifies whether to carry out hmf conversion for missing m200m
    '''
    # stellar fraction
    f_sat = d.f_stars(prms.m200m / 0.7, comp='sat')
    c200m = tools.c_duffy(prms.m200m).reshape(-1)
    c = f_c * c200m

    sat_kwargs = {'cosmo': prms.cosmo,
                  'r_h': prms.r200m,
                  'r200m_inf': prms.r200m,
                  'm200m_inf': prms.m200m,
                  # only f_dm of halo mass in dm
                  'm_h': prms.m200m * f_sat,
                  'profile': profs.profile_NFW,
                  'profile_args': {'m_x': f_sat * prms.m200m,
                                   'c_x': c,
                                   'r_x': prms.r200m},
                  'profile_mass': profs.m_NFW,
                  'profile_f': profs.profile_NFW_f,
                  'profile_f_args': {'m_x': f_sat * prms.m200m,
                                     'c_x': c,
                                     'r_x': prms.r200m}}

    dens_sat = dens.Profile(**sat_kwargs)
    return dens_sat

# ------------------------------------------------------------------------------
# End of load_satellites()
# ------------------------------------------------------------------------------

def load_centrals_rmax(prms, r_max, f_comp='cen'):
    '''
    Return delta profiles with fstars_500c = f_obs

    '''
    # stellar fraction
    f_cen = d.f_stars(prms.m200m / 0.7, comp=f_comp)

    cen_kwargs = {'cosmo': prms.cosmo,
                  'r_h': r_max,
                  'r200m_inf': prms.r200m,
                  'm200m_inf': prms.m200m,
                  # only f_cen of halo mass in dm
                  'm_h': prms.m200m * f_cen,
                  'profile': profs.profile_delta,
                  'profile_args': {'m_range': f_cen * prms.m200m},
                  'profile_mass': profs.m_delta,
                  'profile_f': profs.profile_delta_f,
                  'profile_f_args': {'m_range': f_cen * prms.m200m}}

    dens_cen = dens.Profile(**cen_kwargs)
    return dens_cen

# ------------------------------------------------------------------------------
# End of load_centrals_rmax()
# ------------------------------------------------------------------------------

def load_satellites_rmax(prms, r_max, f_c=0.86):
    '''
    Return NFW profiles with fstars_500c = f_obs

    Parameters
    ----------
    f_c : float
      ratio between c_sat(m) and c_dm(m) used in iHOD
    prms : p.Parameters object
      contains relevant model info
    bar2dmo : bool
      specifies whether to carry out hmf conversion for missing m200m
    '''
    r_range = np.array([np.logspace(prms.r_min, np.log10(rm), prms.r_bins)
                        for i,rm in enumerate(r_max)])
    rx = r_range / prms.r200m.reshape(-1,1)

    # stellar fraction
    f_sat = d.f_stars(prms.m200m / 0.7, comp='sat')
    c200m = tools.c_duffy(prms.m200m).reshape(-1)

    c = f_c * c200m

    prof_stars = np.zeros_like(rx)
    prof_mass = np.zeros_like(prms.m200m)
    for idx, prof in enumerate(prof_stars):
        sl = (rx[idx] <= 1.)
        prof_stars[idx][sl] = profs.profile_NFW(r_range[idx][sl].reshape(1,-1),
                                                (f_sat[idx] * prms.m200m[idx]).reshape(1),
                                                c[idx].reshape(1),
                                                prms.r200m[idx].reshape(1)).reshape(-1)
        prof_mass[idx] = (f_sat[idx] * prms.m200m[idx])


    sat_kwargs = {'cosmo': prms.cosmo,
                  'r_h': r_max,
                  'r200m_inf': prms.r200m,
                  'm200m_inf': prms.m200m,
                  # only f_dm of halo mass in dm
                  'm_h': prms.m200m * f_sat,
                  'profile': prof_stars,
                  'profile_mass': prof_mass,
                  'profile_f': profs.profile_NFW_f,
                  'profile_f_args': {'m_x': f_sat * prms.m200m,
                                     'c_x': c,
                                     'r_x': prms.r200m}}

    dens_sat = dens.Profile(**sat_kwargs)
    return dens_sat

# ------------------------------------------------------------------------------
# End of load_satellites_rmax()
# ------------------------------------------------------------------------------

def load_gamma(prms, r_max,
               gamma=np.array([2.]),
               f_c=0.86,
               q_f=50, q_rc=50, q_beta=50,
               delta=False, bar2dmo=True):
    '''
    Load all of our different models, the ones upto r200m and the ones upto
    r_max.

    Parameters
    ----------
    prms : p.Parameters object
      model parameters
    r_max : float
      maximum radius to extend our profiles upto
    gamma : array
      values of the slope to compute for
    q_f : int
      quantile for which to compute the f_gas,500c relation
    q_rc : int
      quantile for which to fit r_c
    q_beta : int
      quantile for which to fit beta
    delta : bool
      assume delta profile for the stellar contribution
    bar2dmo : bool
      convert halo mass in halo mass function to account for baryonic correction

    Returns
    -------
    results : dict
      dictionary containing the power.Power objects corresponding to our models
    '''
    # the data assumed h=0.7, but resulting f_star is independent of h in our
    # model
    f_stars = d.f_stars(prms.m200m / 0.7, 'all')
    f_dm = prms.cosmo.omegac / prms.cosmo.omegam

    dm_dmo_rmax = load_dm_dmo_rmax(prms, r_max)
    pow_dm_dmo_rmax = power.Power(dm_dmo_rmax, bar2dmo=False)

    # load stars
    if not delta:
        cen_rmax = load_centrals_rmax(prms, r_max, f_comp='cen')
        sat_rmax = load_satellites_rmax(prms, r_max, f_c=f_c)
        stars_rmax = cen_rmax + sat_rmax

    else:
        stars_rmax = load_centrals_rmax(prms, r_max, f_comp='all')

    # need to load the gas up to r500c to get rho_500c
    gas_beta = load_gas_obs(prms, q_f, q_rc, q_beta)

    # load gas_beta_plaw
    rho_500c = np.array([rho[rho > 0.][-1] for rho in gas_beta.rho_r])

    results = {}
    for idx, g in enumerate(gamma):
        gas_plaw_rmax = load_gas_plaw_r500c_rmax(prms, f_stars, rho_500c, r_max, g,
                                                 q_f, q_rc, q_beta)

        dm_plaw_rmax = load_dm_rmax(prms, r_max,
                                    m200m_obs=((f_stars + f_dm + gas_plaw_rmax.f200m_obs)
                                               * prms.m200m))

        pow_gas_plaw_rmax = power.Power(dm_plaw_rmax + gas_plaw_rmax + stars_rmax, bar2dmo=bar2dmo)
        results['{:d}'.format(idx)] = {'pow': pow_gas_plaw_rmax,
                                       'pow_dmo': pow_dm_dmo_rmax,
                                       'd_gas': gas_plaw_rmax,
                                       'd_dm': dm_plaw_rmax,
                                       'd_stars': stars_rmax}

    return results

# ----------------------------------------------------------------------
# End of load_gamma()
# ----------------------------------------------------------------------

def load_models(prms, r_max, gamma=2., q_f=50, q_rc=50, q_beta=50,
                delta=False, bar2dmo=True):
    '''
    Load all of our different models, the ones upto r200m and the ones upto
    r_max.

    Parameters
    ----------
    r_max : float
      maximum radius to extend our profiles upto
    q_f : int
      quantile for which to compute the f_gas,500c relation
    q_rc : int
      quantile for which to fit r_c
    q_beta : int
      quantile for which to fit beta
    delta : bool
      assume delta profile for the stellar contribution
    bar2dmo : bool
      convert halo mass in halo mass function to account for baryonic correction

    Returns
    -------
    results : dict
      dictionary containing the power.Power objects corresponding to our models
    '''
    # the data assumed h=0.7, but resulting f_star is independent of h in our
    # model
    f_stars = d.f_stars(prms.m200m / 0.7, 'all')
    f_dm = prms.cosmo.omegac / prms.cosmo.omegam

    # load dmo power spectrum
    dm_dmo = load_dm_dmo(prms)
    pow_dm_dmo = power.Power(dm_dmo, bar2dmo=False)

    dm_dmo_rmax = load_dm_dmo_rmax(prms, r_max)
    pow_dm_dmo_rmax = power.Power(dm_dmo_rmax, bar2dmo=False)

    # load stars
    if not delta:
        cen = load_centrals(prms, f_comp='cen')
        sat = load_satellites(prms, f_c=0.86)
        stars = cen + sat

        cen_rmax = load_centrals_rmax(prms, r_max, f_comp='cen')
        sat_rmax = load_satellites_rmax(prms, r_max, f_c=0.86)
        stars_rmax = cen_rmax + sat_rmax

    else:
        stars = load_centrals(prms, f_comp='all')

        stars_rmax = load_centrals_rmax(prms, r_max, f_comp='all')

    # --------------------------------------------------------------------------
    # MODEL 1
    # --------------------------------------------------------------------------
    # load gas_obs
    gas = load_gas(prms, f_stars, q_f, q_rc, q_beta)

    # --------------------------------------------------------------------------
    # MODEL 2
    # --------------------------------------------------------------------------
    # load gas_smooth_r500c_r200m
    gas_beta = load_gas_obs(prms, q_f, q_rc, q_beta)
    gas_smooth = load_gas_smooth_r500c_r200m(prms, gas_beta.f200m_obs, f_stars)


    # # --------------------------------------------------------------------------
    # # MODEL 3
    # # --------------------------------------------------------------------------
    # # load_gas_smooth_r500c_rmax
    # gas_r500c_rmax = load_gas_r500c_r200m_rmax(prms, r_max, f_stars, q_f, q_rc, q_beta)
    # gas_smooth_r200m_rmax = load_gas_smooth_r200m_rmax(prms, r_max,
    #                                                    gas_r500c_rmax.f200m_obs,
    #                                                    f_stars)

    # pow_gas_smooth_r500c_rmax = power.Power(dm_rmax + gas_r500c_rmax +
    #                                         gas_smooth_r200m_rmax + stars_rmax,
    #                                         bar2dmo=bar2dmo)

    # --------------------------------------------------------------------------
    # MODEL 4
    # --------------------------------------------------------------------------
    # load gas_smooth_r200m_rmax
    gas_rmax = load_gas_rmax(prms, r_max, f_stars, q_f, q_rc, q_beta)
    gas_smooth_r200m_rmax = load_gas_smooth_r200m_rmax(prms, r_max,
                                                       gas_rmax.f200m_obs,
                                                       f_stars)

    # --------------------------------------------------------------------------
    # MODEL 5
    # --------------------------------------------------------------------------
    # load gas_beta_plaw
    rho_500c = np.array([rho[rho > 0.][-1] for rho in gas_beta.rho_r])
    gas_plaw_rmax = load_gas_plaw_r500c_rmax(prms, f_stars, rho_500c, r_max, gamma,
                                             q_f, q_rc, q_beta)

    # load dm models
    dm = load_dm(prms, m200m_obs=prms.m200m)
    dm_rmax = load_dm_rmax(prms, r_max, m200m_obs=prms.m200m)

    dm_gas = load_dm(prms, m200m_obs=(f_stars + f_dm + gas.f200m_obs) * prms.m200m)
    dm_gas_rmax = load_dm_rmax(prms, r_max, m200m_obs=((f_stars + f_dm + gas.f200m_obs)
                                                       * prms.m200m))

    dm_plaw_rmax = load_dm_rmax(prms, r_max,
                                m200m_obs=((f_stars + f_dm + gas_plaw_rmax.f200m_obs)
                                           * prms.m200m))

    # create power objects
    pow_gas = power.Power(dm_gas + gas + stars, bar2dmo=bar2dmo)
    pow_gas_smooth_r500c_r200m = power.Power(dm + gas_beta + gas_smooth + stars,
                                             bar2dmo=bar2dmo)
    pow_gas_smooth_r200m_rmax = power.Power(dm_gas_rmax + gas_rmax +
                                            gas_smooth_r200m_rmax + stars_rmax,
                                            bar2dmo=bar2dmo)
    pow_gas_plaw_rmax = power.Power(dm_plaw_rmax + gas_plaw_rmax + stars_rmax, bar2dmo=bar2dmo)


    results = {'d_dm_dmo': dm_dmo,
               'd_dm': dm,
               'd_dm_gas': dm_gas,
               'd_stars': stars,
               'd_gas': gas,
               'd_gas_beta': gas_beta,
               'd_gas_beta_smooth': gas_smooth,
               'd_gas_plaw': gas_plaw_rmax,
               'd_dm_dmo_rmax': dm_dmo_rmax,
               'd_dm_rmax': dm_rmax,
               'd_stars_rmax': stars_rmax,
               'd_gas_rmax': gas_rmax,
               'd_gas_smooth_rmax': gas_smooth_r200m_rmax,
               'prms': prms,
               'dm_dmo': pow_dm_dmo,
               'dm_dmo_rmax': pow_dm_dmo_rmax,
               'gas': pow_gas,
               'smooth_r500c_r200m': pow_gas_smooth_r500c_r200m,
               'smooth_r200m_rmax': pow_gas_smooth_r200m_rmax,
               'plaw_r500c_rmax': pow_gas_plaw_rmax}

    return results

# ------------------------------------------------------------------------------
# End of load_models()
# ------------------------------------------------------------------------------

def plot_profiles_paper(dens_gas,
                        dens_gas_r500c_r200m,
                        dens_gas_r200m_5r500c,
                        dens_stars_nodelta,
                        dens_dm,
                        rho_k=False,
                        prms=p.prms):
    '''
    Plot the density for our different gas profiles in one plot
    '''
    fig = plt.figure(figsize=(20,8))
    ax1 = fig.add_axes([0.1,0.1,0.266,0.8])
    ax2 = fig.add_axes([0.366,0.1,0.266,0.8])
    ax3 = fig.add_axes([0.632,0.1,0.266,0.8])

    idx_1 = 0
    idx_2 = 50
    idx_3 = -1
    r200m = prms.r200m
    reload(pl)
    pl.set_style('line')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if not rho_k:
        norm = prms.rho_crit
        # Plot idx_1
        # gas
        l2, = ax1.plot(dens_gas_r500c_r200m.r_range[idx_1] / r200m[idx_1],
                       (dens_gas_r500c_r200m.rho_r[idx_1] *
                        dens_gas_r500c_r200m.f_comp[idx_1] / norm),
                       lw=3, c=colors[1])
        l3, = ax1.plot(dens_gas_r200m_5r500c.r_range[idx_1] / r200m[idx_1],
                       (dens_gas_r200m_5r500c.rho_r[idx_1] *
                        dens_gas_r200m_5r500c.f_comp[idx_1] / norm),
                       lw=4, c=colors[0])
        l1, = ax1.plot(dens_gas.r_range[idx_1] / r200m[idx_1],
                       dens_gas.rho_r[idx_1] * dens_gas.f_comp[idx_1]/norm,
                       lw=2, c=colors[2])
        # stars
        markerson = 0.1
        ls, = ax1.plot(dens_stars_nodelta.r_range[idx_1] / r200m[idx_1],
                       dens_stars_nodelta.rho_r[idx_1] * dens_stars_nodelta.f_comp[idx_1]/norm,
                       c='k', marker='*',
                       markevery=markerson,
                       markersize=8)

        # dark matter
        ld, = ax1.plot(dens_dm.r_range[idx_1] / r200m[idx_1],
                       dens_dm.rho_r[idx_1] * dens_dm.f_comp[idx_1]/norm,
                       c='k', marker='o',
                       markevery=markerson,
                       markersize=8)

        # # dark matter
        # ld5r500c, = ax1.plot(dens_dm_5r500c.r_range[idx_1] / r200m[idx_1],
        #                      dens_dm_5r500c.rho_r[idx_1] * dens_dm_5r500c.f_comp[idx_1]/norm,
        #                      c='r', marker='o',
        #                      markevery=markerson,
        #                      markersize=8)

        ax1.axvline(x=prms.r500c[idx_1] / prms.r200m[idx_1], ls='--', c='k')
        ax1.text(x=prms.r500c[idx_1] / prms.r200m[idx_1], y=1e2, s=r'$r_\mathrm{500c}$',
                 ha='left', va='bottom')

        # Plot idx_2
        # gas
        ax2.plot(dens_gas_r500c_r200m.r_range[idx_2] / r200m[idx_2],
                 (dens_gas_r500c_r200m.rho_r[idx_2] *
                  dens_gas_r500c_r200m.f_comp[idx_2] / norm),
                 lw=3, c=colors[1])
        ax2.plot(dens_gas_r200m_5r500c.r_range[idx_2] / r200m[idx_2],
                 (dens_gas_r200m_5r500c.rho_r[idx_2] *
                  dens_gas_r200m_5r500c.f_comp[idx_2] / norm),
                 lw=4, c=colors[0])
        ax2.plot(dens_gas.r_range[idx_2] / r200m[idx_2],
                 dens_gas.rho_r[idx_2] * dens_gas.f_comp[idx_2]/norm,
                 lw=2, c=colors[2])

        # stars
        ls, = ax2.plot(dens_stars_nodelta.r_range[idx_2] / r200m[idx_2],
                       dens_stars_nodelta.rho_r[idx_2] * dens_stars_nodelta.f_comp[idx_2]/norm,
                       c='k', marker='*',
                       markevery=markerson,
                       markersize=8,
                       label=r'$\mathtt{\star\_NFW}$')

        # dark matter
        ld, = ax2.plot(dens_dm.r_range[idx_2] / r200m[idx_2],
                       dens_dm.rho_r[idx_2] * dens_dm.f_comp[idx_2]/norm,
                       c='k', marker='o',
                       markevery=markerson,
                       markersize=8,
                       label='dark matter')

        # # dark matter
        # ld5r500c, = ax2.plot(dens_dm_5r500c.r_range[idx_2] / r200m[idx_2],
        #                      dens_dm_5r500c.rho_r[idx_2] * dens_dm_5r500c.f_comp[idx_2]/norm,
        #                      c='r', marker='o',
        #                      markevery=markerson,
        #                      markersize=8)

        ax2.axvline(x=prms.r500c[idx_2] / prms.r200m[idx_2], ls='--', c='k')
        ax2.text(x=prms.r500c[idx_2] / prms.r200m[idx_2], y=1e2, s=r'$r_\mathrm{500c}$',
                 ha='left', va='bottom')

        # Plot idx_3
        # gas
        ax3.plot(dens_gas_r500c_r200m.r_range[idx_3] / r200m[idx_3],
                 (dens_gas_r500c_r200m.rho_r[idx_3] *
                  dens_gas_r500c_r200m.f_comp[idx_3] / norm),
                 lw=3, c=colors[1])
        ax3.plot(dens_gas_r200m_5r500c.r_range[idx_3] / r200m[idx_3],
                 (dens_gas_r200m_5r500c.rho_r[idx_3] *
                  dens_gas_r200m_5r500c.f_comp[idx_3] / norm),
                 lw=4, c=colors[0])
        ax3.plot(dens_gas.r_range[idx_3] / r200m[idx_3],
                 dens_gas.rho_r[idx_3] * dens_gas.f_comp[idx_3]/norm,
                 lw=2, c=colors[2])

        # stars
        ls, = ax3.plot(dens_stars_nodelta.r_range[idx_3] / r200m[idx_3],
                       dens_stars_nodelta.rho_r[idx_3] * dens_stars_nodelta.f_comp[idx_3]/norm,
                       c='k', marker='*',
                       markevery=markerson,
                       markersize=8)

        # dark matter
        ld, = ax3.plot(dens_dm.r_range[idx_3] / r200m[idx_3],
                       dens_dm.rho_r[idx_3] * dens_dm.f_comp[idx_3]/norm,
                       c='k', marker='o',
                       markevery=markerson,
                       markersize=8)

        # # dark matter
        # ld5r500c, = ax3.plot(dens_dm_5r500c.r_range[idx_3] / r200m[idx_3],
        #                      dens_dm_5r500c.rho_r[idx_3] * dens_dm_5r500c.f_comp[idx_3]/norm,
        #                      c='r', marker='o',
        #                      markevery=markerson,
        #                      markersize=8)

        ax3.axvline(x=prms.r500c[idx_3] / prms.r200m[idx_3], ls='--', c='k')
        ax3.text(x=prms.r500c[idx_3] / prms.r200m[idx_3], y=110, s=r'$r_\mathrm{500c}$',
                 ha='left', va='bottom')

        ax1.set_xlim(1e-2, 3)
        ax1.set_ylim(1e-1, 1e4)
        ax2.set_xlim(1e-2, 3)
        ax2.set_ylim(1e-1, 1e4)
        ax3.set_xlim(1e-2, 3)
        ax3.set_ylim(1e-1, 1e4)

        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax3.set_xscale('log')
        ax3.set_yscale('log')

        ax1.tick_params(axis='x', which='major', pad=6)
        ax2.tick_params(axis='x', which='major', pad=6)
        ax3.tick_params(axis='x', which='major', pad=6)

        ax2.set_xlabel(r'$r/r_\mathrm{200m}$', labelpad=-10)
        ax1.set_ylabel(r'$\rho(r)/\rho_\mathrm{c}$')
        ax2.set_yticklabels([])
        ax3.set_yticklabels([])
        # ticks2 = ax2.get_xticklabels()
        # ticks2[-6].set_visible(False)
        # ticks3 = ax3.get_xticklabels()
        # ticks3[-6].set_visible(False)

        ax1.set_title(r'$m_{\mathrm{200m}} = 10^{%.1f} \, h^{-1} \, \mathrm{M_\odot}$'%np.log10(prms.m200m[idx_1]), y=1.015, fontsize=28)
        ax2.set_title(r'$m_{\mathrm{200m}} = 10^{%.1f} \, h^{-1} \, \mathrm{M_\odot}$'%np.log10(prms.m200m[idx_2]), y=1.015, fontsize=28)
        ax3.set_title(r'$m_{\mathrm{200m}} = 10^{%.1f} \, h^{-1} \, \mathrm{M_\odot}$'%np.log10(prms.m200m[idx_3]), y=1.015, fontsize=28)


        leg1 = ax1.legend([l1, l2, l3],
                          [r'$\mathtt{\beta\_r200m\_nofb}$',
                           r'$\mathtt{\beta\_r500c\_fb\_r200m}$',
                           r'$\mathtt{\beta\_r200m\_fb\_5r500c}$'],
                          loc=2, fontsize=28, frameon=True, framealpha=0.8)
        leg1.get_frame().set_linewidth(0.0)

        leg2 = ax2.legend(loc=3, fontsize=28, frameon=True, framealpha=0.8)
        leg2.get_frame().set_linewidth(0.0)

        plt.savefig('obs_rho_extrapolated.pdf', transparent=True,
                    bbox_inches='tight')

    else:
        norm = 1.
        # Plot idx_1
        # gas
        ax1.plot(dens_gas.k_range,
                 dens_gas.rho_k[idx_1] * dens_gas.f_comp[idx_1]/norm,
                 lw=2, c=colors[2], label=r'$\mathtt{\beta\_r200m\_nofb}$')
        ax1.plot(dens_gas_r500c_r200m.k_range,
                 (dens_gas_r500c_r200m.rho_k[idx_1] *
                  dens_gas_r500c_r200m.f_comp[idx_1] / norm),
                 lw=3, c=colors[1], label=r'$\mathtt{\beta\_r500c\_fb\_r200m}$')
        ax1.plot(dens_gas_r200m_5r500c.k_range,
                 (dens_gas_r200m_5r500c.rho_k[idx_1] *
                  dens_gas_r200m_5r500c.f_comp[idx_1] / norm),
                 lw=4, c=colors[0], label=r'$\mathtt{\beta\_r200m\_fb\_5r500c}$')

        # stars
        markerson = 0.1
        ls, = ax1.plot(dens_stars_nodelta.k_range,
                       dens_stars_nodelta.rho_k[idx_1] * dens_stars_nodelta.f_comp[idx_1]/norm,
                       c='k', marker='*',
                       markevery=markerson,
                       markersize=8,
                       label=r'$\mathtt{\star\_NFW}$')

        # dark matter
        ld, = ax1.plot(dens_dm.k_range,
                       dens_dm.rho_k[idx_1] * dens_dm.f_comp[idx_1]/norm,
                       c='k', marker='o',
                       markevery=markerson,
                       markersize=8,
                       label='dark matter')

        # # dark matter
        # ld5r500c, = ax1.plot(dens_dm_5r500c.k_range,
        #                      dens_dm_5r500c.rho_k[idx_1] * dens_dm_5r500c.f_comp[idx_1]/norm,
        #                      c='r', marker='o',
        #                      markevery=markerson,
        #                      markersize=8,
        #                      label='dark matter')


        # Plot idx_2
        # gas
        ax2.plot(dens_gas.k_range,
                 dens_gas.rho_k[idx_2] * dens_gas.f_comp[idx_2]/norm,
                 lw=2, c=colors[2], label=r'$\mathtt{\beta\_r200m\_nofb}$')
        ax2.plot(dens_gas_r500c_r200m.k_range,
                 (dens_gas_r500c_r200m.rho_k[idx_2] *
                  dens_gas_r500c_r200m.f_comp[idx_2] / norm),
                 lw=3, c=colors[1], label=r'$\mathtt{\beta\_r500c\_fb\_r200m}$')
        ax2.plot(dens_gas_r200m_5r500c.k_range,
                 (dens_gas_r200m_5r500c.rho_k[idx_2] *
                  dens_gas_r200m_5r500c.f_comp[idx_2] / norm),
                 lw=4, c=colors[0], label=r'$\mathtt{\beta\_r200m\_fb\_5r500c}$')

        # stars
        markerson = 0.1
        ls, = ax2.plot(dens_stars_nodelta.k_range,
                       dens_stars_nodelta.rho_k[idx_2] * dens_stars_nodelta.f_comp[idx_2]/norm,
                       c='k', marker='*',
                       markevery=markerson,
                       markersize=8,
                       label=r'$\mathtt{\star\_NFW}$')

        # dark matter
        ld, = ax2.plot(dens_dm.k_range,
                       dens_dm.rho_k[idx_2] * dens_dm.f_comp[idx_2]/norm,
                       c='k', marker='o',
                       markevery=markerson,
                       markersize=8,
                       label='dark matter')

        # # dark matter
        # ld5r500c, = ax2.plot(dens_dm_5r500c.k_range,
        #                      dens_dm_5r500c.rho_k[idx_2] * dens_dm_5r500c.f_comp[idx_2]/norm,
        #                      c='r', marker='o',
        #                      markevery=markerson,
        #                      markersize=8,
        #                      label='dark matter')


        # Plot idx_3
        # gas
        ax3.plot(dens_gas.k_range,
                 dens_gas.rho_k[idx_3] * dens_gas.f_comp[idx_3]/norm,
                 lw=2, c=colors[2], label=r'$\mathtt{\beta\_r200m\_nofb}$')
        ax3.plot(dens_gas_r500c_r200m.k_range,
                 (dens_gas_r500c_r200m.rho_k[idx_3] *
                  dens_gas_r500c_r200m.f_comp[idx_3] / norm),
                 lw=3, c=colors[1], label=r'$\mathtt{\beta\_r500c\_fb\_r200m}$')
        ax3.plot(dens_gas_r200m_5r500c.k_range,
                 (dens_gas_r200m_5r500c.rho_k[idx_3] *
                  dens_gas_r200m_5r500c.f_comp[idx_3] / norm),
                 lw=4, c=colors[0], label=r'$\mathtt{\beta\_r200m\_fb\_5r500c}$')

        # stars
        markerson = 0.1
        ls, = ax3.plot(dens_stars_nodelta.k_range,
                       dens_stars_nodelta.rho_k[idx_3] * dens_stars_nodelta.f_comp[idx_3]/norm,
                       c='k', marker='*',
                       markevery=markerson,
                       markersize=8,
                       label=r'$\mathtt{\star\_NFW}$')

        # dark matter
        ld, = ax3.plot(dens_dm.k_range,
                       dens_dm.rho_k[idx_3] * dens_dm.f_comp[idx_3]/norm,
                       c='k', marker='o',
                       markevery=markerson,
                       markersize=8,
                       label='dark matter')

        # # dark matter
        # ld5r500c, = ax3.plot(dens_dm_5r500c.k_range,
        #                      dens_dm_5r500c.rho_k[idx_3] * dens_dm_5r500c.f_comp[idx_3]/norm,
        #                      c='r', marker='o',
        #                      markevery=markerson,
        #                      markersize=8,
        #                      label='dark matter')

        ax1.set_xscale('log')
        ax2.set_xscale('log')
        ax3.set_xscale('log')
        ax1.set_yscale('symlog',linthreshy=1e-4)
        ax2.set_yscale('symlog',linthreshy=1e-4)
        ax3.set_yscale('symlog',linthreshy=1e-4)

        ax1.set_xlim(1,100)
        ax2.set_xlim(1,100)
        ax3.set_xlim(1,100)
        ax1.set_ylim(-1, 1)
        ax2.set_ylim(-1, 1)
        ax3.set_ylim(-1, 1)

        ax1.tick_params(axis='x', which='major', pad=6)
        ax2.tick_params(axis='x', which='major', pad=6)
        ax3.tick_params(axis='x', which='major', pad=6)
        ax1.tick_params(axis='x', which='minor', bottom='on', top='on')
        ax2.tick_params(axis='x', which='minor', bottom='on', top='on')
        ax3.tick_params(axis='x', which='minor', bottom='on', top='on')

        ax2.set_xlabel(r'$k \, [h \, \mathrm{Mpc}^{-1}]$', labelpad=-10)
        ax1.set_ylabel(r'$f_\mathrm{i}(m)u(k|m)$')
        ax2.set_yticklabels([])
        ax3.set_yticklabels([])
        # ticks2 = ax2.get_xticklabels()
        # ticks2[-6].set_visible(False)
        # ticks3 = ax3.get_xticklabels()
        # ticks3[-6].set_visible(False)

        ax1.set_title(r'$m_{\mathrm{200m}} = 10^{%.1f} \, h^{-1} \, \mathrm{M_\odot}$'%np.log10(prms.m200m[idx_1]), y=1.015, fontsize=28)
        ax2.set_title(r'$m_{\mathrm{200m}} = 10^{%.1f} \, h^{-1} \, \mathrm{M_\odot}$'%np.log10(prms.m200m[idx_2]), y=1.015, fontsize=28)
        ax3.set_title(r'$m_{\mathrm{200m}} = 10^{%.1f} \, h^{-1} \, \mathrm{M_\odot}$'%np.log10(prms.m200m[idx_3]), y=1.015, fontsize=28)

        ax3.legend(loc='best', fontsize=28)
        plt.savefig('obs_rho_k_extrapolated.pdf', transparent=True,
                    bbox_inches='tight')

# ------------------------------------------------------------------------------
# End of plot_profiles_gas_paper()
# ------------------------------------------------------------------------------

# def plot_fgas200m_paper(comp_gas, comp_gas_r500c_5r500c, prms=p.prms):
#     '''
#     Plot gas mass fractions at r200m for our different models
#     '''
#     fig = plt.figure(figsize=(10,9))
#     ax = fig.add_subplot(111)

#     f_b = 1 - prms.f_dm

#     pl.set_style('line')
#     ax.plot(comp_gas.m200m, comp_gas.f_comp, label='model 1')
#     ax.plot(comp_gas_r500c_5r500c.m200m, comp_gas_r500c_5r500c.f_comp,
#             label='model 3')
#     ax.axhline(y=f_b, c='k', ls='--')

#     ax.tick_params(axis='x', which='major', pad=6)
#     text_props = ax.get_xticklabels()[0].get_font_properties()

#     # add annotation to f_bar
#     ax.annotate(r'$f_{\mathrm{b}}$',
#                  # xy=(1e14, 0.16), xycoords='data',
#                  # xytext=(1e14, 0.15), textcoords='data',
#                  xy=(10**(11), f_b), xycoords='data',
#                  xytext=(1.2 * 10**(11),
#                          f_b * 0.95), textcoords='data',
#                  fontproperties=text_props)

#     ax.set_xscale('log')
#     ax.set_xlabel('$m_\mathrm{200m} \, [\mathrm{M_\odot}/h]$')
#     ax.set_ylabel('$f_\mathrm{gas,200m}$')
#     ax.legend(loc=4)
#     plt.savefig('obs_fgas_extrapolated.pdf', transparent=True)

# # ------------------------------------------------------------------------------
# # End of plot_fgas200m_paper()
# # ------------------------------------------------------------------------------
