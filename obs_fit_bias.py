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

import halo.tools as tools
import halo.density_profiles as profs
import halo.stars as stars
import halo.gas as gas
import halo.bias as bias
import halo.parameters as p
import halo.tools as tools
import halo.model.density as dens
import halo.model.component as comp
import halo.model.power as power
import halo.data.bahamas as b
import halo.data.data as d

import numpy as np

import pdb

def load_dm_dmo(prms=p.prms):
    '''
    Pure dark matter only component with NFW profile and f_dm = 1
    '''
    m200m = prms.m200m
    m200c = prms.m200c
    r200c = prms.r200c

    f_dm = np.ones_like(m200m)
    c200m = prms.c_correa
    # c200m = b.dm_c_dmo(m200m * prms.h)
    r200m = prms.r200m

    # general profile kwargs to be used for all components
    profile_kwargs = {'r_range': prms.r_range_lin,
                      'm_bar': prms.m200m,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}
    # specific dm extra kwargs
    dm_extra = {'profile': profs.profile_NFW,
                'profile_f': profs.profile_NFW_f,
                'profile_args': {'c_x': c200m,
                                 'r_x': r200m,
                                 'rho_mean': prms.rho_m},
                'profile_f_args': {'c_x': c200m,
                                   'r_x': r200m,
                                   'rho_mean': prms.rho_m}}
    prof_dm_kwargs = tools.merge_dicts(profile_kwargs, dm_extra)
    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    comp_dm_kwargs = {'name': 'dm',
                      'r200m': r200m,
                      'm200m': m200m,
                      'p_lin': prms.p_lin,
                      'dndm': prms.dndm,
                      'f_comp': f_dm,}

    dm_kwargs = tools.merge_dicts(prof_dm_kwargs, comp_dm_kwargs)

    comp_dm = comp.Component(**dm_kwargs)
    return comp_dm

# ------------------------------------------------------------------------------
# End of load_dm_dmo()
# ------------------------------------------------------------------------------

def load_dm(m_dmo, prms=p.prms, bar2dmo=True):
    '''
    Dark matter profile with NFW profile and f_dm = 1 - f_b
    '''
    # halo model parameters
    # ! WATCH OUT ! These are the SPH equivalent masses and do thus not
    # correspond to the DMO case, unless f_gas,200m = f_b
    m200m = prms.m200m
    m200c = prms.m200c
    r200c = prms.r200c

    # it is seen in Eagle that DM haloes do not really change their c(m200m)
    # relation when fitting only the dark matter concentration as a function of
    # TOTAL halo mass
    f_dm = np.ones_like(m200m) * prms.f_dm
    c200m = prms.c_correa
    r200m = prms.r200m

    # Can give the profile all of the halo model parameters, since it is
    # a fit to observations, only the power spectrum calculation needs
    # to convert these values to the DMO equivalent cases
    profile_kwargs = {'r_range': prms.r_range_lin,
                      'm_bar': prms.m200m,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}

    # specific dm extra kwargs
    dm_extra = {'profile': profs.profile_NFW,
                'profile_f': profs.profile_NFW_f,
                'profile_args': {'c_x': c200m,
                                 'r_x': r200m,
                                 'rho_mean': prms.rho_m},
                'profile_f_args': {'c_x': c200m,
                                   'r_x': r200m,
                                   'rho_mean': prms.rho_m},}
    prof_dm_kwargs = tools.merge_dicts(profile_kwargs, dm_extra)
    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    if bar2dmo == False:
        comp_dm_kwargs = {'name': 'dm',
                           'r200m': r200m,
                           'm200m': m200m,
                           'p_lin': prms.p_lin,
                           'dndm': prms.dndm,
                           'f_comp': f_dm}
    else:
        prms_dmo = deepcopy(prms)
        prms_dmo.m200m = m_dmo
        comp_dm_kwargs = {'name': 'dm',
                           'r200m': r200m,
                           'm200m': m200m,
                           'p_lin': prms.p_lin,
                           'dndm': prms_dmo.dndm,
                           'f_comp': f_dm}

    dm_kwargs = tools.merge_dicts(prof_dm_kwargs, comp_dm_kwargs)

    comp_dm = comp.Component(**dm_kwargs)
    return comp_dm

# ------------------------------------------------------------------------------
# End of load_dm()
# ------------------------------------------------------------------------------

def load_dm_dmo_rmax(r_max, prms=p.prms):
    '''
    Pure dark matter only component with NFW profile and f_dm = 1
    '''
    m200m = prms.m200m
    m500c = prms.m500c
    r500c = prms.r500c
    r200m = prms.r200m

    r_min = prms.r_range_lin[:,0]
    r_range = np.array([np.logspace(np.log10(r_min[i]), np.log10(rm), prms.r_bins)
                        for i,rm in enumerate(r_max)])
    rx = r_range / r500c.reshape(-1,1)


    f_dm = np.ones_like(m200m)
    c200m = prms.c_correa
    # c200m = b.dm_c_dmo(m200m * prms.h)

    # relative position of virial radius
    x200m = r200m / r500c

    prof_dm = np.zeros_like(rx)
    for idx, prof in enumerate(prof_dm):
        sl = (rx[idx] <= x200m[idx])
        prof_dm[idx][sl] = profs.profile_NFW(r_range[idx][sl].reshape(1,-1),
                                             m200m[idx].reshape(1,1),
                                             c200m[idx], r200m[idx],
                                             prms.rho_m).reshape(-1)


    # Can give the profile all of the halo model parameters, since it is
    # a fit to observations, only the power spectrum calculation needs
    # to convert these values to the DMO equivalent cases
    profile_kwargs = {'r_range': r_range,
                      'm_bar': m200m,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}
    # --------------------------------------------------------------------------
    # we want the analytic solution for the FT, since we cut of the profile at
    # r200m
    dm_extra = {'profile': prof_dm, 'profile_f': profs.profile_NFW_f,
                'profile_f_args': {'c_x': c200m,
                                   'r_x': r200m,
                                   'rho_mean': prms.rho_m},}
    prof_dm_kwargs = tools.merge_dicts(profile_kwargs, dm_extra)
    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    comp_dm_kwargs = {'name': 'dm',
                      'r200m': r200m,
                      'm200m': m200m,
                      'p_lin': prms.p_lin,
                      'dndm': prms.dndm,
                      'f_comp': f_dm}

    dm_kwargs = tools.merge_dicts(prof_dm_kwargs, comp_dm_kwargs)

    comp_dm = comp.Component(**dm_kwargs)
    return comp_dm

# ------------------------------------------------------------------------------
# End of load_dm_dmo_5r500c()
# ------------------------------------------------------------------------------

def load_dm_rmax(r_max, m_dmo, prms=p.prms, bar2dmo=True):
    '''
    Return NFW profiles with up to r200m and 0 up to 5r500c
    '''
    # halo model parameters
    # ! WATCH OUT ! These are the SPH equivalent masses and do thus not
    # correspond to the DMO case, unless f_gas,200m = f_b
    m200m = prms.m200m
    m500c = prms.m500c
    r500c = prms.r500c
    r200m = prms.r200m

    r_min = prms.r_range_lin[:,0]
    r_range = np.array([np.logspace(np.log10(r_min[i]), np.log10(rm), prms.r_bins)
                        for i,rm in enumerate(r_max)])
    rx = r_range / r500c.reshape(-1,1)


    f_dm = np.ones_like(m200m) * prms.f_dm
    c_x = prms.c_correa
    r_x = r200m

    # relative position of virial radius
    x200m = r200m / r500c

    prof_dm = np.zeros_like(rx)
    for idx, prof in enumerate(prof_dm):
        sl = (rx[idx] <= x200m[idx])
        prof_dm[idx][sl] = profs.profile_NFW(r_range[idx][sl].reshape(1,-1),
                                             m200m[idx].reshape(1,1),
                                             c_x[idx], r_x[idx],
                                             prms.rho_m).reshape(-1)

    # Can give the profile all of the halo model parameters, since it is
    # a fit to observations, only the power spectrum calculation needs
    # to convert these values to the DMO equivalent cases
    profile_kwargs = {'r_range': r_range,
                      'm_bar': m200m,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}
    # --------------------------------------------------------------------------
    # we want the analytic solution for the FT, since we cut of the profile at
    # r200m
    dm_extra = {'profile': prof_dm, 'profile_f': profs.profile_NFW_f,
                'profile_f_args': {'c_x': c_x,
                                   'r_x': r200m,
                                   'rho_mean': prms.rho_m},}
    prof_dm_kwargs = tools.merge_dicts(profile_kwargs, dm_extra)
    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    if bar2dmo == False:
        comp_dm_kwargs = {'name': 'dm',
                           'r200m': r200m,
                           'm200m': m200m,
                           'p_lin': prms.p_lin,
                           'dndm': prms.dndm,
                           'f_comp': f_dm}
    else:
        prms_dmo = deepcopy(prms)
        prms_dmo.m200m = m_dmo
        comp_dm_kwargs = {'name': 'dm',
                           'r200m': r200m,
                           'm200m': m200m,
                           'p_lin': prms.p_lin,
                           'dndm': prms_dmo.dndm,
                           'f_comp': f_dm}

    dm_kwargs = tools.merge_dicts(prof_dm_kwargs, comp_dm_kwargs)

    comp_dm = comp.Component(**dm_kwargs)
    return comp_dm

# ------------------------------------------------------------------------------
# End of load_dm_5r500c()
# ------------------------------------------------------------------------------

def prof_beta(x, sl, a, b, m_sl, r500):
    '''beta profile'''
    profile = (1 + (x/a)**2)**(-3*b/2)
    mass = tools.m_h(profile[sl], x[sl] * r500)
    profile *= m_sl/mass

    return profile

def load_gas(f_stars, prms=p.prms, q_f=50, q_rc=50, q_beta=50, bar2dmo=True):
    '''
    Return beta profiles with fgas_500c = f_obs, extrapolated to r200m

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
    # halo model parameters
    # ! WATCH OUT ! These are the SPH equivalent masses and do thus not
    # correspond to the DMO case, unless f_gas,200m = f_b
    m200m = prms.m200m
    m500c = prms.m500c
    r500c = prms.r500c
    r200m = prms.r200m

    # radius in terms of r500c
    r_range = prms.r_range_lin
    rx = r_range / r500c.reshape(-1,1)

    rc, beta = d.fit_prms(x=500, q_rc=q_rc, q_beta=q_beta)
    # gas fractions
    f_prms = d.f_gas_prms_debiased(prms)
    f_gas500 = d.f_gas(m500c, prms=prms, **f_prms)

    # determine the radius at which the beta profile mass exceeds the
    # baryon fraction
    r_cut = tools.r_where_m_beta((prms.f_b - f_stars) * m200m, beta, rc,
                                 f_gas500 * m500c, r500c)

    prof_gas = np.zeros_like(rx)
    for idx, prof in enumerate(prof_gas):
        sl_500 = (rx[idx] <= 1.)
        sl_fb = (rx[idx] >= r_cut[idx] / r500c[idx])
        prof_gas[idx] = prof_beta(rx[idx], sl_500, rc, beta,
                                     f_gas500[idx] * m500c[idx],
                                     r500c[idx])
        prof_gas[idx][sl_fb] = 0.

    mgas200 = tools.m_h(prof_gas, prms.r_range_lin)
    f_gas = mgas200 / m200m

    # Now we can determine the equivalent DMO masses from the f_gas - m relation
    m_dmo = d.m200b_to_m200dmo(m200m, f_gas + f_stars, prms)
    r_dmo = tools.mass_to_radius(m_dmo, 200 * prms.rho_m)

    # # renormalize radial range
    # r_range_dmo = r_range * r_dmo.reshape(-1,1) / r200m.reshape(-1,1)
    # # ! This is not necessary, since the radial range is not required
    # # ! in the halo model, except for the density profiles which are fit
    # # ! in the observations and thus do not need to be renormalized

    # Can give the profile all of the halo model parameters, since it is
    # a fit to observations, only the power spectrum calculation needs
    # to convert these values to the DMO equivalent cases
    profile_kwargs = {'r_range': r_range,
                      'm_bar': m200m,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}
    # --------------------------------------------------------------------------
    # specific gas extra kwargs -> need f_gas
    gas_extra = {'profile': prof_gas / f_gas.reshape(-1,1),}
    prof_gas_kwargs = tools.merge_dicts(profile_kwargs, gas_extra)
    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    if bar2dmo == False:
        comp_gas_kwargs = {'name': 'gas',
                           'r200m': r200m,
                           'm200m': m200m,
                           'p_lin': prms.p_lin,
                           'dndm': prms.dndm,
                           'f_comp': f_gas}
    else:
        prms_dmo = deepcopy(prms)
        prms_dmo.m200m = m_dmo
        comp_gas_kwargs = {'name': 'gas',
                           'r200m': r200m,
                           'm200m': m200m,
                           'p_lin': prms.p_lin,
                           'dndm': prms_dmo.dndm,
                           'f_comp': f_gas}

    gas_kwargs = tools.merge_dicts(prof_gas_kwargs, comp_gas_kwargs)

    comp_gas = comp.Component(**gas_kwargs)
    return comp_gas

# ------------------------------------------------------------------------------
# End of load_gas()
# ------------------------------------------------------------------------------

def load_gas_obs(prms=p.prms, q_f=50, q_rc=50, q_beta=50):
    '''
    Return beta profiles with fgas_500c = f_obs which only reach up until r500c

    This one is to be used with a profile that matches f_b at r200m, since we
    do not correct from m_bar to m_dmo

    Parameters
    ----------
    prms : p.Parameters object
      contains relevant model info
    q_f : float
      percentile for fgas-m500 relation fit
    q_rc : float
      percentile for rc-m500 relation fit
    q_beta : float
      percentile for beta-m500 relation fit
    '''
    rc, beta = d.fit_prms(x=500, q_rc=q_rc, q_beta=q_beta)

    # halo model parameters
    m200m = prms.m200m
    r200m = prms.r200m
    m500c = prms.m500c
    r500c = prms.r500c

    # radius in terms of r500c
    r_range = prms.r_range_lin
    rx = r_range / r500c.reshape(-1,1)

    # gas fractions
    f_prms = d.f_gas_prms_debiased(prms)
    f_gas500 = d.f_gas(m500c, prms=prms, **f_prms)

    prof_gas = np.zeros_like(rx)
    for idx, prof in enumerate(prof_gas):
        sl = (rx[idx] <= 1.)
        prof_gas[idx][sl] = prof_beta(rx[idx][sl], sl[sl], rc, beta,
                                         f_gas500[idx] * m500c[idx],
                                         r500c[idx])

    mgas200 = tools.m_h(prof_gas, prms.r_range_lin)
    f_gas = mgas200 / m200m

    profile_kwargs = {'r_range': prms.r_range_lin,
                      'm_bar': prms.m200m,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}
    # --------------------------------------------------------------------------
    # specific gas extra kwargs -> need f_gas
    gas_extra = {'profile': prof_gas / f_gas.reshape(-1,1),}
    prof_gas_kwargs = tools.merge_dicts(profile_kwargs, gas_extra)
    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    comp_gas_kwargs = {'name': 'gas',
                       'r200m': r200m,
                       'm200m': m200m,
                       'p_lin': prms.p_lin,
                       'dndm': prms.dndm,
                       'f_comp': f_gas}

    gas_kwargs = tools.merge_dicts(prof_gas_kwargs, comp_gas_kwargs)

    comp_gas = comp.Component(**gas_kwargs)
    return comp_gas

# ------------------------------------------------------------------------------
# End of load_gas_obs()
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
    # halo model parameters
    m200m = prms.m200m
    m500c = prms.m500c
    r500c = prms.r500c
    r200m = prms.r200m

    # will need to fill exact r500c values in r_range for uniform profile to match
    r500_in_range = np.zeros_like(m200m)

    r_range = prms.r_range_lin
    rx = r_range / r500c.reshape(-1,1)

    # gas fractions
    f_b = 1 - prms.f_dm
    # relative position of virial radius
    x200m = r200m / r500c

    prof_gas = np.zeros_like(rx)
    prof_gas_f = np.zeros_like(rx)
    for idx, prof in enumerate(prof_gas):
        sl = (rx[idx] >= 1.)
        r500_in_range[idx] = r_range[idx][sl.nonzero()[0][0]]
        prof_gas[idx][sl] = 1.
        mass = tools.m_h(prof_gas[idx], r_range[idx])
        prof_gas[idx] *= (f_b - fgas_200[idx] - f_stars[idx]) * m200m[idx] / mass
        prof_gas_f[idx] = profs.profile_uniform_f(prms.k_range_lin,
                                                  r500_in_range[idx],
                                                  r200m[idx])

    mgas = tools.m_h(prof_gas, r_range)
    f_gas = mgas / (m200m)

    profile_kwargs = {'r_range': r_range,
                      'm_bar': prms.m200m,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}

    # --------------------------------------------------------------------------
    # specific gas extra kwargs -> need f_gas
    gas_extra = {'profile': prof_gas / f_gas.reshape(-1,1),
                 'profile_f': prof_gas_f}
    prof_gas_kwargs = tools.merge_dicts(profile_kwargs, gas_extra)
    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    comp_gas_kwargs = {'name': 'smooth',
                       'r200m': r200m,
                       'm200m': m200m,
                       'p_lin': prms.p_lin,
                       'dndm': prms.dndm,
                       'f_comp': f_gas}

    gas_kwargs = tools.merge_dicts(prof_gas_kwargs, comp_gas_kwargs)

    comp_gas = comp.Component(**gas_kwargs)
    return comp_gas

# ------------------------------------------------------------------------------
# End of load_gas_smooth_r500c_r200m()
# ------------------------------------------------------------------------------

def load_gas_rmax(r_max, f_stars, prms=p.prms, q_f=50, q_rc=50, q_beta=50,
                  bar2dmo=True):
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
    # halo model parameters
    # ! WATCH OUT ! These are the SPH equivalent masses and do thus not
    # correspond to the DMO case, unless f_gas,200m = f_b
    m200m = prms.m200m
    m500c = prms.m500c
    r500c = prms.r500c
    r200m = prms.r200m

    r_min = prms.r_range_lin[:,0]
    r_range = np.array([np.logspace(np.log10(r_min[i]), np.log10(rm), prms.r_bins)
                        for i,rm in enumerate(r_max)])
    rx = r_range / r500c.reshape(-1,1)

    rc, beta = d.fit_prms(x=500, q_rc=q_rc, q_beta=q_beta)
    # gas fractions
    f_prms = d.f_gas_prms_debiased(prms)
    f_gas500 = d.f_gas(m500c, prms=prms, **f_prms)

    # determine the radius at which the beta profile mass exceeds the
    # baryon fraction
    r_cut = tools.r_where_m_beta((prms.f_b - f_stars) * m200m, beta, rc,
                                 f_gas500 * m500c, r500c)

    # relative position of virial radius
    x200m = r200m / r500c

    prof_gas = np.zeros_like(rx)
    for idx, prof in enumerate(prof_gas):
        sl = (rx[idx] <= x200m[idx])
        sl_500 = (rx[idx] <= 1.)
        sl_fb = (rx[idx] >= r_cut[idx] / r500c[idx])
        prof_gas[idx][sl] = prof_beta(rx[idx], sl_500, rc, beta,
                                         f_gas500[idx] * m500c[idx],
                                         r500c[idx])[sl]
        prof_gas[idx][sl_fb] = 0.

    # can integrate entire profile, since it's zero for r>r200m
    mgas = tools.m_h(prof_gas, r_range)
    f_gas = mgas / (m200m)

    # Now we can determine the equivalent DMO masses from the f_gas - m relation
    m_dmo = d.m200b_to_m200dmo(m200m, f_gas + f_stars, prms)
    r_dmo = tools.mass_to_radius(m_dmo, 200 * prms.rho_m)

    # # renormalize radial range
    # r_range_dmo = r_range * r_dmo.reshape(-1,1) / r200m.reshape(-1,1)
    # # ! This is not necessary, since the radial range is not required
    # # ! in the halo model, except for the density profiles which are fit
    # # ! in the observations and thus do not need to be renormalized

    # Can give the profile all of the halo model parameters, since it is
    # a fit to observations, only the power spectrum calculation needs
    # to convert these values to the DMO equivalent cases
    profile_kwargs = {'r_range': r_range,
                      'm_bar': m200m,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}
    # --------------------------------------------------------------------------
    # specific gas extra kwargs -> need f_gas
    gas_extra = {'profile': prof_gas / f_gas.reshape(-1,1)}
    prof_gas_kwargs = tools.merge_dicts(profile_kwargs, gas_extra)
    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    if bar2dmo == False:
        comp_gas_kwargs = {'name': 'gas',
                           'r200m': r200m,
                           'm200m': m200m,
                           'p_lin': prms.p_lin,
                           'dndm': prms.dndm,
                           'f_comp': f_gas}
    else:
        prms_dmo = deepcopy(prms)
        prms_dmo.m200m = m_dmo
        comp_gas_kwargs = {'name': 'gas',
                           'r200m': r200m,
                           'm200m': m200m,
                           'p_lin': prms.p_lin,
                           'dndm': prms_dmo.dndm,
                           'f_comp': f_gas}

    gas_kwargs = tools.merge_dicts(prof_gas_kwargs, comp_gas_kwargs)

    comp_gas = comp.Component(**gas_kwargs)
    return comp_gas

# ------------------------------------------------------------------------------
# End of load_gas_rmax()
# ------------------------------------------------------------------------------

def load_gas_r500c_r200m_rmax(r_max, f_stars, prms=p.prms, q_f=50, q_rc=50, q_beta=50,
                              bar2dmo=True):
    '''
    Return beta profiles with fgas_200m = f_obs = fgas_5r500c, so 0 from r200m
    to 5r500c

    Parameters
    ----------
    r_max : (m,) array
      maximum radius for each halo mass to compute profile up to
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
    # halo model parameters
    # ! WATCH OUT ! These are the SPH equivalent masses and do thus not
    # correspond to the DMO case, unless f_gas,200m = f_b
    m200m = prms.m200m
    m500c = prms.m500c
    r500c = prms.r500c
    r200m = prms.r200m

    r_min = prms.r_range_lin[:,0]
    r_range = np.array([np.logspace(np.log10(r_min[i]), np.log10(rm), prms.r_bins)
                        for i,rm in enumerate(r_max)])
    rx = r_range / r500c.reshape(-1,1)

    # # will need to fill exact r500c values in r_range for uniform profile to match
    # r500_in_range = np.zeros_like(m200m)

    rc, beta = d.fit_prms(x=500, q_rc=q_rc, q_beta=q_beta)
    # gas fractions
    f_prms = d.f_gas_prms_debiased(prms)
    f_gas500 = d.f_gas(m500c, prms=prms, **f_prms)
    f_b = 1 - prms.f_dm

    # determine the radius at which the beta profile mass exceeds the
    # baryon fraction
    r_cut = tools.r_where_m_beta((prms.f_b - f_stars) * m200m, beta, rc,
                                 f_gas500 * m500c, r500c)

    # relative position of virial radius
    x200m = r200m / r500c

    prof_gas = np.zeros_like(rx)
    for idx, prof in enumerate(prof_gas):
        # radial range between r500c and r200m
        sl_500_200 = ((rx[idx] >= 1.) & (rx[idx] <= x200m[idx]))
        sl_gt200 = (rx[idx] >= x200m[idx])
        sl_gt500 = (rx[idx] >= 1.)
        # radial range up to r500c
        sl_500 = (rx[idx] <= 1.)

        # # fill exact value
        # r500_in_range[idx] = r_range[idx][sl_gt500.nonzero()[0][0]]

        # use beta profile up to r500c
        prof_gas[idx][sl_500] = prof_beta(rx[idx], sl_500, rc, beta,
                                             f_gas500[idx] * m500c[idx],
                                             r500c[idx])[sl_500]

        m_gas500 = tools.m_h(prof_gas[idx], r_range[idx])
        # put remaining mass in smooth component outside r500c up to r200m
        prof_gas[idx][sl_gt500] = 1.
        mass_gt500 = tools.m_h(prof_gas[idx][sl_gt500], r_range[idx][sl_gt500])
        prof_gas[idx][sl_gt500] *= ((f_b - f_stars[idx] - m_gas500 / m200m[idx]) *
                                    m200m[idx] / mass_gt500)
        prof_gas[idx][sl_gt200] = 0.

    # can integrate entire profile, since it's zero for r>r200m
    mgas = tools.m_h(prof_gas, r_range)
    f_gas = mgas / (m200m)

    # Now we can determine the equivalent DMO masses from the f_gas - m relation
    m_dmo = d.m200b_to_m200dmo(m200m, f_gas + f_stars, prms)
    r_dmo = tools.mass_to_radius(m_dmo, 200 * prms.rho_m)

    # # renormalize radial range
    # r_range_dmo = r_range * r_dmo.reshape(-1,1) / r200m.reshape(-1,1)
    # # ! This is not necessary, since the radial range is not required
    # # ! in the halo model, except for the density profiles which are fit
    # # ! in the observations and thus do not need to be renormalized

    # Can give the profile all of the halo model parameters, since it is
    # a fit to observations, only the power spectrum calculation needs
    # to convert these values to the DMO equivalent cases
    profile_kwargs = {'r_range': r_range,
                      'm_bar': m200m,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}
    # --------------------------------------------------------------------------
    # specific gas extra kwargs -> need f_gas
    gas_extra = {'profile': prof_gas / f_gas.reshape(-1,1)}
    prof_gas_kwargs = tools.merge_dicts(profile_kwargs, gas_extra)
    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    if bar2dmo == False:
        comp_gas_kwargs = {'name': 'gas',
                           'r200m': r200m,
                           'm200m': m200m,
                           'p_lin': prms.p_lin,
                           'dndm': prms.dndm,
                           'f_comp': f_gas}
    else:
        prms_dmo = deepcopy(prms)
        prms_dmo.m200m = m_dmo
        comp_gas_kwargs = {'name': 'gas',
                           'r200m': r200m,
                           'm200m': m200m,
                           'p_lin': prms.p_lin,
                           'dndm': prms_dmo.dndm,
                           'f_comp': f_gas}

    gas_kwargs = tools.merge_dicts(prof_gas_kwargs, comp_gas_kwargs)

    comp_gas = comp.Component(**gas_kwargs)
    return comp_gas

# ------------------------------------------------------------------------------
# End of load_gas_r500c_r200m_rmax()
# ------------------------------------------------------------------------------

def load_gas_smooth_r200m_rmax(r_max, m_dmo, prms, fgas_200, f_stars, bar2dmo=True):
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
    # halo model parameters
    # ! WATCH OUT ! These are the SPH equivalent masses and do thus not
    # correspond to the DMO case, unless f_gas,200m = f_b
    m200m = prms.m200m
    m500c = prms.m500c
    r500c = prms.r500c
    r200m = prms.r200m

    r_min = prms.r_range_lin[:,0]
    r_range = np.array([np.logspace(np.log10(r_min[i]), np.log10(rm), prms.r_bins)
                        for i,rm in enumerate(r_max)])
    rx = r_range / r500c.reshape(-1,1)


    # gas fractions
    f_b = 1 - prms.f_dm
    # relative position of virial radius
    x200m = r200m / r500c

    prof_gas = np.zeros_like(rx)
    prof_gas_f = np.zeros_like(rx)
    for idx, prof in enumerate(prof_gas):
        sl = (rx[idx] >= x200m[idx])
        prof_gas[idx][sl] = 1.
        mass = tools.m_h(prof_gas[idx], r_range[idx])
        prof_gas[idx] *= (f_b - f_stars[idx] - fgas_200[idx]) * m200m[idx] / mass
        prof_gas_f[idx] = profs.profile_uniform_f(prms.k_range_lin,
                                                  r200m[idx], r_max[idx])

    mgas = tools.m_h(prof_gas, r_range)
    f_gas = mgas / (m200m)

    # Now we can determine the equivalent DMO masses from the f_gas - m relation
    # m_dmo needs to be given from the profile up to r200m
    r_dmo = tools.mass_to_radius(m_dmo, 200 * prms.rho_m)

    # # renormalize radial range
    # r_range_dmo = r_range * r_dmo.reshape(-1,1) / r200m.reshape(-1,1)
    # # ! This is not necessary, since the radial range is not required
    # # ! in the halo model, except for the density profiles which are fit
    # # ! in the observations and thus do not need to be renormalized

    # Can give the profile all of the halo model parameters, since it is
    # a fit to observations, only the power spectrum calculation needs
    # to convert these values to the DMO equivalent cases
    profile_kwargs = {'r_range': r_range,
                      'm_bar': m200m,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}
    # --------------------------------------------------------------------------
    # specific gas extra kwargs -> need f_gas
    gas_extra = {'profile': prof_gas / f_gas.reshape(-1,1),
                 'profile_f': prof_gas_f}
    prof_gas_kwargs = tools.merge_dicts(profile_kwargs, gas_extra)
    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    if bar2dmo == False:
        comp_gas_kwargs = {'name': 'smooth',
                           'r200m': r200m,
                           'm200m': m200m,
                           'p_lin': prms.p_lin,
                           'dndm': prms.dndm,
                           'f_comp': f_gas}
    else:
        prms_dmo = deepcopy(prms)
        prms_dmo.m200m = m_dmo
        comp_gas_kwargs = {'name': 'smooth',
                           'r200m': r200m,
                           'm200m': m200m,
                           'p_lin': prms.p_lin,
                           'dndm': prms_dmo.dndm,
                           'f_comp': f_gas}

    gas_kwargs = tools.merge_dicts(prof_gas_kwargs, comp_gas_kwargs)

    comp_gas = comp.Component(**gas_kwargs)
    return comp_gas

# ------------------------------------------------------------------------------
# End of load_gas_smooth_r200m_5r500c()
# ------------------------------------------------------------------------------

def prof_delta(r_range, m):
    '''
    Returns a delta function profile
    '''
    profile = np.zeros_like(r_range, dtype=float)
    profile[...,0] = 1.
    profile *= m

    return profile

def prof_delta_f(k_range):
    '''
    Return normalized Fourier transform of delta profile
    '''
    profile = np.ones_like(k_range, dtype=float)

    return profile

def load_centrals(f_gas, prms=p.prms, bar2dmo=True, f_comp='cen'):
    '''
    Return delta profiles with fstars_500c = f_obs

    Parameters
    ----------
    prms : p.Parameters object
      contains relevant model info
    bar2dmo : bool
      specifies whether to carry out hmf conversion for missing m200m
    '''
    # halo model parameters
    # ! WATCH OUT ! These are the SPH equivalent masses and do thus not
    # correspond to the DMO case, unless f_gas,200m + f_stars,200m = f_b
    m200m = prms.m200m
    m500c = prms.m500c
    r500c = prms.r500c
    r200m = prms.r200m

    r_range = prms.r_range_lin
    k_range = prms.k_range_lin

    # stellar fraction
    f_stars = d.f_stars(m200m, comp='all')
    f_cen = d.f_stars(m200m, comp=f_comp)

    prof_stars = np.zeros_like(r_range)
    prof_stars_f = np.zeros(m200m.shape + k_range.shape)
    for idx, prof in enumerate(prof_stars):
        prof_stars[idx] = prof_delta(r_range[idx], m200m[idx])
        prof_stars_f[idx] = prof_delta_f(k_range)

    # Now we can determine the equivalent DMO masses from the f_gas - m relation
    m_dmo = d.m200b_to_m200dmo(m200m, f_gas + f_stars, prms)
    r_dmo = tools.mass_to_radius(m_dmo, 200 * prms.rho_m)

    # # renormalize radial range
    # r_range_dmo = r_range * r_dmo.reshape(-1,1) / r200m.reshape(-1,1)
    # # ! This is not necessary, since the radial range is not required
    # # ! in the halo model, except for the density profiles which are fit
    # # ! in the observations and thus do not need to be renormalized

    # Can give the profile all of the halo model parameters, since it is
    # a fit to observations, only the power spectrum calculation needs
    # to convert these values to the DMO equivalent cases
    profile_kwargs = {'r_range': r_range,
                      'm_bar': m200m,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}
    # --------------------------------------------------------------------------
    # specific gas extra kwargs -> need f_gas
    stars_extra = {'profile': prof_stars,
                   'profile_f': prof_stars_f}
    prof_stars_kwargs = tools.merge_dicts(profile_kwargs, stars_extra)
    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    if bar2dmo == False:
        comp_stars_kwargs = {'name': 'cen',
                             'r200m': r200m,
                             'm200m': m200m,
                             'p_lin': prms.p_lin,
                             'dndm': prms.dndm,
                             'f_comp': f_cen}
    else:
        prms_dmo = deepcopy(prms)
        prms_dmo.m200m = m_dmo
        comp_stars_kwargs = {'name': 'cen',
                             'r200m': r200m,
                             'm200m': m200m,
                             'p_lin': prms.p_lin,
                             'dndm': prms_dmo.dndm,
                             'f_comp': f_cen}

    stars_kwargs = tools.merge_dicts(prof_stars_kwargs, comp_stars_kwargs)

    comp_stars = comp.Component(**stars_kwargs)
    return comp_stars

# ------------------------------------------------------------------------------
# End of load_centrals()
# ------------------------------------------------------------------------------

def load_satellites(f_gas, f_c=0.86, prms=p.prms, bar2dmo=True):
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
    # halo model parameters
    # ! WATCH OUT ! These are the SPH equivalent masses and do thus not
    # correspond to the DMO case, unless f_gas,200m + f_stars,200m = f_b
    m200m = prms.m200m
    m500c = prms.m500c
    r500c = prms.r500c
    r200m = prms.r200m

    r_range = prms.r_range_lin
    k_range = prms.k_range_lin

    # stellar fraction
    f_sat = d.f_stars(m200m, comp='sat')
    f_stars = d.f_stars(m200m, comp='all')
    c = f_c * prms.c_correa

    # Can give the profile all of the halo model parameters, since it is
    # a fit to observations, only the power spectrum calculation needs
    # to convert these values to the DMO equivalent cases
    profile_kwargs = {'r_range': prms.r_range_lin,
                      'm_bar': prms.m200m,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}

    # specific stars extra kwargs
    stars_extra = {'profile': profs.profile_NFW,
                   'profile_f': profs.profile_NFW_f,
                   'profile_args': {'c_x': c,
                                    'r_x': r200m,
                                    'rho_mean': prms.rho_m},
                   'profile_f_args': {'c_x': c,
                                      'r_x': r200m,
                                      'rho_mean': prms.rho_m},}
    prof_stars_kwargs = tools.merge_dicts(profile_kwargs, stars_extra)

    # Now we can determine the equivalent DMO masses from the f_gas - m relation
    m_dmo = d.m200b_to_m200dmo(m200m, f_gas + f_stars, prms)
    r_dmo = tools.mass_to_radius(m_dmo, 200 * prms.rho_m)

    # # renormalize radial range
    # r_range_dmo = r_range * r_dmo.reshape(-1,1) / r200m.reshape(-1,1)
    # # ! This is not necessary, since the radial range is not required
    # # ! in the halo model, except for the density profiles which are fit
    # # ! in the observations and thus do not need to be renormalized

    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    if bar2dmo == False:
        comp_stars_kwargs = {'name': 'sat',
                             'r200m': r200m,
                             'm200m': m200m,
                             'p_lin': prms.p_lin,
                             'dndm': prms.dndm,
                             'f_comp': f_sat}
    else:
        prms_dmo = deepcopy(prms)
        prms_dmo.m200m = m_dmo
        comp_stars_kwargs = {'name': 'sat',
                             'r200m': r200m,
                             'm200m': m200m,
                             'p_lin': prms.p_lin,
                             'dndm': prms_dmo.dndm,
                             'f_comp': f_sat}

    stars_kwargs = tools.merge_dicts(prof_stars_kwargs, comp_stars_kwargs)

    comp_stars = comp.Component(**stars_kwargs)
    return comp_stars

# ------------------------------------------------------------------------------
# End of load_satellites()
# ------------------------------------------------------------------------------

def load_centrals_rmax(r_max, f_gas, prms=p.prms, bar2dmo=True, f_comp='cen'):
    '''
    Return delta profiles with fstars_500c = f_obs

    Parameters
    ----------
    prms : p.Parameters object
      contains relevant model info
    bar2dmo : bool
      specifies whether to carry out hmf conversion for missing m200m
    '''
    # halo model parameters
    # ! WATCH OUT ! These are the SPH equivalent masses and do thus not
    # correspond to the DMO case, unless f_gas,200m + f_stars,200m = f_b
    m200m = prms.m200m
    m500c = prms.m500c
    r500c = prms.r500c
    r200m = prms.r200m

    r_min = prms.r_range_lin[:,0]
    r_range = np.array([np.logspace(np.log10(r_min[i]), np.log10(rm), prms.r_bins)
                        for i,rm in enumerate(r_max)])
    rx = r_range / r500c.reshape(-1,1)
    k_range = prms.k_range_lin

    # stellar fraction
    f_stars = d.f_stars(m200m, comp='all')
    # f_cen = d.f_stars(m200m, comp='all')
    f_cen = d.f_stars(m200m, comp=f_comp)

    prof_stars = np.zeros_like(r_range)
    prof_stars_f = np.zeros(m200m.shape + k_range.shape)
    for idx, prof in enumerate(prof_stars):
        prof_stars[idx] = prof_delta(r_range[idx], m200m[idx])
        prof_stars_f[idx] = prof_delta_f(k_range)

    # Now we can determine the equivalent DMO masses from the f_gas - m relation
    m_dmo = d.m200b_to_m200dmo(m200m, f_gas + f_stars, prms)
    r_dmo = tools.mass_to_radius(m_dmo, 200 * prms.rho_m)

    # # renormalize radial range
    # r_range_dmo = r_range * r_dmo.reshape(-1,1) / r200m.reshape(-1,1)
    # # ! This is not necessary, since the radial range is not required
    # # ! in the halo model, except for the density profiles which are fit
    # # ! in the observations and thus do not need to be renormalized

    # Can give the profile all of the halo model parameters, since it is
    # a fit to observations, only the power spectrum calculation needs
    # to convert these values to the DMO equivalent cases
    profile_kwargs = {'r_range': r_range,
                      'm_bar': m200m,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}
    # --------------------------------------------------------------------------
    # specific gas extra kwargs -> need f_gas
    stars_extra = {'profile': prof_stars,
                   'profile_f': prof_stars_f}
    prof_stars_kwargs = tools.merge_dicts(profile_kwargs, stars_extra)
    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    if bar2dmo == False:
        comp_stars_kwargs = {'name': 'cen',
                             'r200m': r200m,
                             'm200m': m200m,
                             'p_lin': prms.p_lin,
                             'dndm': prms.dndm,
                             'f_comp': f_cen}
    else:
        prms_dmo = deepcopy(prms)
        prms_dmo.m200m = m_dmo
        comp_stars_kwargs = {'name': 'cen',
                             'r200m': r200m,
                             'm200m': m200m,
                             'p_lin': prms.p_lin,
                             'dndm': prms_dmo.dndm,
                             'f_comp': f_cen}

    stars_kwargs = tools.merge_dicts(prof_stars_kwargs, comp_stars_kwargs)

    comp_stars = comp.Component(**stars_kwargs)
    return comp_stars

# ------------------------------------------------------------------------------
# End of load_stars_rmax()
# ------------------------------------------------------------------------------

def load_satellites_rmax(r_max, f_gas, f_c=0.86, prms=p.prms, bar2dmo=True):
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
    # halo model parameters
    # ! WATCH OUT ! These are the SPH equivalent masses and do thus not
    # correspond to the DMO case, unless f_gas,200m + f_stars,200m = f_b
    m200m = prms.m200m
    m500c = prms.m500c
    r500c = prms.r500c
    r200m = prms.r200m

    r_min = prms.r_range_lin[:,0]
    r_range = np.array([np.logspace(np.log10(r_min[i]), np.log10(rm), prms.r_bins)
                        for i,rm in enumerate(r_max)])
    rx = r_range / r500c.reshape(-1,1)
    k_range = prms.k_range_lin

    # stellar fraction
    f_sat = d.f_stars(m200m, comp='sat')
    f_stars = d.f_stars(m200m, comp='all')
    c = f_c * prms.c_correa
    r_x = r200m

    # relative position of virial radius
    x200m = r200m / r500c

    prof_stars = np.zeros_like(rx)
    for idx, prof in enumerate(prof_stars):
        sl = (rx[idx] <= x200m[idx])
        prof_stars[idx][sl] = profs.profile_NFW(r_range[idx][sl].reshape(1,-1),
                                                m200m[idx].reshape(1,1),
                                                c[idx], r_x[idx],
                                                prms.rho_m).reshape(-1)

    # Can give the profile all of the halo model parameters, since it is
    # a fit to observations, only the power spectrum calculation needs
    # to convert these values to the DMO equivalent cases
    profile_kwargs = {'r_range': r_range,
                      'm_bar': m200m,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}

    # --------------------------------------------------------------------------
    stars_extra = {'profile': prof_stars}
    prof_stars_kwargs = tools.merge_dicts(profile_kwargs, stars_extra)

    # Now we can determine the equivalent DMO masses from the f_gas - m relation
    m_dmo = d.m200b_to_m200dmo(m200m, f_gas + f_stars, prms)
    r_dmo = tools.mass_to_radius(m_dmo, 200 * prms.rho_m)

    # # renormalize radial range
    # r_range_dmo = r_range * r_dmo.reshape(-1,1) / r200m.reshape(-1,1)
    # # ! This is not necessary, since the radial range is not required
    # # ! in the halo model, except for the density profiles which are fit
    # # ! in the observations and thus do not need to be renormalized

    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    if bar2dmo == False:
        comp_stars_kwargs = {'name': 'sat',
                             'r200m': r200m,
                             'm200m': m200m,
                             'p_lin': prms.p_lin,
                             'dndm': prms.dndm,
                             'f_comp': f_sat}
    else:
        prms_dmo = deepcopy(prms)
        prms_dmo.m200m = m_dmo
        comp_stars_kwargs = {'name': 'sat',
                             'r200m': r200m,
                             'm200m': m200m,
                             'p_lin': prms.p_lin,
                             'dndm': prms_dmo.dndm,
                             'f_comp': f_sat}

    stars_kwargs = tools.merge_dicts(prof_stars_kwargs, comp_stars_kwargs)

    comp_stars = comp.Component(**stars_kwargs)
    return comp_stars

# ------------------------------------------------------------------------------
# End of load_satellites_rmax()
# ------------------------------------------------------------------------------


def load_models(prms=p.prms, q_f=50, q_rc=50, q_beta=50, bar2dmo=True,
                delta=False):
    # load these variables to have optimal order for mass calculations
    m200c = prms.m200c
    m500c = prms.m500c
    c_correa = prms.c_correa

    f_stars = d.f_stars(prms.m200m)
    # load dmo power spectrum
    dm_dmo = load_dm_dmo(prms)
    pow_dm_dmo = power.Power([dm_dmo], name='dmo')

    dm_dmo_5r500c = load_dm_dmo_rmax(5*prms.r500c, prms)
    dm_dmo_1p5r200m = load_dm_dmo_rmax(1.5*prms.r200m, prms)
    pow_dm_dmo_5r500c = power.Power([dm_dmo_5r500c], name='dmo_5r500c')
    pow_dm_dmo_1p5r200m = power.Power([dm_dmo_1p5r200m], name='dmo_1p5r200m')

    # load dm models
    dm = load_dm(m_dmo=prms.m200m, prms=prms, bar2dmo=False)
    dm_5r500c = load_dm_rmax(r_max=5*prms.r500c,m_dmo=prms.m200m,
                               prms=prms, bar2dmo=False)
    dm_1p5r200m = load_dm_rmax(r_max=1.5*prms.r200m, m_dmo=prms.m200m,
                               prms=prms, bar2dmo=False)

    # --------------------------------------------------------------------------
    # MODEL 1
    # --------------------------------------------------------------------------
    # load gas_obs
    gas = load_gas(f_stars, prms, q_f, q_rc, q_beta, bar2dmo=bar2dmo)
    # load stars
    if not delta:
        cen = load_centrals(gas.f_comp, prms, bar2dmo=bar2dmo)
        sat = load_satellites(gas.f_comp, 0.86, prms, bar2dmo=bar2dmo)
        stars = cen + sat
    else:
        stars = load_centrals(gas.f_comp, prms, bar2dmo=bar2dmo, f_comp='all')
    stars.name = 'stars'

    m_dmo_gas = d.m200b_to_m200dmo(gas.m200m, gas.f_comp + f_stars, prms)
    dm_gas = load_dm(m_dmo=m_dmo_gas, prms=prms, bar2dmo=bar2dmo)
    # pow_gas = power.Power([dm_gas, gas, cen], name='model1')
    pow_gas = power.Power([dm_gas, gas, stars], name='model1')

    # --------------------------------------------------------------------------
    # MODEL 2
    # --------------------------------------------------------------------------
    # load gas_smooth_r500c_r200m
    gas_beta = load_gas_obs(prms, q_f, q_rc, q_beta)
    # load stars -> this model has f_gas,200m = f_b - f_stars
    if not delta:
        cen_beta = load_centrals(prms.f_b - f_stars, prms, bar2dmo=bar2dmo)
        sat_beta = load_satellites(prms.f_b - f_stars, 0.86, prms, bar2dmo=bar2dmo)
        stars_beta = cen_beta + sat_beta
    else:
        stars_beta = load_centrals(prms.f_b - f_stars, prms, bar2dmo=bar2dmo, f_comp='all')
    stars_beta.name = 'stars'

    gas_smooth = load_gas_smooth_r500c_r200m(prms, gas_beta.f_comp, f_stars)
    # does not need new dm since fgas_200m = f_b
    # pow_gas_smooth_r500c_r200m = power.Power([dm, gas_beta, gas_smooth,
    #                                           cen_beta],
    #                                          name='model2')
    pow_gas_smooth_r500c_r200m = power.Power([dm, gas_beta, gas_smooth,
                                              stars_beta],
                                             name='model2')

    # --------------------------------------------------------------------------
    # MODEL 3
    # --------------------------------------------------------------------------
    # load_gas_smooth_r500c_5r500c
    gas_r500c_5r500c = load_gas_r500c_r200m_rmax(5*prms.r500c,f_stars, prms,
                                                 q_f, q_rc, q_beta,
                                                 bar2dmo=bar2dmo)
    # load stars
    if not delta:
        cen_r500c_5r500c = load_centrals_rmax(5*prms.r500c, gas_r500c_5r500c.f_comp,
                                              prms, bar2dmo=bar2dmo)
        sat_r500c_5r500c = load_satellites_rmax(5*prms.r500c, gas_r500c_5r500c.f_comp,
                                                0.86, prms, bar2dmo=bar2dmo)
        stars_r500c_5r500c = cen_r500c_5r500c + sat_r500c_5r500c
    else:
        stars_r500c_5r500c = load_centrals_rmax(5*prms.r500c, gas_r500c_5r500c.f_comp,
                                                prms, bar2dmo=bar2dmo, f_comp='all')
    stars_r500c_5r500c.name = 'stars'

    m_dmo_r500c_5r500c = d.m200b_to_m200dmo(gas_r500c_5r500c.m200m,
                                            gas_r500c_5r500c.f_comp + f_stars, prms)
    gas_smooth_r200m_5r500c = load_gas_smooth_r200m_rmax(5*prms.r500c,
                                                         m_dmo_r500c_5r500c,
                                                         prms,
                                                         gas_r500c_5r500c.f_comp,
                                                         f_stars,
                                                         bar2dmo=bar2dmo)
    dm_r500c_5r500c = load_dm_rmax(5*prms.r500c, m_dmo_r500c_5r500c,
                                   prms, bar2dmo=bar2dmo)
    # pow_gas_smooth_r500c_5r500c = power.Power([dm_r500c_5r500c,
    #                                            gas_r500c_5r500c,
    #                                            gas_smooth_r200m_5r500c,
    #                                            cen_r500c_5r500c],
    #                                           name='model3')
    pow_gas_smooth_r500c_5r500c = power.Power([dm_r500c_5r500c,
                                               gas_r500c_5r500c,
                                               gas_smooth_r200m_5r500c,
                                               stars_r500c_5r500c],
                                              name='model3')

    # --------------------------------------------------------------------------
    # MODEL 4
    # --------------------------------------------------------------------------
    # load gas_smooth_r200m_5r500c
    gas_5r500c = load_gas_rmax(5*prms.r500c, f_stars, prms, q_f, q_rc, q_beta,
                               bar2dmo=bar2dmo)
    # load stars
    if not delta:
        cen_5r500c = load_centrals_rmax(5*prms.r500c,gas_5r500c.f_comp, prms,
                                        bar2dmo=bar2dmo)
        sat_5r500c = load_satellites_rmax(5*prms.r500c,gas_5r500c.f_comp, 0.86,
                                          prms, bar2dmo=bar2dmo)
        stars_5r500c = cen_5r500c + sat_5r500c
    else:
        stars_5r500c = load_centrals_rmax(5*prms.r500c,gas_5r500c.f_comp, prms,
                                          bar2dmo=bar2dmo, f_comp='all')

    stars_5r500c.name = 'stars'
    m_dmo_5r500c = d.m200b_to_m200dmo(gas_5r500c.m200m, gas_5r500c.f_comp + f_stars,
                                      prms)
    gas_smooth_r200m_5r500c = load_gas_smooth_r200m_rmax(5*prms.r500c, m_dmo_5r500c,
                                                         prms,
                                                         gas_5r500c.f_comp,
                                                         f_stars,
                                                         bar2dmo=bar2dmo)
    dm_r200m_5r500c = load_dm_rmax(5*prms.r500c, m_dmo_5r500c, prms,
                                   bar2dmo=bar2dmo)
    # pow_gas_smooth_r200m_5r500c = power.Power([dm_r200m_5r500c, gas_5r500c,
    #                                            gas_smooth_r200m_5r500c,
    #                                            cen_5r500c],
    #                                           name='model4')
    pow_gas_smooth_r200m_5r500c = power.Power([dm_r200m_5r500c, gas_5r500c,
                                               gas_smooth_r200m_5r500c,
                                               stars_5r500c],
                                              name='model4')

    # # --------------------------------------------------------------------------
    # # MODEL 5
    # # --------------------------------------------------------------------------
    # # load gas_smooth_r200m_1p5r200m
    # gas_1p5r200m = load_gas_rmax(1.5*prms.r200m, f_stars, prms, q_f, q_rc, q_beta,
    #                              bar2dmo=bar2dmo)
    # # load stars
    # cen_1p5r200m = load_centrals_rmax(1.5*prms.r200m, gas_1p5r200m.f_comp, prms,
    #                                   bar2dmo=bar2dmo)
    # sat_1p5r200m = load_satellites_rmax(1.5*prms.r200m, gas_1p5r200m.f_comp, 0.86, prms,
    #                                     bar2dmo=bar2dmo)
    # m_dmo_1p5r200m = d.m200b_to_m200dmo(gas_1p5r200m.m200m,
    #                                     gas_1p5r200m.f_comp + f_stars, prms)
    # gas_smooth_r200m_1p5r200m = load_gas_smooth_r200m_rmax(1.5*prms.r200m,
    #                                                        m_dmo_1p5r200m, prms,
    #                                                        gas_1p5r200m.f_comp,
    #                                                        f_stars,
    #                                                        bar2dmo=bar2dmo)
    # dm_r200m_1p5r200m = load_dm_rmax(1.5*prms.r200m, m_dmo_1p5r200m,
    #                                  prms, bar2dmo=bar2dmo)
    # pow_gas_smooth_r200m_1p5r200m = power.Power([dm_r200m_1p5r200m, gas_1p5r200m,
    #                                              gas_smooth_r200m_1p5r200m,
    #                                              cen_1p5r200m, sat_1p5r200m],
    #                                             name='model5')

    results = {'dm_dmo': pow_dm_dmo,
               'dm_dmo_5r500c': pow_dm_dmo_5r500c,
               'dm_dmo_1p5r200m': pow_dm_dmo_1p5r200m,
               'gas': pow_gas,
               'smooth_r500c_r200m': pow_gas_smooth_r500c_r200m,
               'smooth_r200m_5r500c': pow_gas_smooth_r200m_5r500c,
               # 'smooth_r200m_1p5r200m': pow_gas_smooth_r200m_1p5r200m,
               'smooth_r500c_5r500c': pow_gas_smooth_r500c_5r500c}

    return results

# ------------------------------------------------------------------------------
# End of load_models()
# ------------------------------------------------------------------------------
