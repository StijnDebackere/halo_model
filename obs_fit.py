import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter

import sys
import cPickle
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
    # c200m = prms.c_correa
    c200m = b.dm_c_dmo(m200m * prms.h)
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

def load_dm_dmo_5r500c(prms=p.prms):
    '''
    Pure dark matter only component with NFW profile and f_dm = 1
    '''
    m200m = prms.m200m
    m500c = prms.m500c
    r500c = prms.r500c
    r200m = prms.r200m

    r_min = prms.r_range_lin[:,0]
    r_max = (5 * r500c)
    r_range = np.array([np.logspace(np.log10(r_min[i]), np.log10(rm), prms.r_bins)
                        for i,rm in enumerate(r_max)])
    rx = r_range / r500c.reshape(-1,1)


    f_dm = np.ones_like(m200m)
    # c200m = prms.c_correa
    c200m = b.dm_c_dmo(m200m * prms.h)

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
    dm_extra = {'profile': prof_dm}
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
# End of load_dm_dmo_5r500c()
# ------------------------------------------------------------------------------

def load_dm_5r500c(m_dmo, prms=p.prms, bar2dmo=True):
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
    r_max = (5 * r500c)
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
    dm_extra = {'profile': prof_dm}
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

def prof_beta(r, rc, beta, m200):
    prof = 1. / (1 + (r/rc)**2)**(beta/2)
    norm = tools.m_h(prof, r)
    prof *= m200 / norm
    return prof

def prof_gas_hot(x, sl, a, b, m_sl, r500):
    '''beta profile'''
    profile = (1 + (x/a)**2)**(-b/2)
    mass = tools.m_h(profile[sl], x[sl] * r500)
    profile *= m_sl/mass

    return profile

def load_gas(prms=p.prms, q_f=50, q_rc=50, q_beta=50, bar2dmo=True):
    '''
    Return beta profiles with fgas_500c = f_obs, extrapolated to r200m

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
    f_prms = d.f_gas_prms(prms, q=q_f)
    f_gas500 = d.f_gas(m500c, prms=prms, **f_prms)

    prof_gas = np.zeros_like(rx)
    for idx, prof in enumerate(prof_gas):
        sl = (rx[idx] <= 1.)
        prof_gas[idx] = prof_gas_hot(rx[idx], sl, rc, beta,
                                     f_gas500[idx] * m500c[idx],
                                     r500c[idx])

    mgas200 = tools.m_h(prof_gas, prms.r_range_lin)
    f_gas = mgas200 / m200m

    # Now we can determine the equivalent DMO masses from the f_gas - m relation
    m_dmo = d.m200b_to_m200dmo(m200m, f_gas, prms)
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
    f_prms = d.f_gas_prms(prms, q=q_f)
    f_gas500 = d.f_gas(m500c, prms=prms, **f_prms)

    prof_gas = np.zeros_like(rx)
    for idx, prof in enumerate(prof_gas):
        sl = (rx[idx] <= 1.)
        prof_gas[idx][sl] = prof_gas_hot(rx[idx][sl], sl[sl], rc, beta,
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

def load_gas_smooth_r500c_r200m(prms, fgas_200):
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

    r_range = prms.r_range_lin
    rx = r_range / r500c.reshape(-1,1)

    # gas fractions
    f_b = 1 - prms.f_dm
    # relative position of virial radius
    x200m = r200m / r500c

    prof_gas = np.zeros_like(rx)
    for idx, prof in enumerate(prof_gas):
        sl = (rx[idx] >= 1.)
        prof_gas[idx][sl] = 1.
        mass = tools.m_h(prof_gas[idx], r_range[idx])
        prof_gas[idx] *= (f_b - fgas_200[idx]) * m200m[idx] / mass

    mgas = tools.m_h(prof_gas, r_range)
    f_gas = mgas / (m200m)

    profile_kwargs = {'r_range': r_range,
                      'm_bar': prms.m200m,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}
    # --------------------------------------------------------------------------
    # specific gas extra kwargs -> need f_gas
    gas_extra = {'profile': prof_gas / f_gas.reshape(-1,1)}
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

def load_gas_5r500c(prms=p.prms, q_f=50, q_rc=50, q_beta=50, bar2dmo=True):
    '''
    Return beta profiles with fgas_200m = f_obs_extrapolated = fgas_5r500c

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
    r_max = (5 * r500c)
    r_range = np.array([np.logspace(np.log10(r_min[i]), np.log10(rm), prms.r_bins)
                        for i,rm in enumerate(r_max)])
    rx = r_range / r500c.reshape(-1,1)

    rc, beta = d.fit_prms(x=500, q_rc=q_rc, q_beta=q_beta)
    # gas fractions
    f_prms = d.f_gas_prms(prms, q=q_f)
    f_gas500 = d.f_gas(m500c, prms=prms, **f_prms)

    # relative position of virial radius
    x200m = r200m / r500c

    prof_gas = np.zeros_like(rx)
    for idx, prof in enumerate(prof_gas):
        sl = (rx[idx] <= x200m[idx])
        sl_500 = (rx[idx] <= 1.)
        prof_gas[idx][sl] = prof_gas_hot(rx[idx], sl_500, rc, beta,
                                         f_gas500[idx] * m500c[idx],
                                         r500c[idx])[sl]

    # can integrate entire profile, since it's zero for r>r200m
    mgas = tools.m_h(prof_gas, r_range)
    f_gas = mgas / (m200m)

    # Now we can determine the equivalent DMO masses from the f_gas - m relation
    m_dmo = d.m200b_to_m200dmo(m200m, f_gas, prms)
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
# End of load_gas_5r500c()
# ------------------------------------------------------------------------------

def load_gas_r500c_r200m_5r500c(prms=p.prms, q_f=50, q_rc=50, q_beta=50,
                                bar2dmo=True):
    '''
    Return beta profiles with fgas_200m = f_obs = fgas_5r500c

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
    r_max = (5 * r500c)
    r_range = np.array([np.logspace(np.log10(r_min[i]), np.log10(rm), prms.r_bins)
                        for i,rm in enumerate(r_max)])
    rx = r_range / r500c.reshape(-1,1)

    rc, beta = d.fit_prms(x=500, q_rc=q_rc, q_beta=q_beta)
    # gas fractions
    f_prms = d.f_gas_prms(prms, q=q_f)
    f_gas500 = d.f_gas(m500c, prms=prms, **f_prms)
    f_b = 1 - prms.f_dm

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

        # use beta profile up to r500c
        prof_gas[idx][sl_500] = prof_gas_hot(rx[idx], sl_500, rc, beta,
                                             f_gas500[idx] * m500c[idx],
                                             r500c[idx])[sl_500]

        m_gas500 = tools.m_h(prof_gas[idx], r_range[idx])
        # put remaining mass in smooth component outside r500c up to r200m
        prof_gas[idx][sl_gt500] = 1.
        mass_gt500 = tools.m_h(prof_gas[idx][sl_gt500], r_range[idx][sl_gt500])
        prof_gas[idx][sl_gt500] *= ((f_b - m_gas500 / m200m[idx]) *
                                    m200m[idx] / mass_gt500)
        prof_gas[idx][sl_gt200] = 0.

    # can integrate entire profile, since it's zero for r>r200m
    mgas = tools.m_h(prof_gas, r_range)
    f_gas = mgas / (m200m)

    # Now we can determine the equivalent DMO masses from the f_gas - m relation
    m_dmo = d.m200b_to_m200dmo(m200m, f_gas, prms)
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
# End of load_gas_r500c_r200m_5r500c()
# ------------------------------------------------------------------------------

def load_gas_smooth_r200m_5r500c(m_dmo, prms, fgas_200, bar2dmo=True):
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
    r_max = (5 * r500c)
    r_range = np.array([np.logspace(np.log10(r_min[i]), np.log10(rm), prms.r_bins)
                        for i,rm in enumerate(r_max)])
    rx = r_range / r500c.reshape(-1,1)


    # gas fractions
    f_b = 1 - prms.f_dm
    # relative position of virial radius
    x200m = r200m / r500c

    prof_gas = np.zeros_like(rx)
    for idx, prof in enumerate(prof_gas):
        sl = (rx[idx] >= x200m[idx])
        prof_gas[idx][sl] = 1.
        mass = tools.m_h(prof_gas[idx], r_range[idx])
        prof_gas[idx] *= (f_b - fgas_200[idx]) * m200m[idx] / mass

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
    gas_extra = {'profile': prof_gas / f_gas.reshape(-1,1)}
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

def load_models(prms=p.prms, q_f=50, q_rc=50, q_beta=50, bar2dmo=True):
    # load dmo power spectrum
    dm_dmo = load_dm_dmo(prms)
    pow_dm_dmo = power.Power([dm_dmo], name='dmo')

    dm_dmo_5r500c = load_dm_dmo_5r500c(prms)
    pow_dm_dmo_5r500c = power.Power([dm_dmo_5r500c], name='dmo_5r500c')

    # load dm models
    dm = load_dm(m_dmo=prms.m200m, prms=prms, bar2dmo=False)
    dm_5r500c = load_dm_5r500c(m_dmo=prms.m200m, prms=prms, bar2dmo=False)

    # load gas_obs
    gas = load_gas(prms, q_f, q_rc, q_beta, bar2dmo=bar2dmo)
    m_dmo_gas = d.m200b_to_m200dmo(gas.m200m, gas.f_comp, prms)
    dm_gas = load_dm(m_dmo=m_dmo_gas, prms=prms, bar2dmo=bar2dmo)
    pow_gas = power.Power([dm_gas, gas], name='gas_extrap')

    # load gas_smooth_r500c_r200m
    gas_beta = load_gas_obs(prms, q_f, q_rc, q_beta)
    gas_smooth = load_gas_smooth_r500c_r200m(prms, gas_beta.f_comp)
    # does not need new dm since fgas_200m = f_b
    pow_gas_smooth_r500c_r200m = power.Power([dm, gas_beta, gas_smooth],
                                             name='gas_smooth_r500c_r200m')

    # load gas_smooth_r200m_5r500c
    gas_5r500c = load_gas_5r500c(prms, q_f, q_rc, q_beta, bar2dmo=bar2dmo)
    m_dmo_5r500c = d.m200b_to_m200dmo(gas_5r500c.m200m, gas_5r500c.f_comp, prms)
    gas_smooth_r200m_5r500c = load_gas_smooth_r200m_5r500c(m_dmo_5r500c, prms,
                                                           gas_5r500c.f_comp,
                                                           bar2dmo=bar2dmo)
    dm_r200m_5r500c = load_dm_5r500c(m_dmo_5r500c, prms, bar2dmo=bar2dmo)
    pow_gas_smooth_r200m_5r500c = power.Power([dm_r200m_5r500c, gas_5r500c,
                                               gas_smooth_r200m_5r500c],
                                              name='gas_smooth_r200m_5r500c')

    # load_gas_smooth_r500c_5r500c
    gas_r500c_5r500c = load_gas_r500c_r200m_5r500c(prms, q_f, q_rc, q_beta,
                                                   bar2dmo=bar2dmo)
    m_dmo_r500c_5r500c = d.m200b_to_m200dmo(gas_r500c_5r500c.m200m,
                                            gas_r500c_5r500c.f_comp, prms)
    gas_smooth_r500c_5r500c = load_gas_smooth_r200m_5r500c(m_dmo_r500c_5r500c,
                                                           prms,
                                                           gas_r500c_5r500c.f_comp,
                                                           bar2dmo=bar2dmo)
    dm_r500c_5r500c = load_dm_5r500c(m_dmo_r500c_5r500c, prms, bar2dmo=bar2dmo)
    pow_gas_smooth_r500c_5r500c = power.Power([dm_r500c_5r500c,
                                               gas_r500c_5r500c,
                                               gas_smooth_r500c_5r500c],
                                              name='gas_smooth_r500c_5r500c')

    results = {'dm_dmo': pow_dm_dmo,
               'dm_dmo_5r500c': pow_dm_dmo_5r500c,
               'gas': pow_gas,
               'smooth_r500c_r200m': pow_gas_smooth_r500c_r200m,
               'smooth_r200m_5r500c': pow_gas_smooth_r200m_5r500c,
               'smooth_r500c_5r500c': pow_gas_smooth_r500c_5r500c}

    return results

# ------------------------------------------------------------------------------
# End of load_models()
# ------------------------------------------------------------------------------

def plot_profiles_gas_paper(comp_gas, comp_gas_r500c_r200m,
                            comp_gas_r500c_5r500c,
                            comp_gas_r200m_5r500c,
                            rho_k=False,
                            prms=p.prms):
    '''
    Plot the density for our different gas profiles in one plot
    '''
    fig = plt.figure(figsize=(30,8))
    ax1 = fig.add_axes([0.1,0.1,0.266,0.8])
    ax2 = fig.add_axes([0.366,0.1,0.266,0.8])
    ax3 = fig.add_axes([0.632,0.1,0.266,0.8])

    idx_1 = 0
    idx_2 = 50
    idx_3 = -1
    r200m = prms.r200m

    pl.set_style('line')

    if not rho_k:
        norm = prms.rho_crit
        # Plot idx_1
        # ax1.plot(comp_gas_dmo.r_range[idx_1] / r200m[idx_1],
        #          comp_gas_dmo.rho_r[idx_1] * comp_gas_dmo.f_comp[idx_1] / norm,
        #          label=r'$f_\mathrm{gas,500c}=f_\mathrm{gas,200m}=f_\mathrm{b}$')
        ax1.plot(comp_gas.r_range[idx_1] / r200m[idx_1],
                 comp_gas.rho_r[idx_1] * comp_gas.f_comp[idx_1]/norm,
                 lw=3, label=r'model 1')
        ax1.plot(comp_gas_r500c_r200m.r_range[idx_1] / r200m[idx_1],
                 (comp_gas_r500c_r200m.rho_r[idx_1] *
                  comp_gas_r500c_r200m.f_comp[idx_1] / norm),
                 lw=2, label=r'model 2')
        ax1.plot(comp_gas_r500c_5r500c.r_range[idx_1] / r200m[idx_1],
                 (comp_gas_r500c_5r500c.rho_r[idx_1] *
                  comp_gas_r500c_5r500c.f_comp[idx_1] / norm), lw=1,
                 label=r'model 3')
        ax1.plot(comp_gas_r200m_5r500c.r_range[idx_1] / r200m[idx_1],
                 (comp_gas_r200m_5r500c.rho_r[idx_1] *
                  comp_gas_r200m_5r500c.f_comp[idx_1] / norm),
                 lw=1, ls='--', label=r'model 4')

        ax1.axvline(x=prms.r500c[idx_1] / prms.r200m[idx_1], ls='--', c='k')
        ax1.text(x=prms.r500c[idx_1] / prms.r200m[idx_1] * 3, y=1e2, s=r'$r_\mathrm{500c}$',
                 ha='center', va='center')

        # Plot idx_2
        # ax2.plot(comp_gas_dmo.r_range[idx_2] / r200m[idx_2],
        #          comp_gas_dmo.rho_r[idx_2] * comp_gas_dmo.f_comp[idx_2] / norm,
        #          label=r'$f_\mathrm{gas,500c}=f_\mathrm{gas,200m}=f_\mathrm{b}$')
        ax2.plot(comp_gas.r_range[idx_2] / r200m[idx_2],
                 comp_gas.rho_r[idx_2] * comp_gas.f_comp[idx_2]/norm,
                 lw=3, label=r'model 1')
        ax2.plot(comp_gas_r500c_r200m.r_range[idx_2] / r200m[idx_2],
                 (comp_gas_r500c_r200m.rho_r[idx_2] *
                  comp_gas_r500c_r200m.f_comp[idx_2] / norm),
                 lw=2, label=r'model 2')
        ax2.plot(comp_gas_r500c_5r500c.r_range[idx_2] / r200m[idx_2],
                 (comp_gas_r500c_5r500c.rho_r[idx_2] *
                  comp_gas_r500c_5r500c.f_comp[idx_2] / norm), lw=1,
                 label=r'model 3')
        ax2.plot(comp_gas_r200m_5r500c.r_range[idx_2] / r200m[idx_2],
                 (comp_gas_r200m_5r500c.rho_r[idx_2] *
                  comp_gas_r200m_5r500c.f_comp[idx_2] / norm),
                 lw=1, ls='--', label=r'model 4')

        ax2.axvline(x=prms.r500c[idx_2] / prms.r200m[idx_2], ls='--', c='k')
        ax2.text(x=prms.r500c[idx_2] / prms.r200m[idx_2] * 3, y=1e2, s=r'$r_\mathrm{500c}$',
                 ha='center', va='center')

        # Plot idx_3
        # ax3.plot(comp_gas_dmo.r_range[idx_3] / r200m[idx_3],
        #          comp_gas_dmo.rho_r[idx_3] * comp_gas_dmo.f_comp[idx_3] / norm,
        #          label=r'$f_\mathrm{gas,500c}=f_\mathrm{gas,200m}=f_\mathrm{b}$')
        ax3.plot(comp_gas.r_range[idx_3] / r200m[idx_3],
                 comp_gas.rho_r[idx_3] * comp_gas.f_comp[idx_3]/norm,
                 lw=3, label=r'model 1')
        ax3.plot(comp_gas_r500c_r200m.r_range[idx_3] / r200m[idx_3],
                 (comp_gas_r500c_r200m.rho_r[idx_3] *
                  comp_gas_r500c_r200m.f_comp[idx_3] / norm),
                 lw=2, label=r'model 2')
        ax3.plot(comp_gas_r500c_5r500c.r_range[idx_3] / r200m[idx_3],
                 (comp_gas_r500c_5r500c.rho_r[idx_3] *
                  comp_gas_r500c_5r500c.f_comp[idx_3] / norm), lw=1,
                 label=r'model 3')
        ax3.plot(comp_gas_r200m_5r500c.r_range[idx_3] / r200m[idx_3],
                 (comp_gas_r200m_5r500c.rho_r[idx_3] *
                  comp_gas_r200m_5r500c.f_comp[idx_3] / norm),
                 lw=1, ls='--', label=r'model 4')

        ax3.axvline(x=prms.r500c[idx_3] / prms.r200m[idx_3], ls='--', c='k')
        ax3.text(x=prms.r500c[idx_3] / prms.r200m[idx_3] * 3, y=1e2, s=r'$r_\mathrm{500c}$',
                 ha='center', va='center')

        ax1.set_xlim(1e-3, 3)
        ax1.set_ylim(1e-1, 1e3)
        ax2.set_xlim(1e-3, 3)
        ax2.set_ylim(1e-1, 1e3)
        ax3.set_xlim(1e-3, 3)
        ax3.set_ylim(1e-1, 1e3)

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

        ax1.set_title(r'$m_{\mathrm{200m}} = 10^{%.1f} \, \mathrm{M_\odot}/h$'%np.log10(prms.m200m[idx_1]), y=1.015, fontsize=28)
        ax2.set_title(r'$m_{\mathrm{200m}} = 10^{%.1f} \, \mathrm{M_\odot}/h$'%np.log10(prms.m200m[idx_2]), y=1.015, fontsize=28)
        ax3.set_title(r'$m_{\mathrm{200m}} = 10^{%.1f} \, \mathrm{M_\odot}/h$'%np.log10(prms.m200m[idx_3]), y=1.015, fontsize=28)

        ax3.legend(loc='best', fontsize=28)
        plt.savefig('obs_rho_extrapolated.pdf', transparent=True)

    else:
        norm = 1.
        # Plot idx_1
        # ax1.plot(comp_gas_dmo.k_range,
        #          comp_gas_dmo.rho_k[idx_1] * comp_gas_dmo.f_comp[idx_1] / norm,
        #          label=r'$f_\mathrm{gas,500c}=f_\mathrm{gas,200m}=f_\mathrm{b}$')
        ax1.plot(comp_gas.k_range,
                 comp_gas.rho_k[idx_1] * comp_gas.f_comp[idx_1]/norm,
                 lw=3, label=r'model 1')
        ax1.plot(comp_gas_r500c_r200m.k_range,
                 (comp_gas_r500c_r200m.rho_k[idx_1] *
                  comp_gas_r500c_r200m.f_comp[idx_1] / norm),
                 lw=2, label=r'model 2')
        ax1.plot(comp_gas_r500c_5r500c.k_range,
                 (comp_gas_r500c_5r500c.rho_k[idx_1] *
                  comp_gas_r500c_5r500c.f_comp[idx_1] / norm), lw=1,
                 label=r'model 3')
        ax1.plot(comp_gas_r200m_5r500c.k_range,
                 (comp_gas_r200m_5r500c.rho_k[idx_1] *
                  comp_gas_r200m_5r500c.f_comp[idx_1] / norm),
                 lw=1, ls='--', label=r'model 4')

        # ax1.axvline(x=prms.r500c[idx_1] / prms.r200m[idx_1], ls='--', c='k')
        # ax1.text(x=prms.r500c[idx_1] / prms.r200m[idx_1] * 3, y=1e2, s=r'$r_\mathrm{500c}$',
        #          ha='center', va='center')

        # Plot idx_2
        # ax2.plot(comp_gas_dmo.k_range,
        #          comp_gas_dmo.rho_k[idx_2] * comp_gas_dmo.f_comp[idx_2] / norm,
        #          label=r'$f_\mathrm{gas,500c}=f_\mathrm{gas,200m}=f_\mathrm{b}$')
        ax2.plot(comp_gas.k_range,
                 comp_gas.rho_k[idx_2] * comp_gas.f_comp[idx_2]/norm,
                 lw=3, label=r'model 1')
        ax2.plot(comp_gas_r500c_r200m.k_range,
                 (comp_gas_r500c_r200m.rho_k[idx_2] *
                  comp_gas_r500c_r200m.f_comp[idx_2] / norm),
                 lw=2, label=r'model 2')
        ax2.plot(comp_gas_r500c_5r500c.k_range,
                 (comp_gas_r500c_5r500c.rho_k[idx_2] *
                  comp_gas_r500c_5r500c.f_comp[idx_2] / norm), lw=1,
                 label=r'model 3')
        ax2.plot(comp_gas_r200m_5r500c.k_range,
                 (comp_gas_r200m_5r500c.rho_k[idx_2] *
                  comp_gas_r200m_5r500c.f_comp[idx_2] / norm),
                 lw=1, ls='--', label=r'model 4')

        # ax2.axvline(x=prms.r500c[idx_2] / prms.r200m[idx_2], ls='--', c='k')
        # ax2.text(x=prms.r500c[idx_2] / prms.r200m[idx_2] * 3, y=1e2, s=r'$r_\mathrm{500c}$',
        #          ha='center', va='center')

        # Plot idx_3
        # ax3.plot(comp_gas_dmo.k_range,
        #          comp_gas_dmo.rho_k[idx_3] * comp_gas_dmo.f_comp[idx_3] / norm,
        #          label=r'$f_\mathrm{gas,500c}=f_\mathrm{gas,200m}=f_\mathrm{b}$')
        ax3.plot(comp_gas.k_range,
                 comp_gas.rho_k[idx_3] * comp_gas.f_comp[idx_3]/norm,
                 lw=3, label=r'model 1')
        ax3.plot(comp_gas_r500c_r200m.k_range,
                 (comp_gas_r500c_r200m.rho_k[idx_3] *
                  comp_gas_r500c_r200m.f_comp[idx_3] / norm),
                 lw=2, label=r'model 2')
        ax3.plot(comp_gas_r500c_5r500c.k_range,
                 (comp_gas_r500c_5r500c.rho_k[idx_3] *
                  comp_gas_r500c_5r500c.f_comp[idx_3] / norm), lw=1,
                 label=r'model 3')
        ax3.plot(comp_gas_r200m_5r500c.k_range,
                 (comp_gas_r200m_5r500c.rho_k[idx_3] *
                  comp_gas_r200m_5r500c.f_comp[idx_3] / norm),
                 lw=1, ls='--', label=r'model 4')

        # ax3.axvline(x=prms.r500c[idx_3] / prms.r200m[idx_3], ls='--', c='k')
        # ax3.text(x=prms.r500c[idx_3] / prms.r200m[idx_3] * 3, y=1e2, s=r'$r_\mathrm{500c}$',
        #          ha='center', va='center')

        # ax1.set_xlim(prms.k_range_lin.min(), prms.k_range_lin.max())
        # ax2.set_xlim(prms.k_range_lin.min(), prms.k_range_lin.max())
        # ax3.set_xlim(prms.k_range_lin.min(), prms.k_range_lin.max())

        ax1.set_xscale('log')
        ax2.set_xscale('log')
        ax3.set_xscale('log')
        ax1.set_yscale('symlog',linthreshy=1e-5)
        ax2.set_yscale('symlog',linthreshy=1e-5)
        ax3.set_yscale('symlog',linthreshy=1e-5)

        ax1.set_xlim(1,100)
        ax2.set_xlim(1,100)
        ax3.set_xlim(1,100)
        ax1.set_ylim(-1, 1)
        ax2.set_ylim(-1, 1)
        ax3.set_ylim(-1, 1)

        # formatter = ScalarFormatter()
        # formatter.set_scientific(False)
        # ax1.xaxis.set_major_formatter(formatter)
        # ax2.xaxis.set_major_formatter(formatter)
        # ax3.xaxis.set_major_formatter(formatter)
        # ax1.xaxis.set_minor_formatter(formatter)
        # ax2.xaxis.set_minor_formatter(formatter)
        # ax3.xaxis.set_minor_formatter(formatter)

        ax1.tick_params(axis='x', which='major', pad=6)
        ax2.tick_params(axis='x', which='major', pad=6)
        ax3.tick_params(axis='x', which='major', pad=6)
        ax1.tick_params(axis='x', which='minor', bottom='on', top='on')
        ax2.tick_params(axis='x', which='minor', bottom='on', top='on')
        ax3.tick_params(axis='x', which='minor', bottom='on', top='on')

        ax2.set_xlabel(r'$k \, [h/\mathrm{Mpc}]$', labelpad=-10)
        ax1.set_ylabel(r'$f_\mathrm{gas}(m)u(k|m)$')
        ax2.set_yticklabels([])
        ax3.set_yticklabels([])
        # ticks2 = ax2.get_xticklabels()
        # ticks2[-6].set_visible(False)
        # ticks3 = ax3.get_xticklabels()
        # ticks3[-6].set_visible(False)

        ax1.set_title(r'$m_{\mathrm{200m}} = 10^{%.1f} \, \mathrm{M_\odot}/h$'%np.log10(prms.m200m[idx_1]), y=1.015, fontsize=28)
        ax2.set_title(r'$m_{\mathrm{200m}} = 10^{%.1f} \, \mathrm{M_\odot}/h$'%np.log10(prms.m200m[idx_2]), y=1.015, fontsize=28)
        ax3.set_title(r'$m_{\mathrm{200m}} = 10^{%.1f} \, \mathrm{M_\odot}/h$'%np.log10(prms.m200m[idx_3]), y=1.015, fontsize=28)

        ax3.legend(loc='best', fontsize=28)
        plt.savefig('obs_rho_k_extrapolated.pdf', transparent=True)

# ------------------------------------------------------------------------------
# End of plot_profiles_gas_paper()
# ------------------------------------------------------------------------------

def plot_fgas200m_paper(comp_gas, comp_gas_r500c_5r500c, prms=p.prms):
    '''
    Plot gas mass fractions at r200m for our different models
    '''
    fig = plt.figure(figsize=(10,9))
    ax = fig.add_subplot(111)

    f_b = 1 - prms.f_dm

    pl.set_style('line')
    ax.plot(comp_gas.m200m, comp_gas.f_comp, label='model 1')
    ax.plot(comp_gas_r500c_5r500c.m200m, comp_gas_r500c_5r500c.f_comp,
            label='model 3')
    ax.axhline(y=f_b, c='k', ls='--')

    ax.tick_params(axis='x', which='major', pad=6)
    text_props = ax.get_xticklabels()[0].get_font_properties()

    # add annotation to f_bar
    ax.annotate(r'$f_{\mathrm{b}}$',
                 # xy=(1e14, 0.16), xycoords='data',
                 # xytext=(1e14, 0.15), textcoords='data',
                 xy=(10**(11), f_b), xycoords='data',
                 xytext=(1.2 * 10**(11),
                         f_b * 0.95), textcoords='data',
                 fontproperties=text_props)

    ax.set_xscale('log')
    ax.set_xlabel('$m_\mathrm{200m} \, [\mathrm{M_\odot}/h]$')
    ax.set_ylabel('$f_\mathrm{gas,200m}$')
    ax.legend(loc=4)
    plt.savefig('obs_fgas_extrapolated.pdf', transparent=True)

# ------------------------------------------------------------------------------
# End of plot_fgas200m_paper()
# ------------------------------------------------------------------------------

def plot_power_ratio_paper(comp, comp_dmo, comp_gas_dmo, prms=p.prms):
    '''
    Plot the power ratio of comp with respect to the dmo and gas_dmo cases
    '''
    fig = plt.figure(figsize=(10,9))
    ax = fig.add_subplot(111)


    pl.set_style('line')
    # ax.plot(prms.k_range, comp.delta_tot / comp_dmo.delta_tot,
    #         label='gas/dmo')
    # ax.plot(prms.k_range, comp.delta_tot / comp_gas_dmo.delta_tot,
    #         label='gas/gas_dmo')
    ax.plot(prms.k_range_lin, comp / comp_dmo,
            label='gas/dmo')
    ax.plot(prms.k_range_lin, comp / comp_gas_dmo,
            label='gas/gas_dmo')

    ax.set_xscale('log')
    ax.set_xlabel('$k \, [h / \mathrm{Mpc}]$')
    ax.set_ylabel('$P_\mathrm{gas}(k)/P_\mathrm{i}(k)$')
    ax.legend()
    plt.savefig('obs_power_ratio.pdf', transparent=True)

# ------------------------------------------------------------------------------
# End of plot_power_ratio_paper()
# ------------------------------------------------------------------------------

def plot_power_comps_paper(comp1, comp2):
    '''
    Compare the power in comp1 to comp2
    '''
    pl.set_style('line')
    plt.clf()
    fig = plt.figure(figsize=(11, 8))
    ax_P = fig.add_axes([0.1, 0.35, 0.8, 0.55])
    ax_r = fig.add_axes([0.1, 0.1, 0.8, 0.25])

    ax_P.plot(comp1.k_range, comp1.delta_tot, label=r'$n(m_\mathrm{dmo}(n_\mathrm{bar}))$')
    ax_P.plot(comp2.k_range, comp2.delta_tot, label=r'$n(m_\mathrm{dmo})$')

    axd = ax_P.twiny()
    l = 2 * np.pi / comp1.k_range
    axd.plot(l, comp1.delta_tot)
    axd.set_xlim(axd.get_xlim()[::-1])
    axd.cla()
    axd.set_xscale('log')
    axd.set_xlabel(r'$\lambda \, [\mathrm{Mpc}/h]$', labelpad=10)
    axd.tick_params(axis='x', pad=5)

    # yticklabs = ax_P.get_yticklabels()
    # yticklabs[0] = ""
    # ax_P.set_yticklabels(yticklabs)
    ax_P.set_ylim(ymin=2e-3)
    ax_P.set_xlim([1e-2,1e2])
    ax_P.axes.set_xscale('log')
    ax_P.axes.set_yscale('log')
    ax_P.set_ylabel(r'$\Delta^2(k)$')
    # ax_P.set_title(r'Power spectra for BAHAMAS')
    ax_P.set_xticklabels([])
    ax_P.legend(loc='best')

    ax_r.plot(comp1.k_range, comp1.delta_tot / comp2.delta_tot)
    ax_r.axhline(y=1, c='k', ls='--')
    ax_r.grid()
    ax_r.set_xlim([1e-2,1e2])
    # ax_r.set_ylim([1e-3,1])
    ax_r.set_ylim([0.75,1.1])
    ax_r.axes.set_xscale('log')
    # ax_r.axes.set_yscale('log')
    ax_r.minorticks_on()
    ax_r.tick_params(axis='x',which='minor',bottom='off')

    ax_r.legend(loc='best')
    ax_r.set_xlabel(r'$k \, [h/\mathrm{Mpc}]$')
    # ax_r.set_ylabel(r'$\frac{P_{\mathrm{AGN}} - P_{\mathrm{DM}}}{P_{\mathrm{DM}}}$',
    #                 labelpad=-2)
    ax_r.set_ylabel(r'$P_\mathrm{bar}/P_\mathrm{dmo}$')

    plt.savefig('ratio_sim_obs.pdf', dpi=900, transparent=True)
    # plt.close(fig)

# ------------------------------------------------------------------------------
# End of plot_power_comps_paper()
# ------------------------------------------------------------------------------
