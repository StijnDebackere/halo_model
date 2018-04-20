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
import halo.gas as gas
import halo.parameters as p
import halo.tools as tools
import halo.model.density as dens
import halo.model.component as comp
import halo.model.power as power
import halo.data.bahamas as b
import halo.data.data as d

import numpy as np

import pdb

def load_dm_dmo_correa(prms=p.prms):
    '''
    Pure dark matter only component with NFW profile and f_dm = 1
    '''
    m200m = prms.m200m
    m200c = prms.m200c
    r200c = prms.r200c

    f_dm = np.ones_like(m200m)
    c200m = prms.c_correa
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
# End of load_dm_dmo_correa()
# ------------------------------------------------------------------------------

def load_dm_dmo_sim(m_dmo, prms=p.prms):
    '''
    Pure dark matter profile with NFW profile and f_dm = 1
    '''
    m200m = prms.m200m
    m200c = prms.m200c
    r200c = prms.r200c

    rs, rs_err, m, m200, rs_prms, c_prms, m_prms = b.fit_dm_bahamas_dmo()

    f_dm = np.ones_like(m200m)
    c200m = b.dm_plaw_fit(m200m, **c_prms)
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
# End of load_dm_dmo_sim()
# ------------------------------------------------------------------------------

def load_dm_correa(m_dmo, prms=p.prms, bar2dmo=True):
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
        r_dmo = tools.mass_to_radius(m_dmo, 200 * prms.rho_m)
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
# End of load_dm_correa()
# ------------------------------------------------------------------------------

def load_dm_sim(m_dmo, prms=p.prms, bar2dmo=True):
    '''
    Dark matter profile with NFW profile and f_dm = 1 - f_b
    '''
    # halo model parameters
    # ! WATCH OUT ! These are the SPH equivalent masses and do thus not
    # correspond to the DMO case, unless f_gas,200m = f_b
    m200m = prms.m200m
    m200c = prms.m200c
    r200c = prms.r200c

    rs, rs_err, m, m200, rs_prms, c_prms, m_prms = b.fit_dm_bahamas()

    f_dm = np.ones_like(m200m) * prms.f_dm
    c200m = b.dm_plaw_fit(m200m, **c_prms)
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
        r_dmo = tools.mass_to_radius(m_dmo, 200 * prms.rho_m)
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
# End of load_dm_sim()
# ------------------------------------------------------------------------------

def prof_beta(r, rc, beta, m200):
    prof = 1. / (1 + (r/rc)**2)**(beta/2)
    norm = tools.m_h(prof, r)
    prof *= m200 / norm
    return prof

def load_gas_dmo(prms=p.prms):
    '''
    Gas component in beta profile with fgas_500c = fgas_200m = f_b
    '''
    f_gas = (1 - prms.f_dm) * np.ones_like(prms.m200m)
    beta, rc, m500c, r, prof_h = d.beta_mass(f_gas, f_gas, prms)
    # print beta / 3.
    # print rc

    # general profile kwargs to be used for all components
    profile_kwargs = {'r_range': prms.r_range_lin,
                      'm_bar': prms.m200m,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}
    # --------------------------------------------------------------------------
    # specific gas extra kwargs -> need to renormalize f_gas to halo mass
    gas_extra = {'profile': prof_h / f_gas.reshape(-1,1),}
    prof_gas_kwargs = tools.merge_dicts(profile_kwargs, gas_extra)
    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    comp_gas_kwargs = {'name': 'gas',
                       'r200m': prms.r200m,
                       'm200m': prms.m200m,
                       'p_lin': prms.p_lin,
                       'dndm': prms.dndm,
                       'f_comp': f_gas,}

    gas_kwargs = tools.merge_dicts(prof_gas_kwargs, comp_gas_kwargs)

    comp_gas = comp.Component(**gas_kwargs)
    return comp_gas

# ------------------------------------------------------------------------------
# End of load_gas_dmo()
# ------------------------------------------------------------------------------

def prof_gas_hot(x, sl, a, b, m_sl, r500):
    '''beta profile'''
    profile = (1 + (x/a)**2)**(-b/2)
    mass = tools.m_h(profile[sl], x[sl] * r500)
    profile *= m_sl/mass

    return profile

def load_gas(prms=p.prms, bar2dmo=True):
    '''
    Return beta profiles with fgas_500c = f_obs, extrapolated to r200m
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

    rc, beta = b.fit_prms(x=500, m_cut=1e14, prms=prms)
    # gas fractions
    fm_prms = b.f_gas_prms(prms)
    f_gas500 = b.f_gas(m500c, prms=prms, **fm_prms)

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

def load_gas_obs(prms=p.prms):
    '''
    Return beta profiles with fgas_500c = f_obs which only reach up until r500c

    This one is to be used with a profile that matches f_b at r200m, since we
    do not correct from m_bar to m_dmo
    '''
    rc, beta = b.fit_prms(x=500, m_cut=1e14, prms=prms)

    # halo model parameters
    m200m = prms.m200m
    r200m = prms.r200m
    m500c = prms.m500c
    r500c = prms.r500c

    # radius in terms of r500c
    r_range = prms.r_range_lin
    rx = r_range / r500c.reshape(-1,1)

    # gas fractions
    fm_prms = b.f_gas_prms(prms)
    f_gas500 = b.f_gas(m500c, prms=prms, **fm_prms)

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
    '''
    # halo model parameters
    m200m = prms.m200m
    m500c = prms.m500c
    r500c = prms.r500c
    r200m = prms.r200m

    fm_prms = b.f_gas_prms(prms)
    fgas_500 = b.f_gas(m500c, prms=prms, **fm_prms)

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

def load_gas_plaw_r500c_r200m(prms, fgas_200, r500, rho_500):
    '''
    Return power law profiles with fgas_500c = 0 and fgas_200m = f_b - f_obs

    rho_500 is the density of the matching beta profile at r500c
    r500 is the corresponding radius
    '''
    # halo model parameters
    m200m = prms.m200m
    m500c = prms.m500c
    r500c = prms.r500c
    r200m = prms.r200m

    fm_prms = b.f_gas_prms(prms)
    fgas_500 = b.f_gas(m500c, prms=prms, **fm_prms)

    r_range = prms.r_range_lin
    rx = r_range / r500.reshape(-1,1)

    # gas fractions
    f_b = 1 - prms.f_dm

    mgas_500 = fgas_500 * m500c
    mgas_200 = f_b * m200m

    # relative position of virial radius
    x200m = r200m / r500c

    prof_gas = np.zeros_like(rx)
    for idx, prof in enumerate(prof_gas):
        m_plaw = mgas_200[idx] - mgas_500[idx]
        a = b.fit_plaw_index_mass(m_plaw, r500[idx], r200m[idx], rho_500[idx])
        sl = (rx[idx] >= 1.)
        # the joining value will be taken from the beta profile
        sl[sl.argmax()] = 0
        prof_gas[idx][sl] = b.gas_plaw(rx[idx][sl], 1.,  a, rho_500[idx])
        mass = tools.m_h(prof_gas[idx], r_range[idx])

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
    comp_gas_kwargs = {'name': 'plaw',
                       'r200m': r200m,
                       'm200m': m200m,
                       'p_lin': prms.p_lin,
                       'dndm': prms.dndm,
                       'f_comp': f_gas}

    gas_kwargs = tools.merge_dicts(prof_gas_kwargs, comp_gas_kwargs)

    comp_gas = comp.Component(**gas_kwargs)
    return comp_gas

# ------------------------------------------------------------------------------
# End of load_gas_plaw_r500c_r200m()
# ------------------------------------------------------------------------------

def load_gas_5r500c(prms=p.prms, bar2dmo=True):
    '''
    Return beta profiles with fgas_200m = f_obs_extrapolated = fgas_5r500c
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

    rc, beta = b.fit_prms(x=500, m_cut=1e14, prms=prms)
    # gas fractions
    fm_prms = b.f_gas_prms(prms)
    f_gas500 = b.f_gas(m500c, prms=prms, **fm_prms)

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
    m_dmo = b.m200b_to_m200dmo(m200m, f_gas, prms)
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

def load_gas_r500c_r200m_5r500c(prms=p.prms, bar2dmo=True):
    '''
    Return beta profiles with fgas_200m = f_obs = fgas_5r500c
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

    rc, beta = b.fit_prms(x=500, m_cut=1e14, prms=prms)
    # gas fractions
    fm_prms = b.f_gas_prms(prms)
    f_gas500 = b.f_gas(m500c, prms=prms, **fm_prms)
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
    m_dmo = b.m200b_to_m200dmo(m200m, f_gas, prms)
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

def load_dm_dmo_correa_5r500c(prms=p.prms):
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
    c200m = prms.c_correa

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
# End of load_dm_dmo_correa_5r500c()
# ------------------------------------------------------------------------------

def load_dm_dmo_sim_5r500c(prms=p.prms):
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

    rs, rs_err, m, m200, rs_prms, c_prms, m_prms = b.fit_dm_bahamas_dmo()

    f_dm = np.ones_like(m200m)
    c200m = b.dm_plaw_fit(m200m, **c_prms)

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
# End of load_dm_dmo_sim_5r500c()
# ------------------------------------------------------------------------------

def load_dm_correa_5r500c(m_dmo, prms=p.prms, bar2dmo=True):
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
# End of load_dm_correa_5r500c()
# ------------------------------------------------------------------------------

def load_dm_sim_5r500c(m_dmo, prms=p.prms, bar2dmo=True):
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

    rs, rs_err, m, m200, rs_prms, c_prms, m_prms = b.fit_dm_bahamas()

    f_dm = np.ones_like(m200m) * prms.f_dm
    c_x = b.dm_plaw_fit(m200m, **c_prms)
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
# End of load_dm_sim_5r500c()
# ------------------------------------------------------------------------------

def load_gas_dmo_5r500c(prms=p.prms):
    '''
    Return beta profiles with fgas_500c = fgas_200m = f_b = fgas_5r500c
    '''
    # halo model parameters
    m200m = prms.m200m
    m500c = prms.m500c
    r500c = prms.r500c
    r200m = prms.r200m

    r_min = prms.r_range_lin[:,0]
    r_max = (5 * r500c)
    r_range = np.array([np.logspace(np.log10(r_min[i]), np.log10(rm), prms.r_bins)
                        for i,rm in enumerate(r_max)])
    rx = r_range / r500c.reshape(-1,1)

    # relative position of virial radius
    x200m = r200m / r500c

    f_gas = (1 - prms.f_dm) * np.ones_like(prms.m200m)
    beta, rc, m500c, r, prof_h = d.beta_mass(f_gas, f_gas, prms)

    prof_gas = np.zeros_like(rx)
    for idx, prof in enumerate(prof_gas):
        sl = (rx[idx] <= x200m[idx])
        prof_gas[idx][sl] = prof_gas_hot(rx[idx][sl], sl[sl], rc[idx], beta[idx],
                                         f_gas[idx] * m200m[idx],
                                         r500c[idx])

    # profile now runs between 0 and 5r500c
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
    comp_gas_kwargs = {'name': 'gas',
                       'r200m': r200m,
                       'm200m': m200m,
                       'p_lin': prms.p_lin,
                       'dndm': prms.dndm,
                       'f_comp': f_gas,}

    gas_kwargs = tools.merge_dicts(prof_gas_kwargs, comp_gas_kwargs)

    comp_gas = comp.Component(**gas_kwargs)

    return comp_gas

# ------------------------------------------------------------------------------
# End of load_gas_dmo_5r500c()
# ------------------------------------------------------------------------------

def load_gas_smooth_r200m_5r500c(m_dmo, prms, fgas_200, bar2dmo=True):
    '''
    Return uniform profiles with fgas_200m = 0 and fgas_5r500c = f_b - fgas_200
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

def load_gas_plaw_r200m_5r500c(m_dmo, prms, fgas_200, r200, rho_200,
                               bar2dmo=False):
    '''
    Return plaw profiles with fgas_200m = 0 and fgas_5r500c = f_b - fgas_200

    rho_200 is the density of the matching beta profile at r500c
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

    mgas_200 = fgas_200 * m200m
    mgas_5r500c = f_b * m200m

    # relative position of virial radius
    x200m = r200m / r500c

    prof_gas = np.zeros_like(rx)
    for idx, prof in enumerate(prof_gas):
        m_plaw = mgas_5r500c[idx] - mgas_200[idx]
        if m_plaw < 0.:
            continue
        else:
            a = b.fit_plaw_index_mass(m_plaw, r200[idx], 5 * r500c[idx],
                                      rho_200[idx])
            sl = (r_range[idx] >= r200[idx])
            # the joining value will be taken from the beta profile
            sl[sl.argmax()] = 0
            prof_gas[idx][sl] = b.gas_plaw(r_range[idx][sl], r200[idx], a,
                                           rho_200[idx])
            mass = tools.m_h(prof_gas[idx], r_range[idx])
            # print a
            # print (mass - m_plaw) / m_plaw
            # print '------------------'

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
    # there are gas fractions which are zero, cannot divide by zero
    prof = np.concatenate([prof_gas[f_gas > 0.] / f_gas[f_gas > 0].reshape(-1,1),
                           prof_gas[f_gas == 0.]])

    gas_extra = {'profile': prof}
    prof_gas_kwargs = tools.merge_dicts(profile_kwargs, gas_extra)
    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    if bar2dmo == False:
        comp_gas_kwargs = {'name': 'plaw',
                           'r200m': r200m,
                           'm200m': m200m,
                           'p_lin': prms.p_lin,
                           'dndm': prms.dndm,
                           'f_comp': f_gas}
    else:
        prms_dmo = deepcopy(prms)
        prms_dmo.m200m = m_dmo
        comp_gas_kwargs = {'name': 'plaw',
                           'r200m': r200m,
                           'm200m': m200m,
                           'p_lin': prms.p_lin,
                           'dndm': prms_dmo.dndm,
                           'f_comp': f_gas}

    gas_kwargs = tools.merge_dicts(prof_gas_kwargs, comp_gas_kwargs)

    comp_gas = comp.Component(**gas_kwargs)
    return comp_gas

# ------------------------------------------------------------------------------
# End of load_gas_plaw_r200m_5r500c()
# ------------------------------------------------------------------------------

def load_models(prms=p.prms, bar2dmo=True):
    m200m = prms.m200m
    m500c = prms.m500c
    r500c = prms.r500c
    r200m = prms.r200m

    r_min = prms.r_range_lin[:,0]
    r_max = (5 * r500c)
    r_range_5r500c = np.array([np.logspace(np.log10(r_min[i]),
                                           np.log10(rm),
                                           prms.r_bins)
                        for i,rm in enumerate(r_max)])

    # load dmo models
    dm_dmo_sim = load_dm_dmo_sim(prms)
    dm_dmo_cor = load_dm_dmo_correa(prms)
    pow_dm_dmo_sim = power.Power([dm_dmo_sim], name='dmo_sim')
    pow_dm_dmo_cor = power.Power([dm_dmo_cor], name='dmo_cor')

    dm_sim = load_dm_sim(m_dmo=prms.m200m, prms=prms, bar2dmo=False)
    dm_cor = load_dm_correa(m_dmo=prms.m200m, prms=prms, bar2dmo=False)
    gas_dmo = load_gas_dmo(prms)
    pow_gas_dmo_sim = power.Power([dm_sim, gas_dmo], name='gas_dmo')
    pow_gas_dmo_cor = power.Power([dm_cor, gas_dmo], name='gas_cor')

    dm_dmo_sim_5r500c = load_dm_dmo_sim_5r500c(prms)
    dm_dmo_cor_5r500c = load_dm_dmo_correa_5r500c(prms)
    pow_dm_dmo_sim_5r500c = power.Power([dm_dmo_sim_5r500c], name='dmo_sim_5r500c')
    pow_dm_dmo_cor_5r500c = power.Power([dm_dmo_cor_5r500c], name='dmo_cor_5r500c')

    dm_sim_5r500c = load_dm_sim_5r500c(m_dmo=prms.m200m, prms=prms, bar2dmo=False)
    dm_cor_5r500c = load_dm_correa_5r500c(m_dmo=prms.m200m, prms=prms,
                                          bar2dmo=False)
    gas_dmo_5r500c = load_gas_dmo_5r500c(prms)
    pow_gas_dmo_sim_5r500c = power.Power([dm_sim_5r500c, gas_dmo_5r500c],
                                         name='gas_dmo_sim_5r500c')
    pow_gas_dmo_cor_5r500c = power.Power([dm_cor_5r500c, gas_dmo_5r500c],
                                         name='gas_dmo_cor_5r500c')

    # load gas_extrap
    gas = load_gas(prms, bar2dmo=bar2dmo)
    m_dmo_gas = d.m200b_to_m200dmo(gas.m200m, gas.f_comp, prms)
    dm_sim_gas = load_dm_sim(m_dmo=m_dmo_gas, prms=prms, bar2dmo=bar2dmo)
    dm_cor_gas = load_dm_correa(m_dmo=m_dmo_gas, prms=prms, bar2dmo=bar2dmo)
    pow_gas_sim = power.Power([dm_sim_gas, gas], name='gas_extrap')
    pow_gas_cor = power.Power([dm_cor_gas, gas], name='gas_extrap')

    # load gas_smooth_r500c_r200m
    gas_beta = load_gas_obs(prms)
    gas_smooth = load_gas_smooth_r500c_r200m(prms, gas_beta.f_comp)
    pow_gas_smooth_sim_r500c_r200m = power.Power([dm_sim, gas_beta, gas_smooth],
                                                 name='gas_smooth_r500c_r200m')
    pow_gas_smooth_cor_r500c_r200m = power.Power([dm_cor, gas_beta, gas_smooth],
                                                 name='gas_smooth_r500c_r200m')

    # need to know location of r500c & r200m for the plaw
    # this relies on first found False being the minimum
    r500c_idx_0 = np.argmin((prms.r_range_lin - prms.r500c.reshape(-1,1)) < 0,
                            axis=1) - 1
    r500c_idx_1 = np.argmin((r_range_5r500c - prms.r500c.reshape(-1,1)) < 0,
                            axis=1) - 1
    r200m_idx_0 = np.argmin((prms.r_range_lin - prms.r200m.reshape(-1,1)) < 0,
                            axis=1) - 1
    r200m_idx_1 = np.argmin((r_range_5r500c - prms.r200m.reshape(-1,1)) < 0,
                            axis=1) - 1

    # load gas_plaw_r500c_r200m
    gas_beta_rho500 = (gas_beta.f_comp *
                       gas_beta.rho_r[np.arange(r500c_idx_0.shape[0]),
                                      r500c_idx_0])
    r500 = prms.r_range_lin[np.arange(r500c_idx_0.shape[0]), r500c_idx_0]
    gas_plaw = load_gas_plaw_r500c_r200m(prms, gas_beta.f_comp, r500,
                                         gas_beta_rho500)
    pow_gas_plaw_sim_r500c_r200m = power.Power([dm_sim, gas_beta, gas_plaw],
                                               name='gas_plaw_r500c_r200m')
    pow_gas_plaw_cor_r500c_r200m = power.Power([dm_cor, gas_beta, gas_plaw],
                                               name='gas_plaw_r500c_r200m')

    # load gas_smooth_r200m_5r500c
    gas_5r500c = load_gas_5r500c(prms, bar2dmo=bar2dmo)
    m_dmo_5r500c = d.m200b_to_m200dmo(gas_5r500c.m200m, gas_5r500c.f_comp, prms)
    dm_sim_r200m_5r500c = load_dm_sim_5r500c(m_dmo_5r500c, prms, bar2dmo=bar2dmo)
    dm_cor_r200m_5r500c = load_dm_correa_5r500c(m_dmo_5r500c, prms, bar2dmo=bar2dmo)

    gas_smooth_r200m_5r500c = load_gas_smooth_r200m_5r500c(m_dmo_5r500c, prms,
                                                           gas_5r500c.f_comp,
                                                           bar2dmo=bar2dmo)
    pow_gas_smooth_sim_r200m_5r500c = power.Power([dm_sim_r200m_5r500c, gas_5r500c,
                                                   gas_smooth_r200m_5r500c],
                                                  name='gas_smooth_r200m_5r500c')
    pow_gas_smooth_cor_r200m_5r500c = power.Power([dm_cor_r200m_5r500c, gas_5r500c,
                                                   gas_smooth_r200m_5r500c],
                                                  name='gas_smooth_r200m_5r500c')
    # and plaw
    gas_rho200 = (gas_5r500c.f_comp *
                  gas_5r500c.rho_r[np.arange(r200m_idx_1.shape[0]),
                                   r200m_idx_1])
    r200 = r_range_5r500c[np.arange(r200m_idx_1.shape[0]), r200m_idx_1]
    gas_plaw_r200m_5r500c = load_gas_plaw_r200m_5r500c(m_dmo_5r500c, prms,
                                                       gas_5r500c.f_comp,
                                                       r200, gas_rho200)
    pow_gas_plaw_sim_r200m_5r500c = power.Power([dm_sim_r200m_5r500c, gas_5r500c,
                                                 gas_plaw_r200m_5r500c],
                                                name='gas_plaw_r200m_5r500c')
    pow_gas_plaw_cor_r200m_5r500c = power.Power([dm_cor_r200m_5r500c, gas_5r500c,
                                                 gas_plaw_r200m_5r500c],
                                                name='gas_plaw_r200m_5r500c')

    # load_gas_smooth_r500c_5r500c
    gas_r500c_5r500c = load_gas_r500c_r200m_5r500c(prms, bar2dmo=bar2dmo)
    m_dmo_r500c_5r500c = d.m200b_to_m200dmo(gas_r500c_5r500c.m200m,
                                            gas_r500c_5r500c.f_comp, prms)
    dm_sim_r500c_5r500c = load_dm_sim_5r500c(m_dmo_r500c_5r500c, prms,
                                             bar2dmo=bar2dmo)
    dm_cor_r500c_5r500c = load_dm_correa_5r500c(m_dmo_r500c_5r500c, prms,
                                                bar2dmo=bar2dmo)

    gas_smooth_r500c_5r500c = load_gas_smooth_r200m_5r500c(m_dmo_r500c_5r500c,
                                                           prms,
                                                           gas_r500c_5r500c.f_comp,
                                                           bar2dmo=bar2dmo)
    pow_gas_smooth_sim_r500c_5r500c = power.Power([dm_sim_r500c_5r500c,
                                                   gas_r500c_5r500c,
                                                   gas_smooth_r500c_5r500c],
                                                  name='gas_smooth_r500c_5r500c')
    pow_gas_smooth_cor_r500c_5r500c = power.Power([dm_cor_r500c_5r500c,
                                                   gas_r500c_5r500c,
                                                   gas_smooth_r500c_5r500c],
                                                  name='gas_smooth_r500c_5r500c')

    # load_gas_plaw_r500c_5r500c
    ## !! Need to adjust function load_gas_r500c_r200m_5r500c to do this...

    results = {'dm_dmo_sim': pow_dm_dmo_sim,
               'gas_dmo_sim': pow_gas_dmo_sim,
               'dm_dmo_sim_5r500c': pow_dm_dmo_sim_5r500c,
               'gas_dmo_sim_5r500c': pow_gas_dmo_sim_5r500c,
               'gas_sim': pow_gas_sim,
               'smooth_sim_r500c_r200m': pow_gas_smooth_sim_r500c_r200m,
               'smooth_sim_r200m_5r500c': pow_gas_smooth_sim_r200m_5r500c,
               'smooth_sim_r500c_5r500c': pow_gas_smooth_sim_r500c_5r500c,
               'plaw_sim_r500c_r200m': pow_gas_plaw_sim_r500c_r200m,
               'plaw_sim_r200m_5r500c': pow_gas_plaw_sim_r200m_5r500c,
               'dm_dmo_cor': pow_dm_dmo_cor,
               'gas_dmo_cor': pow_gas_dmo_cor,
               'dm_dmo_cor_5r500c': pow_dm_dmo_cor_5r500c,
               'gas_dmo_cor_5r500c': pow_gas_dmo_cor_5r500c,
               'gas_cor': pow_gas_cor,
               'smooth_cor_r500c_r200m': pow_gas_smooth_cor_r500c_r200m,
               'smooth_cor_r200m_5r500c': pow_gas_smooth_cor_r200m_5r500c,
               'smooth_cor_r500c_5r500c': pow_gas_smooth_cor_r500c_5r500c,
               'plaw_cor_r500c_r200m': pow_gas_plaw_cor_r500c_r200m,
               'plaw_cor_r200m_5r500c': pow_gas_plaw_cor_r200m_5r500c,}

    return results

# ------------------------------------------------------------------------------
# End of load_models()
# ------------------------------------------------------------------------------
