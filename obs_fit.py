import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import sys
import cPickle

# allow import of plot
sys.path.append('~/Documents/Universiteit/MR/code')
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

import pdb

def load_dm_dmo(prms=p.prms, save=True):
    # general profile kwargs to be used for all components
    profile_kwargs = {'r_range': prms.r_range_lin,
                      'm_range': prms.m_range_lin,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}


    m_range = prms.m_range_lin
    m200c = np.array([gas.m200m_to_m200c(m) for m in m_range])
    r200c = tools.mass_to_radius(m200c, 200 * prms.rho_crit * prms.h**2)
    # --------------------------------------------------------------------------
    # Correa
    f_dm = np.ones_like(m_range)
    c_x = profs.c_correa(m200c, 0).reshape(-1) * prms.r_range_lin[:,-1]/r200c
    r_x = prms.r_range_lin[:,-1]
    # specific dm extra kwargs
    dm_extra = {'profile': profs.profile_NFW,
                'profile_f': profs.profile_NFW_f,
                'profile_args': {'c_x': c_x,
                                 'r_x': r_x,
                                 'rho_mean': prms.rho_m},
                'profile_f_args': {'c_x': c_x,
                                   'r_x': r_x,
                                   'rho_mean': prms.rho_m},
                'f_comp': f_dm}
    prof_dm_kwargs = tools.merge_dicts(profile_kwargs, dm_extra)
    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    comp_dm_kwargs = {'name': 'dm',
                      'p_lin': prms.p_lin,
                      'nu': prms.nu,
                      'fnu': prms.fnu,
                      # 'm_fn': prms.m_fn,
                      'f_comp': f_dm,}
                      # 'bias_fn': bias.bias_Tinker10,
                      # 'bias_fn_args': {'nu': prms.nu}}

    dm_kwargs = tools.merge_dicts(prof_dm_kwargs, comp_dm_kwargs)

    comp_dm = comp.Component(**dm_kwargs)

    if save:
        with open('obs_comp_dm_dmo.p', 'wb') as f:
            cPickle.dump(comp_dm, f)

    return comp_dm

# ------------------------------------------------------------------------------
# End of load_dm()
# ------------------------------------------------------------------------------

def load_dm(prms=p.prms, save=True):
    # general profile kwargs to be used for all components
    profile_kwargs = {'r_range': prms.r_range_lin,
                      'm_range': prms.m_range_lin,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}

    m_range = prms.m_range_lin
    m200c = np.array([gas.m200m_to_m200c(m) for m in m_range])
    r200c = tools.mass_to_radius(m200c, 200 * prms.rho_crit * prms.h**2)
    # --------------------------------------------------------------------------
    # Correa
    f_dm = np.ones_like(m_range) * prms.f_dm
    c_x = profs.c_correa(m200c, 0).reshape(-1) * prms.r_range_lin[:,-1]/r200c
    r_x = prms.r_range_lin[:,-1]
    # specific dm extra kwargs
    dm_extra = {'profile': profs.profile_NFW,
                'profile_f': profs.profile_NFW_f,
                'profile_args': {'c_x': c_x,
                                 'r_x': r_x,
                                 'rho_mean': prms.rho_m},
                'profile_f_args': {'c_x': c_x,
                                   'r_x': r_x,
                                   'rho_mean': prms.rho_m},
                'f_comp': f_dm}
    prof_dm_kwargs = tools.merge_dicts(profile_kwargs, dm_extra)
    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    comp_dm_kwargs = {'name': 'dm',
                      'p_lin': prms.p_lin,
                      'nu': prms.nu,
                      'fnu': prms.fnu,
                      # 'm_fn': prms.m_fn,
                      'f_comp': f_dm,}
                      # 'bias_fn': bias.bias_Tinker10,
                      # 'bias_fn_args': {'nu': prms.nu}}

    dm_kwargs = tools.merge_dicts(prof_dm_kwargs, comp_dm_kwargs)

    comp_dm = comp.Component(**dm_kwargs)

    if save:
        with open('obs_comp_dm.p', 'wb') as f:
            cPickle.dump(comp_dm, f)

    return comp_dm

# ------------------------------------------------------------------------------
# End of load_dm()
# ------------------------------------------------------------------------------

def prof_beta(r, rc, beta, m200):
    prof = 1. / (1 + (r/rc)**2)**(beta/2)
    norm = tools.m_h(prof, r)
    prof *= m200 / norm
    return prof

def load_gas_dmo(prms=p.prms, save=True):
    # general profile kwargs to be used for all components
    profile_kwargs = {'r_range': prms.r_range_lin,
                      'm_range': prms.m_range_lin,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}

    f_gas = (1 - prms.f_dm) * np.ones_like(prms.m_range_lin)
    beta, rc, m500c, r, prof_h = d.beta_mass(f_gas, f_gas, prms)
    print beta / 3.
    print rc
    # --------------------------------------------------------------------------
    # specific gas extra kwargs -> need f_gas
    gas_extra = {'profile': prof_h / f_gas.reshape(-1,1),
                 'f_comp': f_gas}
    prof_gas_kwargs = tools.merge_dicts(profile_kwargs, gas_extra)
    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    comp_gas_kwargs = {'name': 'gas',
                       'p_lin': prms.p_lin,
                       'nu': prms.nu,
                       'fnu': prms.fnu,
                       'f_comp': f_gas,}
                       # 'm_fn': p.prms.m_fn,
                       # 'bias_fn': bias.bias_Tinker10,
                       # 'bias_fn_args': {'nu': prms.nu}}

    gas_kwargs = tools.merge_dicts(prof_gas_kwargs, comp_gas_kwargs)

    comp_gas = comp.Component(**gas_kwargs)
    if save:
        with open('obs_comp_gas_dmo.p', 'wb') as f:
            cPickle.dump(comp_gas, f)

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

# def load_gas(prms=p.prms, fit='med', save=True):
#     # general profile kwargs to be used for all components
#     profile_kwargs = {'r_range': prms.r_range_lin,
#                       'm_range': prms.m_range_lin,
#                       'k_range': prms.k_range_lin,
#                       'n': 80,
#                       'taylor_err': 1.e-50}

#     m_range = prms.m_range_lin
#     m500c = np.array([gas.m200m_to_m500c(m) for m in m_range])
#     r500c = tools.mass_to_radius(m500c, 500 * prms.rho_crit * prms.h**2)
#     rx = prms.r_range_lin / r500c.reshape(-1,1)

#     # gas fractions
#     fm_prms, f1_prms, f2_prms = d.f_gas_prms()
#     if fit == 'med':
#         f_prms = fm_prms
#     # elif fit == 'q16':
#     #     f_prms = f1_prms
#     # elif fit == 'q84':
#     #     f_prms = f2_prms

#     f_gas500 = d.f_gas(m500c, **f_prms)

#     # fit parameters
#     corr_prms, rc_min, rc_max, rc_med, beta_med = d.prof_prms()
#     # beta_min = np.power(10, d.corr_fit(np.log10(rc_min), **corr_prms))
#     # beta_max = np.power(10, d.corr_fit(np.log10(rc_max), **corr_prms))
#     print rc_med
#     print beta_med

#     # prof_gas_min = np.zeros_like(rx)
#     # prof_gas_max = np.zeros_like(rx)
#     prof_gas_med = np.zeros_like(rx)
#     for idx, prof in enumerate(prof_gas_med):
#         sl = (rx[idx] <= 1.)
#         # prof_gas_min[idx] = (prof_gas_hot(rx[idx], sl, rc_min, beta_min,
#         #                                f_gas500[idx] * m500c[idx],
#         #                                r500c[idx]))
#         # prof_gas_max[idx] = (prof_gas_hot(rx[idx], sl, rc_max, beta_max,
#         #                                f_gas500[idx] * m500c[idx],
#         #                                r500c[idx]))
#         prof_gas_med[idx] = (prof_gas_hot(rx[idx], sl, rc_med, beta_med,
#                                        f_gas500[idx] * m500c[idx],
#                                        r500c[idx]))

#     # mgas200_1 = tools.m_h(prof_gas_min, prms.r_range_lin)
#     # f_gas_min = mgas200_1 / m_range
#     # mgas200_2 = tools.m_h(prof_gas_max, prms.r_range_lin)
#     # f_gas_max = mgas200_2 / m_range
#     mgas200_3 = tools.m_h(prof_gas_med, prms.r_range_lin)
#     f_gas_med = mgas200_3 / m_range

#     print f_gas500
#     # print f_gas_min
#     # print f_gas_max
#     print f_gas_med
#     # --------------------------------------------------------------------------
#     # specific gas extra kwargs -> need f_gas
#     # gas_extra1 = {'profile': prof_gas_min / f_gas_min.reshape(-1,1),
#     #               'f_comp': f_gas_min}
#     # gas_extra2 = {'profile': prof_gas_max / f_gas_max.reshape(-1,1),
#     #               'f_comp': f_gas_max}
#     gas_extra3 = {'profile': prof_gas_med / f_gas_med.reshape(-1,1),
#                   'f_comp': f_gas_med}
#     # prof_gas_kwargs1 = tools.merge_dicts(profile_kwargs, gas_extra1)
#     # prof_gas_kwargs2 = tools.merge_dicts(profile_kwargs, gas_extra2)
#     prof_gas_kwargs3 = tools.merge_dicts(profile_kwargs, gas_extra3)
#     # --------------------------------------------------------------------------
#     # additional kwargs for comp.Component
#     comp_gas_kwargs = {'name': 'gas',
#                       'p_lin': prms.p_lin,
#                       'nu': prms.nu,
#                       'fnu': prms.fnu,
#                       'm_fn': p.prms.m_fn,
#                       'bias_fn': bias.bias_Tinker10,
#                       'bias_fn_args': {'nu': prms.nu}}

#     # gas_kwargs1 = tools.merge_dicts(prof_gas_kwargs1, comp_gas_kwargs)
#     # gas_kwargs2 = tools.merge_dicts(prof_gas_kwargs2, comp_gas_kwargs)
#     gas_kwargs3 = tools.merge_dicts(prof_gas_kwargs3, comp_gas_kwargs)

#     # comp_gas_min = comp.Component(**gas_kwargs1)
#     # comp_gas_max = comp.Component(**gas_kwargs2)
#     comp_gas_med = comp.Component(**gas_kwargs3)
#     if save:
#     #     with open('obs_comp_gas_min_%s.p'%fit, 'wb') as f:
#     #         cPickle.dump(comp_gas_min, f)
#     #     with open('obs_comp_gas_max_%s.p'%fit, 'wb') as f:
#     #         cPickle.dump(comp_gas_max, f)
#         with open('obs_comp_gas_med_%s.p'%fit, 'wb') as f:
#             cPickle.dump(comp_gas_med, f)

#     # return comp_gas_min, comp_gas_max, comp_gas_med
#     return comp_gas_med

# # ------------------------------------------------------------------------------
# # End of load_gas()
# # ------------------------------------------------------------------------------

def load_gas(prms=p.prms):
    '''
    Return gas profiles for halo model
    '''
    profile_kwargs = {'r_range': prms.r_range_lin,
                      'm_range': prms.m_range_lin,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}

    rc, beta = d.fit_prms()

    # halo model parameters
    m_range = prms.m_range_lin
    m500c = np.array([gas.m200m_to_m500c(m) for m in m_range])
    r500c = tools.mass_to_radius(m500c, 500 * prms.rho_crit * prms.h**2)
    # radius in terms of r500c
    r_range = prms.r_range_lin
    rx = r_range / r500c.reshape(-1,1)

    # gas fractions
    f_b = 1 - prms.f_dm
    fm_prms, f1_prms, f2_prms = d.f_gas_prms()
    f_gas500 = d.f_gas(m500c, **fm_prms)

    prof_gas = np.zeros_like(rx)
    for idx, prof in enumerate(prof_gas):
        sl = (rx[idx] <= 1.)
        prof_gas[idx] = prof_gas_hot(rx[idx], sl, rc, beta,
                                     f_gas500[idx] * m500c[idx],
                                     r500c[idx])

    mgas200 = tools.m_h(prof_gas, prms.r_range_lin)
    f_gas = mgas200 / m_range

    # --------------------------------------------------------------------------
    # specific gas extra kwargs -> need f_gas
    gas_extra = {'profile': prof_gas / f_gas.reshape(-1,1),
                 'f_comp': f_gas}
    prof_gas_kwargs = tools.merge_dicts(profile_kwargs, gas_extra)
    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    comp_gas_kwargs = {'name': 'gas',
                       'p_lin': prms.p_lin,
                       'nu': prms.nu,
                       'fnu': prms.fnu,
                       'f_comp': f_gas}
                      # 'm_fn': p.prms.m_fn,
                      # 'bias_fn': bias.bias_Tinker10,
                      # 'bias_fn_args': {'nu': prms.nu}}

    gas_kwargs = tools.merge_dicts(prof_gas_kwargs, comp_gas_kwargs)

    comp_gas = comp.Component(**gas_kwargs)
    return comp_gas

# ------------------------------------------------------------------------------
# End of load_gas()
# ------------------------------------------------------------------------------

def load_gas_r500c(prms):
    '''
    Return gas profiles for halo model
    '''
    profile_kwargs = {'r_range': prms.r_range_lin,
                      'm_range': prms.m_range_lin,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}

    rc, beta = d.fit_prms()

    # halo model parameters
    m_range = prms.m_range_lin
    m500c = np.array([gas.m200m_to_m500c(m) for m in m_range])
    r500c = tools.mass_to_radius(m500c, 500 * prms.rho_crit * prms.h**2)
    # radius in terms of r500c
    r_range = prms.r_range_lin
    rx = r_range / r500c.reshape(-1,1)

    # gas fractions
    f_b = 1 - prms.f_dm
    fm_prms, f1_prms, f2_prms = d.f_gas_prms()
    f_gas500 = d.f_gas(m500c, **fm_prms)

    prof_gas = np.zeros_like(rx)
    for idx, prof in enumerate(prof_gas):
        sl = (rx[idx] <= 1.)
        prof_gas[idx][sl] = prof_gas_hot(rx[idx], sl, rc, beta,
                                         f_gas500[idx] * m500c[idx],
                                         r500c[idx])[sl]

    mgas200 = tools.m_h(prof_gas, prms.r_range_lin)
    f_gas = mgas200 / m_range

    # --------------------------------------------------------------------------
    # specific gas extra kwargs -> need f_gas
    gas_extra = {'profile': prof_gas / f_gas.reshape(-1,1),
                 'f_comp': f_gas}
    prof_gas_kwargs = tools.merge_dicts(profile_kwargs, gas_extra)
    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    comp_gas_kwargs = {'name': 'gas',
                      'p_lin': prms.p_lin,
                      'nu': prms.nu,
                      'fnu': prms.fnu,}
                      # 'm_fn': p.prms.m_fn,
                      # 'bias_fn': bias.bias_Tinker10,
                      # 'bias_fn_args': {'nu': prms.nu}}

    gas_kwargs = tools.merge_dicts(prof_gas_kwargs, comp_gas_kwargs)

    comp_gas = comp.Component(**gas_kwargs)
    return comp_gas

# ------------------------------------------------------------------------------
# End of load_gas_r500c()
# ------------------------------------------------------------------------------

def load_gas_smooth_r500c(prms=p.prms):
    '''
    Return smooth gas beyond r500c up to virial radius such that the total mass
    equals f_b
    '''
    # halo model parameters
    m_range = prms.m_range_lin
    m500c = np.array([gas.m200m_to_m500c(m) for m in m_range])
    r500c = tools.mass_to_radius(m500c, 500 * prms.rho_crit * prms.h**2)
    r200m = prms.r_range_lin[:,-1]

    fm_prms, f1_prms, f2_prms = d.f_gas_prms()
    fgas_500 = d.f_gas(m500c, **fm_prms)

    r_range = prms.r_range_lin
    rx = r_range / r500c.reshape(-1,1)

    # profile now runs between 0 and 5r500c
    profile_kwargs = {'r_range': r_range,
                      'm_range': prms.m_range_lin,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}

    # gas fractions
    f_b = 1 - prms.f_dm
    # relative position of virial radius
    x200m = r200m / r500c

    prof_gas = np.zeros_like(rx)
    for idx, prof in enumerate(prof_gas):
        sl = (rx[idx] >= 1.)
        prof_gas[idx][sl] = 1.
        mass = tools.m_h(prof_gas[idx], r_range[idx])
        prof_gas[idx] *= (f_b - fgas_500[idx]) * m_range[idx] / mass

    mgas = tools.m_h(prof_gas, r_range)
    f_gas = mgas / (m_range)
    # --------------------------------------------------------------------------
    # specific gas extra kwargs -> need f_gas
    gas_extra = {'profile': prof_gas / f_gas.reshape(-1,1),
                 'f_comp': f_gas}
    prof_gas_kwargs = tools.merge_dicts(profile_kwargs, gas_extra)
    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    comp_gas_kwargs = {'name': 'smooth',
                      'p_lin': prms.p_lin,
                      'nu': prms.nu,
                      'fnu': prms.fnu,}
                      # 'm_fn': p.prms.m_fn,
                      # 'bias_fn': bias.bias_Tinker10,
                      # 'bias_fn_args': {'nu': prms.nu}}

    gas_kwargs = tools.merge_dicts(prof_gas_kwargs, comp_gas_kwargs)

    comp_gas = comp.Component(**gas_kwargs)
    return comp_gas

# ------------------------------------------------------------------------------
# End of load_gas_smooth_r500c()
# ------------------------------------------------------------------------------

def load_gas_smooth(prms, fgas_200):
    '''
    Return smooth gas beyond virial radius such that the total mass equals f_b
    '''
    # halo model parameters
    m_range = prms.m_range_lin
    m500c = np.array([gas.m200m_to_m500c(m) for m in m_range])
    r500c = tools.mass_to_radius(m500c, 500 * prms.rho_crit * prms.h**2)
    r200m = prms.r_range_lin[:,-1]

    r_range = prms.r_range_lin * (5 * r500c / r200m).reshape(-1,1)
    rx = r_range / r500c.reshape(-1,1)

    # profile now runs between 0 and 5r500c
    profile_kwargs = {'r_range': r_range,
                      'm_range': prms.m_range_lin,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}

    # gas fractions
    f_b = 1 - prms.f_dm
    # relative position of virial radius
    x200m = r200m / r500c

    prof_gas = np.zeros_like(rx)
    for idx, prof in enumerate(prof_gas):
        sl = (rx[idx] >= x200m[idx])
        prof_gas[idx][sl] = 1.
        mass = tools.m_h(prof_gas[idx], r_range[idx])
        prof_gas[idx] *= (f_b - fgas_200[idx]) * m_range[idx] / mass

    mgas = tools.m_h(prof_gas, r_range)
    f_gas = mgas / (m_range)
    # --------------------------------------------------------------------------
    # specific gas extra kwargs -> need f_gas
    gas_extra = {'profile': prof_gas / f_gas.reshape(-1,1),
                 'f_comp': f_gas}
    prof_gas_kwargs = tools.merge_dicts(profile_kwargs, gas_extra)
    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    comp_gas_kwargs = {'name': 'smooth',
                      'p_lin': prms.p_lin,
                      'nu': prms.nu,
                      'fnu': prms.fnu,}
                      # 'm_fn': p.prms.m_fn,
                      # 'bias_fn': bias.bias_Tinker10,
                      # 'bias_fn_args': {'nu': prms.nu}}

    gas_kwargs = tools.merge_dicts(prof_gas_kwargs, comp_gas_kwargs)

    comp_gas = comp.Component(**gas_kwargs)
    return comp_gas

# ------------------------------------------------------------------------------
# End of load_gas_smooth()
# ------------------------------------------------------------------------------

def load_gas_fb_fb(prms=p.prms):
    '''
    REFERENCE!!!

    Load gas profile with f500c=f_b and f200m=f_b, since low mass
    systems cannot increase fast enough, we assume them to be uniform density

    --> No missing mass here
    '''
    # general profile kwargs to be used for all components
    profile_kwargs = {'r_range': prms.r_range_lin,
                      'm_range': prms.m_range_lin,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}

    m_range = prms.m_range_lin
    m500c = np.array([gas.m200m_to_m500c(m) for m in m_range])
    r500c = tools.mass_to_radius(m500c, 500 * prms.rho_crit * prms.h**2)
    rx = prms.r_range_lin / r500c.reshape(-1,1)

    f_b = (1 - prms.f_dm) * np.ones_like(m_range)

    # fit parameters
    beta, rc, m500c, r, prof_h = d.beta_mass(f_b, f_b)

    # good fit indices
    idcs = ~((beta == 0) | (np.abs(beta - 3) < 1e-3))

    prof_gas_med = np.zeros_like(rx)
    for idx, prof in enumerate(prof_gas_med):
        # good index
        if idcs[idx]:
            sl = (rx[idx] <= 1.)
            prof_gas_med[idx] = (prof_gas_hot(rx[idx], sl,
                                              rc[idx], beta[idx],
                                              f_b[idx] *
                                              m500c[idx],
                                              r500c[idx]))
        else:
            prof_gas_med[idx] = np.ones_like(rx[idx])
            norm = tools.m_h(prof_gas_med[idx], prms.r_range_lin[idx])
            # normalize profile to f_b * m200
            prof_gas_med[idx] *= f_b[idx] * m_range[idx]/ norm

    mgas200 = tools.m_h(prof_gas_med, prms.r_range_lin)
    f_gas_med = mgas200 / m_range

    print f_gas_med
    # --------------------------------------------------------------------------
    # specific gas extra kwargs -> need f_gas
    gas_extra = {'profile': prof_gas_med / f_gas_med.reshape(-1,1),
                 'f_comp': f_gas_med}
    prof_gas_kwargs = tools.merge_dicts(profile_kwargs, gas_extra)
    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    comp_gas_kwargs = {'name': 'gas',
                      'p_lin': prms.p_lin,
                      'nu': prms.nu,
                      'fnu': prms.fnu,}
                      # 'm_fn': p.prms.m_fn,
                      # 'bias_fn': bias.bias_Tinker10,
                      # 'bias_fn_args': {'nu': prms.nu}}

    gas_kwargs = tools.merge_dicts(prof_gas_kwargs, comp_gas_kwargs)

    comp_gas_med = comp.Component(**gas_kwargs)

    return comp_gas_med

# ------------------------------------------------------------------------------
# End of load_gas_f500_f200()
# ------------------------------------------------------------------------------

def load_gas_f500_fb(prms=p.prms):
    '''
    Load gas profile with f500c=f500c_obs and f200m=f_b, since low mass
    systems cannot increase fast enough, we assume them to be uniform density

    --> No missing mass here
    '''
    # general profile kwargs to be used for all components
    profile_kwargs = {'r_range': prms.r_range_lin,
                      'm_range': prms.m_range_lin,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}

    m_range = prms.m_range_lin
    m500c = np.array([gas.m200m_to_m500c(m) for m in m_range])
    r500c = tools.mass_to_radius(m500c, 500 * prms.rho_crit * prms.h**2)
    rx = prms.r_range_lin / r500c.reshape(-1,1)

    # gas fractions
    fm_prms, f1_prms, f2_prms = d.f_gas_prms()

    # only fit median values
    f_prms = fm_prms

    f_gas500 = d.f_gas(m500c, **f_prms)
    f_b = (1 - prms.f_dm) * np.ones_like(f_gas500)

    # fit parameters
    beta, rc, m500c, r, prof_h = d.beta_mass(f_gas500, f_b)

    # good fit indices
    idcs = ~((beta == 0) | (np.abs(beta - 3) < 1e-3))

    prof_gas_med = np.zeros_like(rx)
    for idx, prof in enumerate(prof_gas_med):
        # good index
        if idcs[idx]:
            sl = (rx[idx] <= 1.)
            prof_gas_med[idx] = (prof_gas_hot(rx[idx], sl,
                                              rc[idx], beta[idx],
                                              f_gas500[idx] *
                                              m500c[idx],
                                              r500c[idx]))
        else:
            prof_gas_med[idx] = np.ones_like(rx[idx])
            norm = tools.m_h(prof_gas_med[idx], prms.r_range_lin[idx])
            # normalize profile to f_b * m200
            prof_gas_med[idx] *= f_b[idx] * m_range[idx]/ norm

    mgas200 = tools.m_h(prof_gas_med, prms.r_range_lin)
    f_gas_med = mgas200 / m_range

    print f_gas500
    print f_gas_med
    # --------------------------------------------------------------------------
    # specific gas extra kwargs -> need f_gas
    gas_extra = {'profile': prof_gas_med / f_gas_med.reshape(-1,1),
                 'f_comp': f_gas_med}
    prof_gas_kwargs = tools.merge_dicts(profile_kwargs, gas_extra)
    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    comp_gas_kwargs = {'name': 'gas',
                      'p_lin': prms.p_lin,
                      'nu': prms.nu,
                      'fnu': prms.fnu,}
                      # 'm_fn': p.prms.m_fn,
                      # 'bias_fn': bias.bias_Tinker10,
                      # 'bias_fn_args': {'nu': prms.nu}}

    gas_kwargs = tools.merge_dicts(prof_gas_kwargs, comp_gas_kwargs)

    comp_gas_med = comp.Component(**gas_kwargs)

    return comp_gas_med

# ------------------------------------------------------------------------------
# End of load_gas_f500_fb()
# ------------------------------------------------------------------------------

def load_gas_f500_fb_plaw(prms=p.prms):
    '''
    Load gas profile with f500c=f500c_obs and f200m=f_b, since low mass
    systems cannot increase fast enough, we assume them to be uniform density

    --> No missing mass here
    '''
    # general profile kwargs to be used for all components
    profile_kwargs = {'r_range': prms.r_range_lin,
                      'm_range': prms.m_range_lin,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}

    m_range = prms.m_range_lin
    m500c = np.array([gas.m200m_to_m500c(m) for m in m_range])
    r500c = tools.mass_to_radius(m500c, 500 * prms.rho_crit * prms.h**2)
    rx = prms.r_range_lin / r500c.reshape(-1,1)

    # gas fractions
    fm_prms, f1_prms, f2_prms = d.f_gas_prms()

    # only fit median values
    f_prms = fm_prms

    f_gas500 = d.f_gas(m500c, **f_prms)
    f_b = (1 - prms.f_dm) * np.ones_like(f_gas500)

    # fit parameters
    beta, rc, m500c, r, prof_h = d.beta_mass(f_gas500, f_b)

    # good fit indices
    idcs = ~((beta == 0) | (np.abs(beta - 3) < 1e-3))

    prof_gas_med = np.zeros_like(rx)
    for idx, prof in enumerate(prof_gas_med):
        # good index
        if idcs[idx]:
            sl = (rx[idx] <= 1.)
            prof_gas_med[idx] = (prof_gas_hot(rx[idx], sl,
                                              rc[idx], beta[idx],
                                              f_gas500[idx] *
                                              m500c[idx],
                                              r500c[idx]))
        else:
            prof_gas_med[idx] = np.ones_like(rx[idx])
            norm = tools.m_h(prof_gas_med[idx], prms.r_range_lin[idx])
            # normalize profile to f_b * m200
            prof_gas_med[idx] *= f_b[idx] * m_range[idx]/ norm

    mgas200 = tools.m_h(prof_gas_med, prms.r_range_lin)
    f_gas_med = mgas200 / m_range

    print f_gas500
    print f_gas_med
    # --------------------------------------------------------------------------
    # specific gas extra kwargs -> need f_gas
    gas_extra = {'profile': prof_gas_med / f_gas_med.reshape(-1,1),
                 'f_comp': f_gas_med}
    prof_gas_kwargs = tools.merge_dicts(profile_kwargs, gas_extra)
    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    comp_gas_kwargs = {'name': 'gas',
                      'p_lin': prms.p_lin,
                      'nu': prms.nu,
                      'fnu': prms.fnu,}
                      # 'm_fn': p.prms.m_fn,
                      # 'bias_fn': bias.bias_Tinker10,
                      # 'bias_fn_args': {'nu': prms.nu}}

    gas_kwargs = tools.merge_dicts(prof_gas_kwargs, comp_gas_kwargs)

    comp_gas_med = comp.Component(**gas_kwargs)

    return comp_gas_med

# ------------------------------------------------------------------------------
# End of load_gas_f500_fb_plaw()
# ------------------------------------------------------------------------------

def load_gas_hmf_mod(prms=p.prms):
    '''
    Put missing mass
    '''
    # general profile kwargs to be used for all components
    profile_kwargs = {'r_range': prms.r_range_lin,
                      'm_range': prms.m_range_lin,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}

    m_range = prms.m_range_lin
    m500c = np.array([gas.m200m_to_m500c(m) for m in m_range])
    r500c = tools.mass_to_radius(m500c, 500 * prms.rho_crit * prms.h**2)
    rx = prms.r_range_lin / r500c.reshape(-1,1)

    # gas fractions
    fm_prms, f1_prms, f2_prms = d.f_gas_prms()
    f_prms = fm_prms

    f_gas500 = d.f_gas(m500c, **f_prms)

    # fit parameters
    corr_prms, rc_min, rc_max, rc_med, beta_med = d.prof_prms()

    prof_gas_med = np.zeros_like(rx)
    for idx, prof in enumerate(prof_gas_med):
        sl = (rx[idx] <= 1.)
        prof_gas_med[idx] = (prof_gas_hot(rx[idx], sl, rc_med, beta_med,
                                       f_gas500[idx] * m500c[idx],
                                       r500c[idx]))

    mgas200 = tools.m_h(prof_gas_med, prms.r_range_lin)
    f_gas_med = mgas200 / m_range

    print f_gas500
    print f_gas_med
    # --------------------------------------------------------------------------
    # specific gas extra kwargs -> need f_gas
    gas_extra = {'profile': prof_gas_med / f_gas_med.reshape(-1,1),
                 'f_comp': f_gas_med}
    prof_gas_kwargs = tools.merge_dicts(profile_kwargs, gas_extra)
    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    comp_gas_kwargs = {'name': 'gas',
                      'p_lin': prms.p_lin,
                      'nu': prms.nu,
                      'fnu': prms.fnu,}
                      # 'm_fn': p.prms.m_fn,
                      # 'bias_fn': bias.bias_Tinker10,
                      # 'bias_fn_args': {'nu': prms.nu}}

    gas_kwargs = tools.merge_dicts(prof_gas_kwargs, comp_gas_kwargs)

    comp_gas_med = comp.Component(**gas_kwargs3)

    return comp_gas_med

# ------------------------------------------------------------------------------
# End of load_gas_hmf_mod()
# ------------------------------------------------------------------------------

def load_saved():
    with open('obs_comp_dm.p', 'rb') as f:
        comp_dm = cPickle.load(f)
    with open('obs_comp_gas_dmo.p', 'rb') as f:
        comp_gas_dmo = cPickle.load(f)
    with open('obs_comp_gas_med.p', 'rb') as f:
        comp_gas_med = cPickle.load(f)
    with open('obs_comp_gas_q16.p', 'rb') as f:
        comp_gas_q16 = cPickle.load(f)
    with open('obs_comp_gas_q84.p', 'rb') as f:
        comp_gas_q84 = cPickle.load(f)

    return comp_dm, comp_gas_dmo, comp_gas_med, comp_gas_q16, comp_gas_q84

# ------------------------------------------------------------------------------
# End of load_saved()
# ------------------------------------------------------------------------------

def compare():
    # comp_dm_dmo = load_dm_dmo()
    # comp_dm = load_dm()
    # comp_gas_dmo = load_gas_dmo()
    # comp_gas_med = load_gas(fit='med')
    # comp_gas_q16 = load_gas(fit='q16')
    # comp_gas_q84 = load_gas(fit='q84')
    comp_dm, comp_gas_dmo, comp_gas_med, comp_gas_q16, comp_gas_q84 = load_saved()

    # pwr_dmo = power.Power([comp_dm_dmo], comp_dm_dmo.p_lin)
    pwr_dmo = power.Power([comp_dm, comp_gas_dmo], comp_dm.p_lin, name='dmo')
    pwr_med = power.Power([comp_dm, comp_gas_med], comp_dm.p_lin, name='med')
    pwr_q16 = power.Power([comp_dm, comp_gas_q16], comp_dm.p_lin, name='q16')
    pwr_q84 = power.Power([comp_dm, comp_gas_q84], comp_dm.p_lin, name='q84')

    k_range = comp_dm.k_range

    pl.set_style()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axhline(y=1, ls='--', c='k')
    ax.axhspan(0.99, 1.01, ls=None, color='k', alpha=0.2)
    ax.set_prop_cycle(pl.cycle_line())
    ax.plot(k_range, pwr_med.delta_tot/pwr_dmo.delta_tot, label=r'median')
    ax.plot(k_range, pwr_q16.delta_tot/pwr_dmo.delta_tot, label=r'q16')
    ax.plot(k_range, pwr_q84.delta_tot/pwr_dmo.delta_tot, label=r'q84')

    # Get twiny instance for main plot, to also have physical scales
    axd = ax.twiny()
    l = 2 * np.pi / k_range
    axd.plot(l, l)
    axd.set_xlim(axd.get_xlim()[::-1])
    axd.cla()

    ax.set_xscale('log')
    ax.set_ylim([0.8, 1.2])
    axd.set_xscale('log')
    axd.set_xlabel(r'$\lambda \, [\mathrm{Mpc}/h]$', labelpad=5)
    ax.set_xlabel(r'$k \, [h\,\mathrm{Mpc}^{-1}]$')
    ax.set_ylabel(r'$\Delta^2_i/\Delta^2_{\mathrm{dmo}}$')
    ax.legend(loc='best')
    plt.show()

# ------------------------------------------------------------------------------
# End of compare()
# ------------------------------------------------------------------------------

def compare_profiles(comp_dm, comp_gas_dmo, comp_gas1q84, comp_gas2q16):
    '''
    Compare the dm+gas profiles for different prms
    '''
    pl.set_style()
    fig = plt.figure(figsize=(20,6))
    ax1 = fig.add_axes([0.1, 0.1, 0.2, 0.8])
    ax2 = fig.add_axes([0.3, 0.1, 0.2, 0.8])
    ax3 = fig.add_axes([0.5, 0.1, 0.2, 0.8])
    ax4 = fig.add_axes([0.7, 0.1, 0.2, 0.8])

    masses = np.array([1e12, 1e13, 1e14, 1e15])
    axes = [ax1, ax2, ax3, ax4]
    m200 = p.prms.m_range_lin
    # r = p.prms.r_range_lin
    r200 = p.prms.r_range_lin[:,-1]

    for idx, mass in enumerate(masses):
        # find closest matching halo
        idx_match = np.argmin(np.abs(m200 - mass))
        r = comp_dm.r_range[idx_match] / r200[idx_match]
        prof = comp_dm.rho_r[idx_match]
        prof1 = comp_gas_dmo.rho_r[idx_match]
        prof2 = comp_gas1q84.rho_r[idx_match]
        prof3 = comp_gas2q16.rho_r[idx_match]

        axes[idx].plot(r, (prof * r**2), label='dm')
        axes[idx].plot(r, (prof1 * r**2), label=r'gas ref')
        axes[idx].plot(r, (prof2 * r**2), label=r'min q84')
        axes[idx].plot(r, (prof3 * r**2), label=r'max q16')

        # axes[idx].plot(r, (prof), label='dm')
        # axes[idx].plot(r, (prof1), label=r'gas dmo')
        # axes[idx].plot(r, (prof2), label=r'max q16')
        # axes[idx].plot(r, (prof3), label=r'min q84')
        axes[idx].set_title(r'$m_{200\mathrm{m}} = 10^{%.2f}\mathrm{M_\odot}$'
                        %np.log10(m200[idx_match]))
        axes[idx].set_ylim([4e11, 2e13])
        axes[idx].set_xlim([1e-2, 1])
        if idx == 0:
            text = axes[idx].set_ylabel(r'$\rho(r) \cdot (r/r_{200\mathrm{m}})^2 \, [\mathrm{M_\odot/Mpc^3}]$')
            font_properties = text.get_fontproperties()
        # need to set visibility to False BEFORE log scaling
        if idx > 0:
            ticks = axes[idx].get_xticklabels()
            # strange way of getting correct label
            ticks[-5].set_visible(False)

        axes[idx].set_xscale('log')
        axes[idx].set_yscale('log')
        if idx > 0:
            axes[idx].yaxis.set_ticklabels([])

    fig.text(0.5, 0.03,
             r'$r/r_{200\mathrm{m}}$', ha='center',
             va='center', rotation='horizontal', fontproperties=font_properties)
    ax1.legend(loc='best')
    plt.show()

# ------------------------------------------------------------------------------
# End of compare_fit_dm_bahamas()
# ------------------------------------------------------------------------------

def load_models():
    # load dm
    comp_dm = load_dm(p.prms)

    # load dmo
    comp_gas_dmo = load_gas_dmo(p.prms)

    # initialize dmo power
    pow_dmo = power.Power([comp_dm, comp_gas_dmo], comp_dm.p_lin, name='dmo')

    # load different gas models
    comp_g_min_med, comp_g_max_med, comp_g_med_med = load_gas(p.prms,fit='med')
    comp_g_min_q16, comp_g_max_q16, comp_g_med_q16 = load_gas(p.prms,fit='q16')
    comp_g_min_q84, comp_g_max_q84, comp_g_med_q84 = load_gas(p.prms,fit='q84')

    # initialize different powers
    pow_med_med = power.Power([comp_dm, comp_g_med_med], comp_dm.p_lin, name='med, med')
    pow_med_q16 = power.Power([comp_dm, comp_g_med_q16], comp_dm.p_lin, name='med, q16')
    pow_med_q84 = power.Power([comp_dm, comp_g_med_q84], comp_dm.p_lin, name='med, q84')

    pow_min_med = power.Power([comp_dm, comp_g_min_med], comp_dm.p_lin, name='min, med')
    pow_min_q16 = power.Power([comp_dm, comp_g_min_q16], comp_dm.p_lin, name='min, q16')
    pow_min_q84 = power.Power([comp_dm, comp_g_min_q84], comp_dm.p_lin, name='min, q84')

    pow_max_med = power.Power([comp_dm, comp_g_max_med], comp_dm.p_lin, name='max, med')
    pow_max_q16 = power.Power([comp_dm, comp_g_max_q16], comp_dm.p_lin, name='max, q16')
    pow_max_q84 = power.Power([comp_dm, comp_g_max_q84], comp_dm.p_lin, name='max, q84')

    return pow_med_med, pow_med_q16, pow_med_q84, pow_min_med, pow_min_q16, pow_min_q84, pow_max_med, pow_max_q16, pow_max_q84
# ------------------------------------------------------------------------------
# End of load_models()
# ------------------------------------------------------------------------------

def plot_beta_profile_dmo_presentation(comp_gas_dmo, prms=p.prms):
    '''
    Show example of gas profile for presentation with beta profile up to r200
    and missing mass
    '''
    fig = plt.figure(figsize=(10,9))
    ax = fig.add_subplot(111)

    idx = 50
    r200m = p.prms.r_h

    pl.set_style('line')
    ax.plot(comp_gas_dmo.r_range[idx] / r200m[idx],
            comp_gas_dmo.rho_r[idx] * comp_gas_dmo.f_comp[idx], label=r'$\beta$')

    ax.set_xlim(1e-3, 3)
    ax.set_ylim(1e11, 3e14)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(r'$r/r_\mathrm{200m}$')
    ax.set_ylabel(r'$\rho(r) \, [\mathrm{M_\odot/Mpc^3}]$')
    ax.set_title(r'$m_{\mathrm{200m}} = 10^{%.1f} \mathrm{M_\odot}$'%np.log10(p.prms.m_range_lin[idx]), y=1.015, fontsize=42)
    ax.legend(loc='best')
    plt.savefig('hm_beta_dmo.pdf', transparent=True)
    plt.show()

# ------------------------------------------------------------------------------
# End of plot_beta_profile_presentation()
# ------------------------------------------------------------------------------

def plot_beta_profile_presentation(comp_gas, prms=p.prms):
    '''
    Show example of gas profile for presentation with beta profile up to r200
    and missing mass
    '''
    fig = plt.figure(figsize=(10,9))
    ax = fig.add_subplot(111)

    idx = 50
    r200m = p.prms.r_h

    pl.set_style('line')
    ax.plot(comp_gas.r_range[idx] / r200m[idx],
            comp_gas.rho_r[idx] * comp_gas.f_comp[idx], label=r'$\beta$')

    ax.set_xlim(1e-3, 3)
    ax.set_ylim(1e11, 3e14)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(r'$r/r_\mathrm{200m}$')
    ax.set_ylabel(r'$\rho(r) \, [\mathrm{M_\odot/Mpc^3}]$')
    ax.set_title(r'$m_{\mathrm{200m}} = 10^{%.1f} \mathrm{M_\odot}$'%np.log10(p.prms.m_range_lin[idx]), y=1.015, fontsize=42)
    ax.legend(loc='best')
    plt.savefig('hm_beta.pdf', transparent=True)
    plt.show()

# ------------------------------------------------------------------------------
# End of plot_beta_profile_presentation()
# ------------------------------------------------------------------------------

def plot_beta_profile_r500_presentation(comp_gas_r500, comp_gas_smooth_r500,
                                        prms=p.prms):
    '''
    Show example of gas profile for presentation with beta profile up to r500
    and missing mass between r500c and r200m
    '''
    fig = plt.figure(figsize=(10,9))
    ax = fig.add_subplot(111)

    idx = 50
    r200m = p.prms.r_h

    pl.set_style('line')
    ax.plot(comp_gas_r500.r_range[idx] / r200m[idx],
            comp_gas_r500.rho_r[idx] * comp_gas_r500.f_comp[idx], label=r'$\beta$')
    ax.plot(comp_gas_smooth_r500.r_range[idx] / r200m[idx],
            comp_gas_smooth_r500.rho_r[idx] * comp_gas_smooth_r500.f_comp[idx],
            label=r'smooth $r_\mathrm{500c}-r_\mathrm{200m}$')

    ax.set_xlim(1e-3, 3)
    ax.set_ylim(1e11, 3e14)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(r'$r/r_\mathrm{200m}$')
    ax.set_ylabel(r'$\rho(r) \, [\mathrm{M_\odot/Mpc^3}]$')
    ax.set_title(r'$m_{\mathrm{200m}} = 10^{%.1f} \mathrm{M_\odot}$'%np.log10(p.prms.m_range_lin[idx]), y=1.015, fontsize=42)
    ax.legend(loc='best')
    plt.savefig('hm_beta_r500.pdf', transparent=True)
    plt.show()

# ------------------------------------------------------------------------------
# End of plot_beta_profile_presentation()
# ------------------------------------------------------------------------------

def plot_beta_profile_r200_presentation(comp_gas_r200, comp_gas_smooth_r200,
                                        prms=p.prms):
    '''
    Show example of gas profile for presentation with beta profile up to r200
    and missing mass between r200m and 5r500c
    '''
    fig = plt.figure(figsize=(10,9))
    ax = fig.add_subplot(111)

    idx = 50
    r200m = p.prms.r_h

    pl.set_style('line')
    ax.plot(comp_gas_r200.r_range[idx] / r200m[idx],
            comp_gas_r200.rho_r[idx] * comp_gas_r200.f_comp[idx], label=r'$\beta$')
    ax.plot(comp_gas_smooth_r200.r_range[idx] / r200m[idx],
            comp_gas_smooth_r200.rho_r[idx] * comp_gas_smooth_r200.f_comp[idx],
            label=r'smooth $r_\mathrm{200m}-5\,r_\mathrm{500c}$')

    ax.set_xlim(1e-3, 3)
    ax.set_ylim(1e11, 3e14)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(r'$r/r_\mathrm{200m}$')
    ax.set_ylabel(r'$\rho(r) \, [\mathrm{M_\odot/Mpc^3}]$')
    ax.set_title(r'$m_{\mathrm{200m}} = 10^{%.1f} \mathrm{M_\odot}$'%np.log10(p.prms.m_range_lin[idx]), y=1.015, fontsize=42)
    ax.legend(loc='best')
    plt.savefig('hm_beta_r200.pdf', transparent=True)
    plt.show()

# ------------------------------------------------------------------------------
# End of plot_beta_profile_presentation()
# ------------------------------------------------------------------------------
