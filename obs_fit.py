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
                                 'rho_mean': 1.},
                'profile_f_args': {'c_x': c_x,
                                   'r_x': r_x,
                                   'rho_mean': 1.},
                'f_comp': f_dm}
    prof_dm_kwargs = tools.merge_dicts(profile_kwargs, dm_extra)
    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    comp_dm_kwargs = {'name': 'dm',
                      'p_lin': prms.p_lin,
                      'nu': prms.nu,
                      'fnu': prms.fnu,
                      # 'm_fn': prms.m_fn,
                      'bias_fn': bias.bias_Tinker10,
                      'bias_fn_args': {'nu': prms.nu}}

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
                                 'rho_mean': 1.},
                'profile_f_args': {'c_x': c_x,
                                   'r_x': r_x,
                                   'rho_mean': 1.},
                'f_comp': f_dm}
    prof_dm_kwargs = tools.merge_dicts(profile_kwargs, dm_extra)
    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    comp_dm_kwargs = {'name': 'dm',
                      'p_lin': prms.p_lin,
                      'nu': prms.nu,
                      'fnu': prms.fnu,
                      # 'm_fn': prms.m_fn,
                      'bias_fn': bias.bias_Tinker10,
                      'bias_fn_args': {'nu': prms.nu}}

    dm_kwargs = tools.merge_dicts(prof_dm_kwargs, comp_dm_kwargs)

    comp_dm = comp.Component(**dm_kwargs)

    if save:
        with open('obs_comp_dm.p', 'wb') as f:
            cPickle.dump(comp_dm, f)

    return comp_dm

# ------------------------------------------------------------------------------
# End of load_dm()
# ------------------------------------------------------------------------------

def prof_gas_hot(x, sl, a, b, m_sl, r500):
    '''beta profile'''
    profile = (1 + (x/a)**2)**(-b/2)
    mass = tools.m_h(profile[sl], x[sl] * r500)
    profile *= m_sl/mass

    return profile

def load_gas(prms=p.prms, fit='med', save=True):
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
    if fit == 'med':
        f_prms = fm_prms
    elif fit == 'q16':
        f_prms = f1_prms
    elif fit == 'q84':
        f_prms = f2_prms

    f_gas500 = d.f_gas(m500c, **f_prms)

    # fit parameters
    rc_med, rc_q16, rc_q84, b_med, b_q16, b_q84 = d.prof_prms()
    if fit == 'med':
        rc = rc_med
        bc = b_med
    elif fit == 'q16':
        rc = rc_q16
        bc = b_q16
    elif fit == 'q84':
        rc = rc_q84
        bc = b_q84
    prof_gas = np.zeros_like(rx)
    for idx, prof in enumerate(prof_gas):
        sl = (rx[idx] <= 1.)
        prof_gas[idx] = (prof_gas_hot(rx[idx], sl, rc, bc,
                                      f_gas500[idx] * m500c[idx],
                                      r500c[idx])
                         * np.exp(-rx[idx]/2.))
        # plt.plot(rx[idx], prof_gas[idx])
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()

    mgas200 = tools.m_h(prof_gas, prms.r_range_lin)
    f_gas = mgas200 / m_range
    print f_gas500
    print f_gas
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
                      # 'm_fn': prms.m_fn,
                      'bias_fn': bias.bias_Tinker10,
                      'bias_fn_args': {'nu': prms.nu}}

    gas_kwargs = tools.merge_dicts(prof_gas_kwargs, comp_gas_kwargs)

    comp_gas = comp.Component(**gas_kwargs)
    if save:
        with open('obs_comp_gas_%s.p'%fit, 'wb') as f:
            cPickle.dump(comp_gas, f)

    return comp_gas

# ------------------------------------------------------------------------------
# End of load_gas()
# ------------------------------------------------------------------------------

def compare():
    comp_dm_dmo = load_dm_dmo()
    comp_dm = load_dm()
    comp_gas_med = load_gas(fit='med')
    comp_gas_q16 = load_gas(fit='q16')
    comp_gas_q84 = load_gas(fit='q84')

    pwr_dmo = power.Power([comp_dm_dmo], comp_dm_dmo.p_lin)
    pwr_med = power.Power([comp_dm, comp_gas_med], comp_dm.p_lin)
    pwr_q16 = power.Power([comp_dm, comp_gas_q16], comp_dm.p_lin)
    pwr_q84 = power.Power([comp_dm, comp_gas_q84], comp_dm.p_lin)

    k_range = comp_dm.k_range

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(k_range, pwr_med.delta_tot/pwr_dmo.delta_tot, label=r'median')
    ax.plot(k_range, pwr_q16.delta_tot/pwr_dmo.delta_tot, label=r'q16')
    ax.plot(k_range, pwr_q84.delta_tot/pwr_dmo.delta_tot, label=r'q84')
    ax.axhline(y=1, ls='--', c='k')

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
    ax.set_ylabel(r'$\Delta^2_i/\Delta^2_{\mathrm{DMO}}$')
    ax.legend(loc='best')
    plt.show()
