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

    beta, rc, m500c, r, prof_h = d.beta_mass(prms)
    f_gas = (1 - prms.f_dm) * np.ones_like(prms.m_range_lin)
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
                      # 'm_fn': p.prms_dmo.m_fn,
                      'bias_fn': bias.bias_Tinker10,
                      'bias_fn_args': {'nu': prms.nu}}

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
    corr_prms, rc_min, rc_max, rc_med, beta_med = d.prof_prms()
    beta_min = np.power(10, d.corr_fit(np.log10(rc_min), **corr_prms))
    beta_max = np.power(10, d.corr_fit(np.log10(rc_max), **corr_prms))

    prof_gas_min = np.zeros_like(rx)
    prof_gas_max = np.zeros_like(rx)
    prof_gas_med = np.zeros_like(rx)
    for idx, prof in enumerate(prof_gas_min):
        sl = (rx[idx] <= 1.)
        prof_gas_min[idx] = (prof_gas_hot(rx[idx], sl, rc_min, beta_min,
                                       f_gas500[idx] * m500c[idx],
                                       r500c[idx]))
        prof_gas_max[idx] = (prof_gas_hot(rx[idx], sl, rc_max, beta_max,
                                       f_gas500[idx] * m500c[idx],
                                       r500c[idx]))
        prof_gas_med[idx] = (prof_gas_hot(rx[idx], sl, rc_med, beta_med,
                                       f_gas500[idx] * m500c[idx],
                                       r500c[idx]))

    # # Look at gas fraction -> may not exceed baryon fraction
    # # as soon as fraction exceeds baryon, extrapolate  r^-3 slope
    # m_range = prms.m_range_lin
    # m200c = np.array([gas.m200m_to_m200c(m) for m in m_range])
    # r200c = tools.mass_to_radius(m200c, 200 * prms.rho_crit * prms.h**2)
    # # --------------------------------------------------------------------------
    # # Correa
    # c_x = profs.c_correa(m200c, 0).reshape(-1) * prms.r_range_lin[:,-1]/r200c
    # r_x = prms.r_range_lin[:,-1]
    # r_s = r_x / c_x

    # comp_dm = load_dm(prms)
    # m_dm = np.array([tools.m_h(comp_dm.rho_r[:,:i] * comp_dm.f_comp.reshape(-1,1),
    #                            comp_dm.r_range[:,:i])
    #                  for i in range(1, comp_dm.r_range.shape[-1])])
    # m_gas1 = np.array([tools.m_h(prof_gas1[:,:i], prms.r_range_lin[:,:i])
    #                    for i in range(1, prms.r_range_lin.shape[-1])])
    # m_gas2 = np.array([tools.m_h(prof_gas1[:,:i], prms.r_range_lin[:,:i])
    #                    for i in range(1, prms.r_range_lin.shape[-1])])

    # m_tot1 = m_dm + m_gas1
    # m_tot2 = m_dm + m_gas2

    # idx1 = np.argmax((m_gas1 / m_tot1 > 1 - prms.f_dm), axis=0)
    # idx2 = np.argmax((m_gas2 / m_tot1 > 1 - prms.f_dm), axis=0)

    # for idx, r in enumerate(prms.r_range_lin):
    #     rs = r_s[idx]
    #     idx_sl = np.argmax(r >= rs)
    #     sl = (r >= rs)
    #     prof_gas1[idx, idx_sl:] = (prof_gas1[idx, idx_sl] * 4 /
    #                                ((r[sl]/rs) * (1 + r[sl]/rs)**2))

    mgas200_1 = tools.m_h(prof_gas_min, prms.r_range_lin)
    f_gas_min = mgas200_1 / m_range
    mgas200_2 = tools.m_h(prof_gas_max, prms.r_range_lin)
    f_gas_max = mgas200_2 / m_range
    mgas200_3 = tools.m_h(prof_gas_med, prms.r_range_lin)
    f_gas_med = mgas200_3 / m_range

    print f_gas500
    print f_gas_min
    print f_gas_max
    print f_gas_med
    # --------------------------------------------------------------------------
    # specific gas extra kwargs -> need f_gas
    gas_extra1 = {'profile': prof_gas_min / f_gas_min.reshape(-1,1),
                  'f_comp': f_gas_min}
    gas_extra2 = {'profile': prof_gas_max / f_gas_max.reshape(-1,1),
                  'f_comp': f_gas_max}
    gas_extra3 = {'profile': prof_gas_med / f_gas_med.reshape(-1,1),
                  'f_comp': f_gas_med}
    prof_gas_kwargs1 = tools.merge_dicts(profile_kwargs, gas_extra1)
    prof_gas_kwargs2 = tools.merge_dicts(profile_kwargs, gas_extra2)
    prof_gas_kwargs3 = tools.merge_dicts(profile_kwargs, gas_extra3)
    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    comp_gas_kwargs = {'name': 'gas',
                      'p_lin': prms.p_lin,
                      'nu': prms.nu,
                      'fnu': prms.fnu,
                      # 'm_fn': p.prms_dmo.m_fn,
                      'bias_fn': bias.bias_Tinker10,
                      'bias_fn_args': {'nu': prms.nu}}

    gas_kwargs1 = tools.merge_dicts(prof_gas_kwargs1, comp_gas_kwargs)
    gas_kwargs2 = tools.merge_dicts(prof_gas_kwargs2, comp_gas_kwargs)
    gas_kwargs3 = tools.merge_dicts(prof_gas_kwargs3, comp_gas_kwargs)

    comp_gas_min = comp.Component(**gas_kwargs1)
    comp_gas_max = comp.Component(**gas_kwargs2)
    comp_gas_med = comp.Component(**gas_kwargs3)
    if save:
        with open('obs_comp_gas_min_%s.p'%fit, 'wb') as f:
            cPickle.dump(comp_gas_min, f)
        with open('obs_comp_gas_max_%s.p'%fit, 'wb') as f:
            cPickle.dump(comp_gas_max, f)
        with open('obs_comp_gas_med_%s.p'%fit, 'wb') as f:
            cPickle.dump(comp_gas_med, f)

    return comp_gas_min, comp_gas_max, comp_gas_med

# ------------------------------------------------------------------------------
# End of load_gas()
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

def load_pows(prms=p.prms):
    comp_dm = load_dm(prms)

    


