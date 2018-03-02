import numpy as np
import scipy.optimize as opt
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import sys

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

import pdb

def fdmo_to_f200(mdmo, fdmo):
    '''
    Velliscig conversion formula for the hmf
    '''
    a = -0.0872
    b = 0.0872
    c = -13.6339
    d = 0.3509

    log_ratio = a + b / (1 + np.exp(-(np.log10(mdmo) + c) / d))
    f200 = fdmo * np.power(10, log_ratio)

    return f200

def mdmo_to_m200(mdmo):
    '''
    Velliscig conversion formula -> we need mdmo > 1e11
    '''
    a = -0.1077
    b = 0.1077
    c = -13.5715
    d = 0.2786

    log_ratio = a + b / (1 + np.exp(-(np.log10(mdmo) + c) / d))
    m200 = mdmo * np.power(10, log_ratio)

    return m200

def m200_to_mdmo(m200):
    m2mdmo = tools.inverse(mdmo_to_m200)
    mdmo = np.ones_like(m200)
    for idx, val in enumerate(m200):
        mdmo[idx] = m2mdmo(val)

    return mdmo

def prof_dm_inner(r, sl, ri, b, m_sl):
    '''
    Profile for inner dark matter bump

    r in physical coordinates, sl denotes r/r200 <= 0.05
    '''
    profile = np.exp(-(np.log10(r) - np.log10(ri))**2/b)
    mass = tools.m_h(profile[sl], r[sl])
    profile *= m_sl/mass
    # y = r / ri
    # profile  = (y)**(-1) * (1 + y**2)**(-1)
    # mass = tools.m_h(profile[sl], r[sl])
    # profile *= m_sl / mass

    return profile

def prof_nfw(r, sl, c, m_sl):
    '''
    Normal NFW profile

    r in physical coordinates, sl denotes r/r200 > 0.05
    '''
    x = c * r/r[-1]
    profile  = (x)**(-1) * (1 + x)**(-2)
    mass = tools.m_h(profile[sl], r[sl])
    profile *= m_sl/mass

    return profile

def prof_gas_warm(x, a, b, m, r200):
    '''einasto profile'''
    profile = np.exp(-(x/a)**b)
    mass = tools.m_h(profile, x * r200)
    profile *= m/mass

    return profile

def prof_gas_hot_c(x, a, b, c, m, r200):
    '''beta profile'''
    profile = (1 + (x/a)**2)**(-b/2) * np.exp(-(x/c)**2)
    mass = tools.m_h(profile, x * r200)
    profile *= m/mass

    return profile

def prof_gas_hot_s(x, a, b, m, r200):
    '''lognormal'''
    profile = np.exp(-(np.log10(x) - np.log10(a))**2/b)
    mass = tools.m_h(profile, x * r200)
    profile *= m/mass

    return profile

def S(m, mc, t):
    '''
    Sigmoid function
    '''
    return (m/mc)**t/(1 + (m/mc)**t)

def prof_stars_s(x, a, b, m, r200):
    '''lognormal'''
    profile = np.exp(-(np.log10(x) - np.log10(a))**2/b)
    mass = tools.m_h(profile, x * r200)
    profile *= m/mass

    return profile

def prof_stars_c(x, a, b, c, m, r200):
    profile = (1 + (x/a)**b)**(-1) * np.exp(-(x/c))
    # profile = (np.exp(-(x/a)**b) + np.exp(-x/c)) * np.exp(-(x/d))
    mass = tools.m_h(profile, x * r200)
    profile *= m/mass

    return profile

def c_dmo(m):
    A = 8.7449969011763216
    B = -0.093399926987858539
    plaw =  A * (m/1e14)**B
    return plaw

def load_dm_dmo1(prms):
    m_range = mdmo_to_m200(prms.m200m)
    # general profile kwargs to be used for all components
    profile_kwargs = {'r_range': prms.r_range_lin,
                      'm_bar': m_range,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}


    # --------------------------------------------------------------------------
    # DMO fit
    f_dm = np.ones_like(m_range)
    c_x = c_dmo(m_range)
    r_x = tools.mass_to_radius(m_range, 200 * prms.rho_m)# prms.r_range_lin[:,-1]
    # specific dm extra kwargs
    dm_extra = {'profile': profs.profile_NFW,
                'profile_f': profs.profile_NFW_f,
                'profile_args': {'c_x': c_x,
                                 'r_x': r_x,
                                 'rho_mean': 1.},
                'profile_f_args': {'c_x': c_x,
                                   'r_x': r_x,
                                   'rho_mean': 1.}}
    prof_dm_kwargs = tools.merge_dicts(profile_kwargs, dm_extra)
    # --------------------------------------------------------------------------

    # additional kwargs for comp.Component
    comp_dm_kwargs = {'name': 'dm',
                      'p_lin': prms.p_lin,
                      'nu': prms.nu,
                      'fnu': prms.fnu,
                      'm_fn': prms.m_fn,
                      'f_comp': f_dm,
                      'bias_fn': bias.bias_Tinker10,
                      'bias_fn_args': {'nu': prms.nu}}

    dm_kwargs = tools.merge_dicts(prof_dm_kwargs, comp_dm_kwargs)

    comp_dm = comp.Component(**dm_kwargs)

    return comp_dm

# ------------------------------------------------------------------------------
# End of load_dm_dmo()
# ------------------------------------------------------------------------------

def load_dm_dmo2(prms):
    # general profile kwargs to be used for all components
    profile_kwargs = {'r_range': prms.r_range_lin,
                      'm_bar': prms.m200m,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}


    m_range = prms.m200m
    # --------------------------------------------------------------------------
    # DMO fit
    f_dm = np.ones_like(m_range)
    c_x = c_dmo(m_range)
    r_x = prms.r_range_lin[:,-1]
    # specific dm extra kwargs
    dm_extra = {'profile': profs.profile_NFW,
                'profile_f': profs.profile_NFW_f,
                'profile_args': {'c_x': c_x,
                                 'r_x': r_x,
                                 'rho_mean': 1.},
                'profile_f_args': {'c_x': c_x,
                                   'r_x': r_x,
                                   'rho_mean': 1.}}
    prof_dm_kwargs = tools.merge_dicts(profile_kwargs, dm_extra)
    # --------------------------------------------------------------------------
    # additional kwargs for comp.Component
    comp_dm_kwargs = {'name': 'dm',
                      'p_lin': prms.p_lin,
                      'nu': prms.nu,
                      'fnu': prms.fnu,
                      'm_fn': prms.m_fn,
                      'f_comp': f_dm,
                      'bias_fn': bias.bias_Tinker10,
                      'bias_fn_args': {'nu': prms.nu}}

    dm_kwargs = tools.merge_dicts(prof_dm_kwargs, comp_dm_kwargs)

    comp_dm = comp.Component(**dm_kwargs)

    return comp_dm

# ------------------------------------------------------------------------------
# End of load_dm_dmo()
# ------------------------------------------------------------------------------

def load_dm(prms=p.prmst):
    # general profile kwargs to be used for all components
    profile_kwargs = {'r_range': prms.r_range_lin,
                      'm_bar': prms.m200m,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}


    m_range = prms.m200m
    # # --------------------------------------------------------------------------
    # # DMO fit
    # f_dm = np.ones_like(m_range)
    # c_x = c_dmo(m_range)
    # r_x = prms.r_range_lin[:,-1]
    # # specific dm extra kwargs
    # dm_extra = {'profile': profs.profile_NFW,
    #             'profile_f': profs.profile_NFW_f,
    #             'profile_args': {'c_x': c_x,
    #                              'r_x': r_x,
    #                              'rho_mean': 1.},
    #             'profile_f_args': {'c_x': c_x,
    #                                'r_x': r_x,
    #                                'rho_mean': 1.}}
    # prof_dm_kwargs = tools.merge_dicts(profile_kwargs, dm_extra)
    # # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Test BAHAMAS fit
    # extract all fit parameters
    rs, rs_err, c, c_err, ri, ri_err, bi, bi_err, m1, m2, m, m200, r200,  ri_prms, bi_prms, rs_prms, c_prms, m1_prms, m2_prms, f_prms = b.fit_dm_bahamas_bar()

    ri = b.dm_plaw_fit(m_range, **ri_prms) * prms.r_range_lin[:,-1]
    bi = b.dm_plaw_fit(m_range, **bi_prms)
    # rs = b.dm_plaw_fit(m_range, **rs_prms)
    c_x = b.dm_c_fit(m_range, **c_prms)
    c_x[m_range < 2e11] = b.dm_c_dmo(m_range[m_range < 2e11])

    mi = m_range * b.dm_m1_fit(m_range, **m1_prms)
    ms = m_range * b.dm_m2_fit(m_range, **m2_prms)
    f_dm = b.dm_f_fit(m_range, **f_prms)
    # f_dm = np.ones_like(m_range) * prms.f_dm
    prof_dm = np.zeros((m_range.shape[0], prms.r_range_lin.shape[1]),
                       dtype=float)
    for idx, m in enumerate(m_range):
        r = prms.r_range_lin[idx]
        r_x = prms.r_range_lin[idx,-1]
        x = r / r_x
        sl1 = (x > 0.05)
        sl2 = (x <= 0.05)
        p_nfw = prof_nfw(r, sl1, c_x[idx], ms[idx])
        if m >= 1e11:
            p_bar = prof_dm_inner(r, sl2, ri[idx], bi[idx], mi[idx])
            mass = tools.m_h(p_nfw + p_bar, r)
            prof_dm[idx] = (p_nfw + p_bar) * m / mass
        else:
            mass = tools.m_h(p_nfw, r)
            prof_dm[idx] = p_nfw * m / mass


    prof_dm_kwargs = tools.merge_dicts(profile_kwargs, {'profile': prof_dm})
    # --------------------------------------------------------------------------

    # additional kwargs for comp.Component
    comp_dm_kwargs = {'name': 'dm',
                      'p_lin': prms.p_lin,
                      'nu': prms.nu,
                      'fnu': prms.fnu,
                      'm_fn': prms.m_fn,
                      'f_comp': f_dm,
                      'bias_fn': bias.bias_Tinker10,
                      'bias_fn_args': {'nu': prms.nu}}

    dm_kwargs = tools.merge_dicts(prof_dm_kwargs, comp_dm_kwargs)

    comp_dm = comp.Component(**dm_kwargs)

    return comp_dm

def alpha_power(k_range, alpha, P1, P2):
    # f = 0.188 * p.prms.sigma_8**(4.29)
    # sv = 2.
    # Pn = P2 * (1 - f*np.tanh(k_range * sv/np.sqrt(f))**2)
    # return (P1**alpha + Pn**alpha)**(1./alpha)
    return (P1**alpha + P2**alpha)**(1./alpha)

def fit_alpha(k_range, P1, P2, power):
    sl = (k_range > 2e-1)
    popt, pcov = opt.curve_fit(lambda k_range, alpha: \
                               alpha_power(k_range, alpha, P1[sl], P2[sl]),
                               k_range[sl], power[sl],
                               bounds=([0.5, 2]))

    alpha = popt[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # r8 = np.load('8bar_8dmo.npy')
    # ax.plot(k_range, alpha_power(k_range, popt[0], P1, P2)/power,
    #         label=r'$(P_{\mathrm{1h}}^{\alpha} + P_{\mathrm{2h}}^{\alpha})^{1/\alpha}$')
    # ax.plot(k_range, (P1+P2)/power, label=r'$P_{\mathrm{1h}} + P_{\mathrm{2h}}$')
    # ax.axhline(y=1, c='k', ls='--')

    ax.plot(k_range, (P1**alpha + P2**alpha)**(1./alpha)/power,
            label=r'$(P_{\mathrm{1h}}^{\alpha} + P_{\mathrm{2h}}^{\alpha})^{1/\alpha}$')
    ax.plot(k_range, (P1+P2)/power, label=r'$P_{\mathrm{1h}} + P_{\mathrm{2h}}$')
    # plt.plot(k_range, power)
    ax.axhline(y=1, c='k', ls='--')
    minor_locator = AutoMinorLocator(2)
    ax.yaxis.set_minor_locator(minor_locator)
    # ax.yaxis.grid(which='both')
    # ax.xaxis.grid()
    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_ylim([0.75,1.4])
    ax.set_xlabel(r'$k \, [h\,\mathrm{Mpc}^{-1}]$')
    ax.set_ylabel(r'ratio')# , rotation=270, labelpad=20)
    leg = ax.legend(loc='best', frameon=True, framealpha=1.)
    leg.get_frame().set_linewidth(0.0)
    plt.show()

    return popt[0]

def load_gas(prms=p.prmst):
    # general profile kwargs to be used for all components
    profile_kwargs = {'r_range': prms.r_range_lin,
                      'm_bar': prms.m200m,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}
    # extract all fit parameters
    rw_prms, sw_prms, rc_prms, b_prms, rs_prms, ss_prms, mw_prms, mc_prms, ms_prms, r0w_prms, r0s_prms = b.plot_gas_fit_bahamas_median()


    f_w = b.gas_warm_mw_fit(prms.m200m, **mw_prms)
    fc_h = b.gas_hot_mc_fit(prms.m200m, **mc_prms)
    fs_h = b.gas_hot_ms_fit(prms.m200m, **ms_prms)

    # # reshape profile parameters
    # r_x = (prms.r200m / prms.h)
    # m = (prms.m200m / prms.h)
    m = prms.m200m
    r_x = prms.r200m

    ############################################################################
    # Need to slice with r0
    # Need to remove masses for which we do not have profile
    rw = b.gas_warm_rw_fit(m, **rw_prms)# * prms.h
    sw = b.gas_warm_sigma_fit(m, **sw_prms)
    r0w = b.gas_warm_r0_fit(m, **r0w_prms)# * prms.h
    prof_w = np.zeros_like(prms.r_range_lin)
    for idx, mass in enumerate(m):
        if mass <= 1e13:# * prms.h:
            r_sl = (prms.r_range_lin[idx] >= r0w[idx] * r_x[idx])
            if r_sl.sum() > 0:
                prof_w[idx, r_sl] = prof_gas_warm((prms.r_range_lin[idx, r_sl]/
                                                   r_x[idx]), rw[idx], sw[idx],
                                                  m[idx],
                                                  r_x[idx])

    w_kwargs = tools.merge_dicts(profile_kwargs, {'profile': prof_w})
    comp_w_kwargs = {'name': 'warm',
                     'p_lin': prms.p_lin,
                     'nu': prms.nu,
                     'fnu': prms.fnu,
                     # 'm_fn': prms.m_fn,
                     'f_comp': f_w,
                     'bias_fn': bias.bias_Tinker10,
                     'bias_fn_args': {'nu': prms.nu}}
    w_kwargs = tools.merge_dicts(w_kwargs, comp_w_kwargs)
    comp_w = comp.Component(**w_kwargs)


    rc = b.gas_hot_rc_fit(m, **rc_prms)# * prms.h
    bc = b.gas_hot_beta_fit(m, **b_prms)
    rs = b.gas_hot_rs_fit(m, **rs_prms)# * prms.h
    # prof_c = prof_gas_hot_c((prms.r_range_lin/r_x) / prms.h, rc, bc, rs, m, r_x)
    prof_c = np.zeros_like(prms.r_range_lin)
    for idx, mass in enumerate(m):
        if (mass >= 1e13):
            prof_c[idx] = prof_gas_hot_c((prms.r_range_lin[idx]/r_x[idx]),
                                         rc[idx], bc[idx], rs[idx],
                                         m[idx], r_x[idx])


    c_kwargs = tools.merge_dicts(profile_kwargs, {'profile': prof_c})
    comp_c_kwargs = {'name': 'cen',
                     'p_lin': prms.p_lin,
                     'nu': prms.nu,
                     'fnu': prms.fnu,
                     'm_fn': prms.m_fn,
                     'f_comp': fc_h,
                     'bias_fn': bias.bias_Tinker10,
                     'bias_fn_args': {'nu': prms.nu}}
    c_kwargs = tools.merge_dicts(c_kwargs, comp_c_kwargs)
    comp_c = comp.Component(**c_kwargs)


    ss = b.gas_hot_sigma_fit(m, **ss_prms)
    r0s = b.gas_hot_r0_fit(m, **r0s_prms)# * prms.h
    # prof_s = prof_gas_hot_s((prms.r_range_lin/r_x) / prms.h, rs, ss, m, r_x)
    prof_s = np.zeros_like(prms.r_range_lin)
    for idx, mass in enumerate(m):
        if mass >= 1e13: #* prms.h:
            r_sl = (prms.r_range_lin[idx] >= r0s[idx] * r_x[idx])
            if r_sl.sum() > 0:
                prof_s[idx, r_sl] = prof_gas_hot_s((prms.r_range_lin[idx, r_sl]/
                                                    r_x[idx]), rs[idx], ss[idx],
                                                   m[idx], r_x[idx])


    s_kwargs = tools.merge_dicts(profile_kwargs, {'profile': prof_s})
    comp_s_kwargs = {'name': 'sat',
                     'p_lin': prms.p_lin,
                     'nu': prms.nu,
                     'fnu': prms.fnu,
                     # 'm_fn': prms.m_fn,
                     'f_comp': fs_h,
                     'bias_fn': bias.bias_Tinker10,
                     'bias_fn_args': {'nu': prms.nu}}
    s_kwargs = tools.merge_dicts(s_kwargs, comp_s_kwargs)
    comp_s = comp.Component(**s_kwargs)

    comp_gas = comp_w + comp_c + comp_s
    comp_gas.name = 'gas'

    return comp_gas

# ------------------------------------------------------------------------------
# End of load_gas()
# ------------------------------------------------------------------------------

def load_stars(prms=p.prmst):
    # general profile kwargs to be used for all components
    profile_kwargs = {'r_range': prms.r_range_lin,
                      'm_bar': prms.m200m,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}
    # extract all fit parameters
    as_prms, bs_prms, r0_prms, ms_prms, ac_prms, bc_prms, cc_prms, mc_prms = b.plot_stars_fit_bahamas_median()


    fc = b.stars_mc_fit(prms.m200m, **mc_prms)
    fs = b.stars_ms_fit(prms.m200m, **ms_prms)
    f_cold = 0.4 * fc

    # # reshape profile parameters
    # r_x = (prms.r200m / prms.h)
    # m = (prms.m200m / prms.h)
    r_x = prms.r200m
    m = prms.m200m

    ############################################################################
    # Satellites
    # Need to slice with r0
    rs = b.dm_plaw_fit(m, **as_prms)# * prms.h
    ss = b.dm_plaw_fit(m, **bs_prms)
    r0w = b.dm_plaw_fit(m, **r0_prms)# * prms.h
    prof_s = np.zeros_like(prms.r_range_lin)
    for idx, mass in enumerate(m):
        r_sl = (prms.r_range_lin[idx] >= r0w[idx] * r_x[idx])
        if r_sl.sum() > 0:
            prof_s[idx, r_sl] = prof_stars_s((prms.r_range_lin[idx, r_sl]/
                                              r_x[idx]), rs[idx], ss[idx],
                                             m[idx],
                                             r_x[idx])

    s_kwargs = tools.merge_dicts(profile_kwargs, {'profile': prof_s})
    comp_s_kwargs = {'name': r'\star sat',
                     'p_lin': prms.p_lin,
                     'nu': prms.nu,
                     'fnu': prms.fnu,
                     'm_fn': prms.m_fn,
                     'f_comp': fs,
                     'bias_fn': bias.bias_Tinker10,
                     'bias_fn_args': {'nu': prms.nu}}
    s_kwargs = tools.merge_dicts(s_kwargs, comp_s_kwargs)
    comp_s = comp.Component(**s_kwargs)


    rc = b.dm_plaw_fit(m, **ac_prms)# * prms.h
    bc = b.dm_plaw_fit(m, **bc_prms)
    cc = b.dm_plaw_fit(m, **cc_prms)# * prms.h
    prof_c = np.zeros_like(prms.r_range_lin)
    prof_cold = np.zeros_like(prms.r_range_lin)
    for idx, mass in enumerate(m):
        prof_c[idx] = prof_gas_hot_c((prms.r_range_lin[idx]/r_x[idx]),
                                     rc[idx], bc[idx], cc[idx],
                                     m[idx], r_x[idx])

    c_kwargs = tools.merge_dicts(profile_kwargs, {'profile': prof_c})
    comp_c_kwargs = {'name': r'\star cen',
                     'p_lin': prms.p_lin,
                     'nu': prms.nu,
                     'fnu': prms.fnu,
                     'm_fn': prms.m_fn,
                     'f_comp': fc,
                     'bias_fn': bias.bias_Tinker10,
                     'bias_fn_args': {'nu': prms.nu}}
    comp_cold_kwargs = {'name': r'cold',
                        'p_lin': prms.p_lin,
                        'nu': prms.nu,
                        'fnu': prms.fnu,
                        'm_fn': prms.m_fn,
                        'f_comp': f_cold,
                        'bias_fn': bias.bias_Tinker10,
                        'bias_fn_args': {'nu': prms.nu}}
    c_kwargs = tools.merge_dicts(c_kwargs, comp_c_kwargs)
    cold_kwargs = tools.merge_dicts(c_kwargs, comp_cold_kwargs)
    comp_c = comp.Component(**c_kwargs)
    comp_cold = comp.Component(**cold_kwargs)

    comp_stars = comp_c + comp_s
    comp_stars.name = 'stars'

    return comp_stars, comp_cold

# ------------------------------------------------------------------------------
# End of load_stars()
# ------------------------------------------------------------------------------

def load_masses():
    comp_dm1 = load_dm(p.prms1)
    comp_dm2 = load_dm(p.prms2)
    comp_dm3 = load_dm(p.prms3)
    comp_dm4 = load_dm(p.prms4)
    comp_dm5 = load_dm(p.prms5)
    comp_dmt = load_dm(p.prmst)

    comp_gas1 = load_gas(p.prms1)
    comp_gas2 = load_gas(p.prms2)
    comp_gas3 = load_gas(p.prms3)
    comp_gas4 = load_gas(p.prms4)
    comp_gas5 = load_gas(p.prms5)
    comp_gast = load_gas(p.prmst)

    comp_stars1, comp_cold1 = load_stars(p.prms1)
    comp_stars2, comp_cold2 = load_stars(p.prms2)
    comp_stars3, comp_cold3 = load_stars(p.prms3)
    comp_stars4, comp_cold4 = load_stars(p.prms4)
    comp_stars5, comp_cold5 = load_stars(p.prms5)
    comp_starst, comp_coldt = load_stars(p.prmst)

    comp_gas1 += comp_cold1
    comp_gas2 += comp_cold2
    comp_gas3 += comp_cold3
    comp_gas4 += comp_cold4
    comp_gas5 += comp_cold5
    comp_gast += comp_coldt

    powt = power.Power([comp_dmt, comp_gast, comp_starst], comp_dmt.p_lin, name='10-15')

    comps_dm = [comp_dm1, comp_dm2, comp_dm3, comp_dm4, comp_dm5]
    comps_gas = [comp_gas1, comp_gas2, comp_gas3, comp_gas4, comp_gas5]
    comps_stars = [comp_stars1, comp_stars2, comp_stars3, comp_stars4, comp_stars5]

    return comps_dm, comps_gas, comps_stars, powt

# ------------------------------------------------------------------------------
# End of load_masses()
# ------------------------------------------------------------------------------

def plot_power_ratio_paper(comp, name='gas_smooth_r200m_5r500c'):
    '''
    Compare the power in component to the simulation power spectrum
    '''
    k1, P1, delta1 = np.loadtxt('halo/data/data_vandaalen/BAHAMAS/tables/AGN_TUNED_nu0_L400N1024_WMAP9_032_gas_T_gt_1e6K_table.dat', unpack=True)

    f_D = interp.interp1d(k1, delta1)

    pl.set_style('line')
    plt.clf()
    fig = plt.figure(figsize=(11, 8))
    # ax_P = fig.add_axes([0.1, 0.35, 0.8, 0.55])
    # ax_r = fig.add_axes([0.1, 0.1, 0.8, 0.25])
    ax_P = fig.add_subplot(111)

    ax_P.plot(k1, delta1, label=r'gas_T_gt_1e6K')
    ax_P.plot(comp.k_range, comp.delta_1h, label=name)

    axd = ax_P.twiny()
    l = 2 * np.pi / k1
    axd.plot(l, delta1)
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

    # ax_r.plot(comp.k_range, comp.delta_1h / f_D(comp.k_range))
    # ax_r.axhline(y=1, c='k', ls='--')
    # ax_r.grid()
    # ax_r.set_xlim([1e-2,1e2])
    # # ax_r.set_ylim([1e-3,1])
    # ax_r.set_ylim([0.82,1.2])
    # ax_r.axes.set_xscale('log')
    # # ax_r.axes.set_yscale('log')
    # ax_r.minorticks_on()
    # ax_r.tick_params(axis='x',which='minor',bottom='off')

    # ax_r.legend(loc='best')
    # ax_r.set_xlabel(r'$k \, [h/\mathrm{Mpc}]$')
    # # ax_r.set_ylabel(r'$\frac{P_{\mathrm{AGN}} - P_{\mathrm{DM}}}{P_{\mathrm{DM}}}$',
    # #                 labelpad=-2)
    # ax_r.set_ylabel(r'$P_\mathrm{obs}/P_\mathrm{sim}$')

    plt.savefig('ratio_sim_obs.pdf', dpi=900, transparent=True)
    # plt.close(fig)

# ------------------------------------------------------------------------------
# End of plot_power_ratio_paper()
# ------------------------------------------------------------------------------
