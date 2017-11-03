import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import sys

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

import pdb

# load default settings for all parameters
prms = p.prms

# general profile kwargs to be used for all components
profile_kwargs = {'r_range': prms.r_range_lin,
                  'm_range': prms.m_range_lin,
                  'k_range': prms.k_range_lin,
                  'n': 80,
                  'taylor_err': 1.e-50}

# ------------------------------------------------------------------------------
def load_dm():
    # Fit NFW profile with c(z,M) relation from Correa et al (2015c)
    # f_dm = 1. * np.ones_like(prms.m_range_lin)
    f_dm = 1 - prms.omegab/prms.omegam * np.ones_like(prms.m_range_lin)

    # concentration and virial radius -> needed for profile_NFW
    # c_correa want [M_sun], not [M_sun/h]
    ############################################################################
    # c_x = profs.c_correa(f_dm * prms.m_range_lin, z_range=0.)
    c_x = profs.c_correa(prms.m_range_lin / prms.h, z_range=0.)
    ############################################################################
    r_x = prms.r_h

    # DM profile
    # specific dm extra kwargs
    dm_extra = {'profile': profs.profile_NFW,
                'profile_f': profs.profile_NFW_f,
                'profile_args': {'c_x': c_x,
                                 'r_x': r_x,
                                 'rho_mean': prms.rho_m},
                'profile_f_args': {'c_x': c_x,
                                   'r_x': r_x,
                                   'rho_mean': prms.rho_m}}
    prof_dm_kwargs = tools.merge_dicts(profile_kwargs, dm_extra)

    # additional kwargs for comp.Component
    comp_dm_kwargs = {'name': 'dm',
                      # 'm_fn': prms.m_fn,
                      'p_lin': prms.p_lin,
                      'nu': prms.nu,
                      'fnu': prms.fnu,
                      'f_comp': f_dm,}
                      # 'bias_fn': bias.bias_Tinker10,
                      # 'bias_fn_args': {'m_fn': prms.m_fn}}

    dm_kwargs = tools.merge_dicts(prof_dm_kwargs, comp_dm_kwargs)

    comp_dm = comp.Component(**dm_kwargs)
    return comp_dm

# ------------------------------------------------------------------------------
# End of load_dm()
# ------------------------------------------------------------------------------

def load_sat():
    f_sat = stars.f_s(prms.m_range_lin)
    # Satellite profile
    # generalized NFW profile determined by van der Burg, Hoekstra et al. (2015)
    prof_sat = profs.profile_gNFW(prms.r_range_lin, c_x=0.64*np.ones(prms.m_bins),
                                  alpha=1.63, r_x=prms.r_h,
                                  m_s=prms.m_range_lin)

    sat_extra = {'profile': prof_sat,
                 'profile_f': None}
    prof_sat_kwargs = tools.merge_dicts(profile_kwargs, sat_extra)

    comp_sat_kwargs = {'name': 'sat',
                       # 'm_fn': prms.m_fn,
                       'p_lin': prms.p_lin,
                       'nu': prms.nu,
                       'fnu': prms.fnu,
                       'f_comp': f_sat,}
                       # 'bias_fn': bias.bias_Tinker10,
                       # 'bias_fn_args': {'m_fn': prms.m_fn}}

    sat_kwargs = tools.merge_dicts(prof_sat_kwargs, comp_sat_kwargs)
    comp_sat = comp.Component(**sat_kwargs)
    return comp_sat

# ------------------------------------------------------------------------------
# End of load_sat()
# ------------------------------------------------------------------------------

def load_bcg():
    f_cen = stars.f_c(prms.m_range_lin)

    # # BCG profile
    # # Kravtsov (2014) -> r1/2 = 0.015 r200
    # prof_bcg = profs.profile_BCG(prms.r_range_lin, prms.m_range_lin,#st.m_cen_fit,
    #                              r_half=0.015*prms.r_range_lin[:,-1])

    # quite arbitrary...
    n1_idx = (prms.m_range_lin * f_cen < 1e11)
    n4_idx = (prms.m_range_lin * f_cen >= 1e11)

    # Fit sersic profile with n=1 for M<1e11 and n=4 for M>=1e11
    # Kravtsov (2014) -> r1/2 = 0.015 r200
    prof_bcg_n1 = profs.profile_sersic(prms.r_range_lin[n1_idx],
                                       prms.m_range_lin[n1_idx],
                                       r_eff=0.015*prms.r_range_lin[n1_idx,-1],
                                       p=1)
    prof_bcg_n4 = profs.profile_sersic(prms.r_range_lin[n4_idx],
                                       prms.m_range_lin[n4_idx],
                                       r_eff=0.015*prms.r_range_lin[n4_idx,-1],
                                       p=4)
    prof_bcg = np.concatenate([prof_bcg_n1, prof_bcg_n4], axis=0)

    # prof_bcg = profs.profile_delta(prms.r_range_lin, prms.m_range_lin)
    # prof_bcg_f = profs.profile_delta_f(prms.k_range_lin, prms.m_range_lin)
    bcg_extra = {'profile': prof_bcg,
                 'profile_f' : None}
                 # 'profile_f': prof_bcg_f}
    prof_bcg_kwargs = tools.merge_dicts(profile_kwargs, bcg_extra)

    print '! Check M_h-M* relation !'

    comp_bcg_kwargs = {'name': 'bcg',
                       # 'm_fn': prms.m_fn,
                       'p_lin': prms.p_lin,
                       'nu': prms.nu,
                       'fnu': prms.fnu,
                       'f_comp': f_cen,}
                       # 'bias_fn': bias.bias_Tinker10,
                       # 'bias_fn_args': {'m_fn': prms.m_fn}}

    bcg_kwargs = tools.merge_dicts(prof_bcg_kwargs, comp_bcg_kwargs)
    comp_bcg = comp.Component(**bcg_kwargs)
    return comp_bcg

# ------------------------------------------------------------------------------
# End of load_bcg()
# ------------------------------------------------------------------------------

def load_icl():
    f_cen = stars.f_c(prms.m_range_lin)
    # ICL profile
    prof_icl = profs.profile_ICL(prms.r_range_lin, prms.m_range_lin,
                                 r_half=0.015*prms.r_range_lin[:,-1], n=5)

    icl_extra = {'profile': prof_icl,
                 'profile_f': None}
    prof_icl_kwargs = tools.merge_dicts(profile_kwargs, icl_extra)

    comp_icl_kwargs = {'name': 'icl',
                       # 'm_fn': prms.m_fn,
                       'p_lin': prms.p_lin,
                       'nu': prms.nu,
                       'fnu': prms.fnu,
                       'f_comp': f_cen,}
                       # 'bias_fn': bias.bias_Tinker10,
                       # 'bias_fn_args': {'m_fn': prms.m_fn}}

    icl_kwargs = tools.merge_dicts(prof_icl_kwargs, comp_icl_kwargs)
    comp_icl = comp.Component(**icl_kwargs)
    return comp_icl

# ------------------------------------------------------------------------------
# End of load_icl()
# ------------------------------------------------------------------------------

def load_gas():
    c_x = profs.c_correa(prms.m_range_lin / 0.7, z_range=0).reshape(-1)
    m500c = np.array([gas.m200m_to_m500c(m / 0.7) for m in prms.m_range_lin])
    r500c = tools.mass_to_radius(m500c, 500 * prms.rho_crit * 0.7**2)

    r_range = prms.r_range_lin / (0.7 * r500c.reshape(-1,1))

    idx_500 = np.argmin(np.abs(r_range - 1.), axis=-1)
    r_x = np.array([r_range[i, idx] for i, idx in enumerate(idx_500)])

    f_gas500 = gas.f_gas(m500c, **gas.f_gas_fit())

    def beta_fit(m_range, m_c, alpha):
        return 1 + 2. / (1 + (m_c/m_range)**alpha)

    def rc_fit(m_range, m_c, alpha):
        return 0.4 / (1 + (m_c/m_range)**alpha)

    # Values for profile
    beta_m = 305484041479805.88
    beta_a = 1.0817928922531508
    beta = beta_fit(m500c, beta_m, beta_a)

    r_c_m = 556798173789358.31
    r_c_a = 0.64336173115208184
    r_c = rc_fit(m500c, r_c_m, r_c_a)

    # # Values for q16
    # beta_m = 305484041479805.88
    # beta_a = 1.0817928922531508
    # beta = beta_fit(m500c, beta_m, beta_a)

    # r_c_m = 556798173789358.31
    # r_c_a = 0.64336173115208184
    # r_c = rc_fit(m500c, r_c_m, r_c_a)

    # # Values for q84
    # beta_m = 305484041479805.88
    # beta_a = 1.0817928922531508
    # beta = beta_fit(m500c, beta_m, beta_a)

    # r_c_m = 556798173789358.31
    # r_c_a = 0.64336173115208184
    # r_c = rc_fit(m500c, r_c_m, r_c_a)

    # print beta
    # print r_c

    # correct for mass integration with x=r/r500c -> profile integrates to m500_gas
    prof_gas = profs.profile_beta(r_range,
                                  f_gas500 * m500c/r500c**3,
                                  r_x=r_x,
                                  beta=beta, r_c=r_c)

    # prof_gas = profs.profile_beta_extra(r_range, prof_gas, r_x=r_x, a=3)
    m200_prof = tools.m_h(prof_gas, prms.r_range_lin, axis=-1)
    m500_prof = np.array([tools.m_h(prof_gas[idx, :i_500+1],
                                    prms.r_range_lin[idx, :i_500+1] / 0.7)
                          for idx, i_500 in enumerate(idx_500)])
    # correct gas mass!
    norm = prms.m_range_lin / m200_prof

    # profile has to integrate to m200m
    prof_gas *= norm.reshape(-1,1)

    m500_new = np.array([tools.m_h(prof_gas[idx, :i_500+1],
                                   prms.r_range_lin[idx, :i_500+1] / 0.7)
                         for idx, i_500 in enumerate(idx_500)])

    # but f_gas needs to make sure we still match observations at r500c
    f_gas = m500_prof / (m500_new)

    # plt.clf()
    # pl.set_style('line')
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # l1, = ax1.plot(m500c, f_gas500, ls='-')
    # ax2 = ax1.twiny()
    # l2, = ax2.plot(prms.m_range_lin, f_gas, ls='--')
    # ax1.set_xlim([m500c.min(), m500c.max()])
    # ax2.set_xlim([prms.m_range_lin.min(), prms.m_range_lin.max()])
    # ax1.set_xscale('log')
    # ax2.set_xscale('log')
    # ax1.set_xlabel(r'$m_{500c} \, [M_\odot]$')
    # ax2.set_xlabel(r'$m_{200m} \, [M_\odot]$', labelpad=5)
    # ax1.set_ylabel(r'$f_{\mathrm{gas}}$')
    # ax1.legend([l1, l2], [r'$f_{500c}$', r'$f_{200m}$'], loc=2)
    # plt.show()


    gas_extra = {'profile': prof_gas,
                 'profile_f': None}
    prof_gas_kwargs = tools.merge_dicts(profile_kwargs, gas_extra)

    comp_gas_kwargs = {'name': 'gas',
                       # 'm_fn': prms.m_fn,
                       'p_lin': prms.p_lin,
                       'nu': prms.nu,
                       'fnu': prms.fnu,
                       'f_comp': f_gas,}
                       # 'bias_fn': bias.bias_Tinker10,
                       # 'bias_fn_args': {'m_fn': prms.m_fn}}

    gas_kwargs = tools.merge_dicts(prof_gas_kwargs, comp_gas_kwargs)
    comp_gas = comp.Component(**gas_kwargs)

    return comp_gas

# ------------------------------------------------------------------------------
# End of load_gas()
# ------------------------------------------------------------------------------

# power_st = power.Power([comp_sat, comp_bcg])
# power_m = power.Power([comp_dm, comp_sat, comp_bcg])#, comp_gas])
# power_m_g = power.Power([comp_dm, comp_sat, comp_bcg, comp_gas])
# power_m_icl = power.Power([comp_dm, comp_sat, comp_icl, comp_gas])
