import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
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

def prof_nfw(r, rs, m):
    x = r/rs.reshape(-1,1)
    # dc = dc.reshape(-1,1)
    m = m.reshape(-1,1)
    profile = x**(-1) * (1 + x)**(-2) #* p.prms.h**2
    mass = tools.m_h(profile, r).reshape(-1,1)
    profile *= m/mass

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
    return (m/mc)**t/(1 + (m/mc)**t)

def load_dm(prms):
    # general profile kwargs to be used for all components
    profile_kwargs = {'r_range': prms.r_range_lin,
                      'm_range': prms.m_range_lin,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}
    # extract all fit parameters
    rs, rs_err, dc, dc_err, masses, m200, c_prms, r_prms, d_prms, f_prms = b.fit_dm_bahamas()

    # DMO test
    f_dm = np.ones_like(prms.m_range_lin)
    # c_x = profs.c_correa(prms.m_range_lin, 0).reshape(-1)
    # rs = prms.r_h / c_x
    r_x = prms.r_h

    # Test BAHAMAS fit
    rs = b.dm_rs_fit(prms.m_range_lin / prms.h, **r_prms) * prms.h
    c_x = r_x / rs
    # dc = b.dm_dc_fit(prms.m_range_lin / prms.h, **d_prms) / prms.h**2
    # f_dm = b.dm_f_fit(prms.m_range_lin / prms.h, **f_prms)

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

    # bahamas self consistent
    # prof_dm = prof_nfw(prms.r_range_lin, rs, prms.m_range_lin)
    # prof_dm_kwargs = tools.merge_dicts(profile_kwargs, {'profile': prof_dm})

    # additional kwargs for comp.Component
    comp_dm_kwargs = {'name': 'dm',
                      'm_fn': prms.m_fn,
                      'f_comp': f_dm,
                      'bias_fn': bias.bias_Tinker10,
                      'bias_fn_args': {'m_fn': prms.m_fn},
                      'k_c': 0.1}

    dm_kwargs = tools.merge_dicts(prof_dm_kwargs, comp_dm_kwargs)

    comp_dm = comp.Component(**dm_kwargs)

    return comp_dm

def load_gas(prms):
    # general profile kwargs to be used for all components
    profile_kwargs = {'r_range': prms.r_range_lin,
                      'm_range': prms.m_range_lin,
                      'k_range': prms.k_range_lin,
                      'n': 80,
                      'taylor_err': 1.e-50}
    rs_w, s_w, rsw_err, sw_err, r0_w, m_w, m200_w = b.fit_gas_warm_bahamas()
    m200_h, mc_h, rc_h, bc_h, rc_err, bc_err, ms_h, rs_h, s_h, rs_err, s_err, r0_h = b.fit_gas_hot_bahamas()
    # extract all fit parameters
    rw_prms, sw_prms, rc_prms, b_prms, rs_prms, ss_prms, mw_prms, mc_prms, ms_prms, r0w_prms, r0s_prms = b.plot_gas_fit_bahamas_median()


    f_w = b.gas_warm_mw_fit(prms.m_range_lin / prms.h, **mw_prms)
    fc_h = b.gas_hot_mc_fit(prms.m_range_lin / prms.h, **mc_prms)
    fs_h = b.gas_hot_ms_fit(prms.m_range_lin / prms.h, **ms_prms)

    # # reshape profile parameters
    # r_x = (prms.r_h / prms.h)
    # m = (prms.m_range_lin / prms.h)
    r_x = prms.r_h
    m = prms.m_range_lin

    ############################################################################
    # Need to slice with r0
    # Need to remove masses for which we do not have profile
    rw = b.gas_warm_rw_fit(m, **rw_prms) * prms.h
    sw = b.gas_warm_sigma_fit(m, **sw_prms)
    r0w = b.gas_warm_r0_fit(m, **r0w_prms) * prms.h
    prof_w = np.zeros_like(prms.r_range_lin)
    for idx, mass in enumerate(m):
        if mass <= 1e13 * prms.h:
            r_sl = (prms.r_range_lin[idx] >= r0w[idx] * r_x[idx])
            if r_sl.sum() > 0:
                prof_w[idx, r_sl] = prof_gas_warm((prms.r_range_lin[idx, r_sl]/
                                                   r_x[idx]), rw[idx], sw[idx],
                                                  m[idx],
                                                  r_x[idx])

    w_kwargs = tools.merge_dicts(profile_kwargs, {'profile': prof_w})
    comp_w_kwargs = {'name': 'warm',
                     'm_fn': prms.m_fn,
                     'f_comp': f_w,
                     'bias_fn': bias.bias_Tinker10,
                     'bias_fn_args': {'m_fn': prms.m_fn},
                      'k_c': 0.1}
    w_kwargs = tools.merge_dicts(w_kwargs, comp_w_kwargs)
    comp_w = comp.Component(**w_kwargs)


    rc = b.gas_hot_rc_fit(m, **rc_prms) * prms.h
    bc = b.gas_hot_beta_fit(m, **b_prms)
    rs = b.gas_hot_rs_fit(m, **rs_prms) * prms.h
    # prof_c = prof_gas_hot_c((prms.r_range_lin/r_x) / prms.h, rc, bc, rs, m, r_x)
    prof_c = np.zeros_like(prms.r_range_lin)
    for idx, mass in enumerate(m):
        if (mass >= 1e13 * prms.h):
            prof_c[idx] = prof_gas_hot_c((prms.r_range_lin[idx]/r_x[idx]),
                                         rc[idx], bc[idx], rs[idx],
                                         m[idx], r_x[idx])


    c_kwargs = tools.merge_dicts(profile_kwargs, {'profile': prof_c})
    comp_c_kwargs = {'name': 'cen',
                     'm_fn': prms.m_fn,
                     'f_comp': fc_h,
                     'bias_fn': bias.bias_Tinker10,
                     'bias_fn_args': {'m_fn': prms.m_fn},
                     'k_c': 0.1}
    c_kwargs = tools.merge_dicts(c_kwargs, comp_c_kwargs)
    comp_c = comp.Component(**c_kwargs)


    ss = b.gas_hot_sigma_fit(m, **ss_prms)
    r0s = b.gas_hot_r0_fit(m, **r0s_prms) * prms.h
    # prof_s = prof_gas_hot_s((prms.r_range_lin/r_x) / prms.h, rs, ss, m, r_x)
    prof_s = np.zeros_like(prms.r_range_lin)
    for idx, mass in enumerate(m):
        if mass >= 1e13 * prms.h:
            r_sl = (prms.r_range_lin[idx] >= r0s[idx] * r_x[idx])
            if r_sl.sum() > 0:
                prof_s[idx, r_sl] = prof_gas_hot_s((prms.r_range_lin[idx, r_sl]/
                                                    r_x[idx]), rs[idx], ss[idx],
                                                   m[idx], r_x[idx])


    s_kwargs = tools.merge_dicts(profile_kwargs, {'profile': prof_s})
    comp_s_kwargs = {'name': 'sat',
                     'm_fn': prms.m_fn,
                     'f_comp': fs_h,
                     'bias_fn': bias.bias_Tinker10,
                     'bias_fn_args': {'m_fn': prms.m_fn},
                     'k_c': 0.1}
    s_kwargs = tools.merge_dicts(s_kwargs, comp_s_kwargs)
    comp_s = comp.Component(**s_kwargs)

    return comp_w, comp_c, comp_s
