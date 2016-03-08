import numpy as np

import halo.tools as tools
import halo.density_profiles as profs
import halo.stars as stars
import halo.gas as gas
import halo.bias as bias
import halo.parameters as p
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
    # concentration and virial radius -> needed for profile_NFW
    c_x = profs.c_correa(prms.m_range_lin, z_range=0.)
    r_x = prms.r_h

    # Fit NFW profile with c(z,M) relation from Correa et al (2015c)
    f_dm = 1 - prms.omegab/prms.omegam * np.ones_like(prms.m_range_lin)
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
                      'm_fn': prms.m_fn,
                      'f_comp': f_dm,
                      'bias_fn': bias.bias_Tinker10,
                      'bias_fn_args': {'m_fn': prms.m_fn}}

    dm_kwargs = tools.merge_dicts(prof_dm_kwargs, comp_dm_kwargs)

    comp_dm = comp.Component(**dm_kwargs)
    return comp_dm

# ------------------------------------------------------------------------------
# End of load_dm()
# ------------------------------------------------------------------------------

def load_sat():
    f_sat = stars.f_s(p.prms.m_range_lin)
    # Satellite profile
    # generalized NFW profile determined by van der Burg, Hoekstra et al. (2015)
    prof_sat = profs.profile_gNFW(prms.r_range_lin, c_x=0.64*np.ones(prms.m_bins),
                                  alpha=1.63, r_x=prms.r_h,
                                  m_s=prms.m_range_lin)

    sat_extra = {'profile': prof_sat,
                 'profile_f': None}
    prof_sat_kwargs = tools.merge_dicts(profile_kwargs, sat_extra)

    comp_sat_kwargs = {'name': 'sat',
                       'm_fn': prms.m_fn,
                       'f_comp': f_sat,
                       'bias_fn': bias.bias_Tinker10,
                       'bias_fn_args': {'m_fn': prms.m_fn}}

    sat_kwargs = tools.merge_dicts(prof_sat_kwargs, comp_sat_kwargs)
    comp_sat = comp.Component(**sat_kwargs)
    return comp_sat

# ------------------------------------------------------------------------------
# End of load_sat()
# ------------------------------------------------------------------------------

def load_bcg():
    f_cen = stars.f_c(p.prms.m_range_lin)

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

    # prof_bcg = profs.profile_delta(p.prms.r_range_lin, p.prms.m_range_lin)
    # prof_bcg_f = profs.profile_delta_f(p.prms.k_range_lin, p.prms.m_range_lin)
    bcg_extra = {'profile': prof_bcg,
                 'profile_f' : None}
                 # 'profile_f': prof_bcg_f}
    prof_bcg_kwargs = tools.merge_dicts(profile_kwargs, bcg_extra)

    print '! Check M_h-M* relation !'

    comp_bcg_kwargs = {'name': 'bcg',
                       'm_fn': prms.m_fn,
                       'f_comp': f_cen,
                       'bias_fn': bias.bias_Tinker10,
                       'bias_fn_args': {'m_fn': prms.m_fn}}

    bcg_kwargs = tools.merge_dicts(prof_bcg_kwargs, comp_bcg_kwargs)
    comp_bcg = comp.Component(**bcg_kwargs)
    return comp_bcg

# ------------------------------------------------------------------------------
# End of load_bcg()
# ------------------------------------------------------------------------------

def load_icl():
    f_cen = stars.f_c(p.prms.m_range_lin)
    # ICL profile
    prof_icl = profs.profile_ICL(prms.r_range_lin, prms.m_range_lin,
                                 r_half=0.015*prms.r_range_lin[:,-1], n=5)

    icl_extra = {'profile': prof_icl,
                 'profile_f': None}
    prof_icl_kwargs = tools.merge_dicts(profile_kwargs, icl_extra)

    comp_icl_kwargs = {'name': 'icl',
                       'm_fn': prms.m_fn,
                       'f_comp': f_cen,
                       'bias_fn': bias.bias_Tinker10,
                       'bias_fn_args': {'m_fn': prms.m_fn}}

    icl_kwargs = tools.merge_dicts(prof_icl_kwargs, comp_icl_kwargs)
    comp_icl = comp.Component(**icl_kwargs)
    return comp_icl

# ------------------------------------------------------------------------------
# End of load_icl()
# ------------------------------------------------------------------------------

def load_gas():
    fit_prms, fit, m_idx = gas.f_gas_fit()

    c_x = profs.c_correa(p.prms.m_range_lin, z_range=0)
    M500 = tools.Mx_to_My(p.prms.m_range_lin, 500, 200, c_x)
    f_gas = gas.f_gas(M500, **fit_prms)

    prof_gas = profs.profile_beta_plaw(prms.r_range_lin,
                                       prms.m_range_lin,
                                       r_x=prms.r_range_lin[:,-1],
                                       beta=fit_prms[0,0],r_c=fit_prms[0,1])
    gas_extra = {'profile': prof_gas,
                 'profile_f': None}
    prof_gas_kwargs = tools.merge_dicts(profile_kwargs, gas_extra)

    # TODO:
    # - Look at FT of gas profile
    #   -> for smallest mass, we get almost flat profile -> sinc as FT
    #      mass dependence in core radius should be added?
    #   -> extrapolate highest mass gas profile, also gives contributions at
    #      largest scales, but not negative there, compensates small M

    comp_gas_kwargs = {'name': 'gas',
                       'm_fn': prms.m_fn,
                       'f_comp': f_gas,
                       'bias_fn': bias.bias_Tinker10,
                       'bias_fn_args': {'m_fn': prms.m_fn}}

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
