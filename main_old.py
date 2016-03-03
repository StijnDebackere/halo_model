import numpy as np

import tools
import halo.density_profiles as profs
import halo.model.density as dens
import halo.model.component as comp
import halo.model.power
import halo.parameters as p
import halo.stars
import halo.bias

import pdb

ddir = '/Volumes/Data/stijn/Documents/Universiteit/MR/code/data/'
prms = p.prms

# general profile kwargs to be used for all components
profile_kwargs = {'r_range': prms.r_range_lin,
                  'm_range': prms.m_range_lin,
                  'k_range': prms.k_range_lin,
                  'n': 80,
                  'taylor_err': 1.e-50}

# ------------------------------------------------------------------------------
# Dark matter
# concentration and virial radius
c_x = profs.c_correa(prms.m_range_lin, z_range=0.)
r_x = prms.r_h

# # ------------------------------------------------------------------------------
# # Stars
# # load in relations from Zu & Mandelbaum
# m_h_in, f, f_c, f_s = np.loadtxt(ddir + 'data_mccarthy/StellarFraction-Mh.txt',
#                                  unpack=True)
# st = stars.Stars(m_h=prms.m_range_lin, m_h_in=m_h_in, f_c_in=f_c, f_s_in=f_s)

# ------------------------------------------------------------------------------
# Gas
# data for median halo mass M500 ~ 8.7e13 Msun
# r500, rho/rho_crit(med, 16th, 84th percentiles)
r500, rho_m, rho_16, rho_84 = np.loadtxt(ddir + 'data_mccarthy/gas/Sun09_rhogas_profiles.dat',
                                         unpack=True)

M = 8.7e13
# Look for closest halo mass in our range, need to convert to M500 first.
M500 = prms.m_range_lin * tools.Mx_to_My(1., 500, 200,
                                         c_x.reshape(prms.m_range_lin.shape[0],))
idx_M = np.argmin(np.abs(M500 - M))
# r converted to our units
r = r500 * tools.rx_to_r200(1, 500) * prms.r_range_lin[idx_M, -1]
rho = rho_m * prms.rho_crit * 0.7**2
fit_prms, fit = profs.fit_profile_beta(r,
                                       np.array([prms.m_range_lin[idx_M]]),
                                       prms.r_range_lin[idx_M,-1],
                                       rho)

# ------------------------------------------------------------------------------
# fractions
# f_dm = 1 - prms.omegab/prms.omegam * np.ones_like(prms.m_range_lin)
# f_gas =  1 - st.f_c - st.f_s - f_dm
    

# ------------------------------------------------------------------------------
def load_dm():
    # DM profile
    # specific dm extra kwargs
    dm_extra = {#'m_range': prms.m_range_lin * f_dm,
        'profile': profs.profile_NFW,
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
    # Satellite profile
    # generalized NFW profile determined by van der Burg, Hoekstra et al. (2015)
    prof_sat = profs.profile_gNFW(prms.r_range_lin, c_x=0.64*np.ones(prms.m_bins),
                                  alpha=1.63, r_x=prms.r_h,
                                  m_s=prms.m_range_lin)

    sat_extra = {#'m_range': st.f_s/st.f_c * st.m_cen_fit,
        'profile': prof_sat,
        'profile_f': None}
    prof_sat_kwargs = tools.merge_dicts(profile_kwargs, sat_extra)

    comp_sat_kwargs = {'name': 'sat',
                       'm_fn': prms.m_fn,
                       'f_comp': st.f_s,
                       'bias_fn': bias.bias_Tinker10,
                       'bias_fn_args': {'m_fn': prms.m_fn}}

    sat_kwargs = tools.merge_dicts(prof_sat_kwargs, comp_sat_kwargs)
    comp_sat = comp.Component(**sat_kwargs)
    return comp_sat

# ------------------------------------------------------------------------------
# End of load_sat()
# ------------------------------------------------------------------------------
def load_bcg():
    # BCG profile
    # Kravtsov (2014) -> r1/2 = 0.015 r200
    # prof_bcg = profs.profile_BCG(prms.r_range_lin, prms.m_range_lin,#st.m_cen_fit,
    #                              r_half=0.015*prms.r_range_lin[:,-1])
    # quite arbitrary...
    n1_idx = (prms.m_range_lin * st.f_c < 1e11)
    n4_idx = (prms.m_range_lin * st.f_c >= 1e11)

    prof_bcg_n1 = profs.profile_sersic(prms.r_range_lin[n1_idx],
                                       prms.m_range_lin[n1_idx],
                                       r_eff=0.015*prms.r_range_lin[n1_idx,-1],
                                       p=1)
    prof_bcg_n4 = profs.profile_sersic(prms.r_range_lin[n4_idx],
                                       prms.m_range_lin[n4_idx],
                                       r_eff=0.015*prms.r_range_lin[n4_idx,-1],
                                       p=4)
    prof_bcg = np.concatenate([prof_bcg_n1, prof_bcg_n4], axis=0)

    bcg_extra = {#'m_range': st.m_cen_fit,
        'profile': prof_bcg,
        'profile_f': None}
    prof_bcg_kwargs = tools.merge_dicts(profile_kwargs, bcg_extra)

    print '! Check M_h-M* relation !'

    comp_bcg_kwargs = {'name': 'bcg',
                       'm_fn': prms.m_fn,
                       'f_comp': st.f_c,
                       'bias_fn': bias.bias_Tinker10,
                       'bias_fn_args': {'m_fn': prms.m_fn}}

    bcg_kwargs = tools.merge_dicts(prof_bcg_kwargs, comp_bcg_kwargs)
    comp_bcg = comp.Component(**bcg_kwargs)
    return comp_bcg

# ------------------------------------------------------------------------------
# End of load_bcg()
# ------------------------------------------------------------------------------
def load_icl():
    # ICL profile
    prof_icl = profs.profile_ICL(prms.r_range_lin, prms.m_range_lin,#st.m_cen_fit,
                                 r_half=0.015*prms.r_range_lin[:,-1], n=5)
    
    icl_extra = {#'m_range': st.m_cen_fit,
                 'profile': prof_icl,
                 'profile_f': None}
    prof_icl_kwargs = tools.merge_dicts(profile_kwargs, icl_extra)
        
    comp_icl_kwargs = {'name': 'icl',
                       'm_fn': prms.m_fn,
                       'f_comp': st.f_c,
                       'bias_fn': bias.bias_Tinker10,
                       'bias_fn_args': {'m_fn': prms.m_fn}}
    
    icl_kwargs = tools.merge_dicts(prof_icl_kwargs, comp_icl_kwargs)
    comp_icl = comp.Component(**icl_kwargs)
    return comp_icl
    
# ------------------------------------------------------------------------------
# End of load_icl()
# ------------------------------------------------------------------------------
def load_gas():
    global rho
    # Gas profile    
    fit_prms, fit = profs.fit_profile_beta(r,
                                           np.array([prms.m_range_lin[idx_M]]),
                                           prms.r_range_lin[idx_M,-1],
                                           rho)
    print fit_prms
    prof_gas = profs.profile_beta(prms.r_range_lin, prms.m_range_lin,#st.m_cen_fit,
                                  r_x=prms.r_range_lin[:,-1],
                                  beta=fit_prms[0,0],r_c=fit_prms[0,1])
    gas_extra = {#'m_range': st.m_cen_fit,
                 'profile': prof_gas,
                 'profile_f': None}
    prof_gas_kwargs = tools.merge_dicts(profile_kwargs, gas_extra)
        
    # TODO:
    # - Implement correct gas mass fractions
    # - Look at FT of gas profile
    #   -> for smallest mass, we get almost flat profile -> sinc as FT
    #      mass dependence in core radius should be added?
    #   -> extrapolate highest mass gas profile, also gives contributions at
    #      largest scales, but not negative there, compensates small M
    # - DONE: Conversion to r200 -> still need to multiply with r200 to
    #   get correct units
    
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
