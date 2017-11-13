import numpy as np
import pandas as pd
import scipy.interpolate as intp

import new_halo.density_profiles as profs
import new_halo.density as dens
import new_halo.bias as bias
import new_halo.tools as tools
import new_halo.component as comp
import new_halo.main as m

# data goes 2:21
d = pd.read_csv('schaller_data/FitParametersStackRelaxed_E_100_z0p0.dat')

# convert from kpc to Mpc h^-1
r_nfw = d[2:21]['newProf_rs']/(m.prms.H0/100. * 1000.) 
d_nfw = d[2:21]['newProf_dc']
r_star = d[2:21]['newProf_rstar']/(m.prms.H0/100. * 1000.) 
d_star = d[2:21]['newProf_dstar']
M200 = d[2:21]['M_200']/(m.prms.H0/100.)
r_200 = d[2:21]['R_200']/(m.prms.H0/100. * 1000.)

# interpolate to our mass range
f_ds = intp.interp1d(M200, d_nfw)
f_rs = intp.interp1d(M200, r_nfw)
f_di = intp.interp1d(M200, d_star)
f_ri = intp.interp1d(M200, r_star)
f_rh = intp.interp1d(M200, r_200)

prms = m.prms
prms.m_min = 11.0
prms.m_max = 14.0

d_s = f_ds(prms.m200m)
r_s = f_rs(prms.m200m)
d_i = f_di(prms.m200m)
r_i = f_ri(prms.m200m)
r_h = f_rh(prms.m200m)

r_range_lin = np.array([np.linspace(0.005 * rh, rh, prms.r_bins) for rh in r_h])
m_range_lin = tools.radius_to_mass(r_h, prms.rho_crit)

m.prms.k_min = -2.7
m.prms.k_max = np.log(20)

rho_r = profs.profile_Schaller(r_range=r_range_lin,
                               r_s=r_s,
                               d_s=d_s,
                               r_i=r_i,
                               d_i=d_i,
                               rho_crit=prms.rho_crit)

profile_kwargs = {'r_range': r_range_lin,
                  'm_range': m_range_lin,
                  'k_range': prms.k_range_lin,
                  'profile': rho_r,
                  'profile_f': None,
                  'n': 84,
                  'taylor_err': 1.e-50}

prof_schaller = comp.Component(m_fn=prms.m_fn, f_comp=1.,
                               bias_fn=bias.bias_Tinker10,
                               bias_fn_args={'m_fn': prms.m_fn},
                               **profile_kwargs)



# rho_k = profs.profile_Schaller_f(k_range=prms.k_range_lin,
#                                  r_range=prms.r_range_lin,
#                                  m_range=prms.m200m,
#                                  rho_mean=prms.rho_m,
#                                  omegam=prms.omegam,
#                                  r_i=r_i,
#                                  m_i=m_i,
#                                  d_i=d_i)

# profile_kwargs = {
#     'r_range': prms.r_range_lin,
#     'm_range': prms.m200m,
#     'k_range': prms.k_range_lin,
#     'profile': rho_r,
#     'profile_f': rho_k
# }

# comp_schaller = comp.Component(m_fn=prms.m_fn,
#                                f_comp=1.,
#                                bias_fn=bias.bias_Tinker10,
#                                bias_fn_args={'m_fn': prms.m_fn},
#                                **profile_kwargs)

