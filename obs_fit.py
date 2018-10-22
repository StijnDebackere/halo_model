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

import halo.hmf as hmf
import halo.parameters as p
import halo.tools as tools
import halo.density_profiles as dp
import halo.tools as tools
import halo.model.density as dens
import halo.model.power as power
import halo.data.data as d

import pdb

# ------------------------------------------------------------------------------
# Definition of different matter components
# ------------------------------------------------------------------------------

def load_dm_dmo_rmax(prms, r_max, m200m_dmo, r200m_dmo, c200m_dmo):
    '''
    Return NFW profiles with up to r_max
    '''
    profile_args = {'m_x': m200m_dmo,
                    'c_x': c200m_dmo,
                    'r_x': r200m_dmo}
    dm_kwargs = {'cosmo': prms.cosmo,
                 'r_h': r_max,
                 'profile': dp.profile_NFW,
                 'profile_args': profile_args,
                 'profile_mass': lambda r, **kwargs: dp.m_NFW(r, **profile_args),
                 'profile_f': dp.profile_NFW_f,
                 'profile_f_args': profile_args}

    dens_dm = dens.Profile(**dm_kwargs)
    return dens_dm

# ------------------------------------------------------------------------------
# End of load_dm_dmo_rmax()
# ------------------------------------------------------------------------------

def load_dm_rmax(prms, r_max, m500c, r500c, c500c):
    '''
    Return NFW profiles up to r_max
    '''
    profile_args = {'m_x': m500c,
                    'c_x': c500c,
                    'r_x': r500c}

    dm_kwargs = {'cosmo': prms.cosmo,
                 'r_h': r_max,
                 'profile': dp.profile_NFW,
                 'profile_args': profile_args,
                 'profile_mass': lambda r, **kwargs: dp.m_NFW(r, **profile_args),
                 'profile_f': dp.profile_NFW_f,
                 'profile_f_args': profile_args}

    dens_dm = dens.Profile(**dm_kwargs)
    return dens_dm

# ------------------------------------------------------------------------------
# End of load_dm_rmax()
# ------------------------------------------------------------------------------

def load_gas_plaw_r500c_rmax(prms, rho_500c, r_max, gamma, rc=None,
                             beta=None, f_gas500=None, q_f=50,
                             q_rc=50, q_beta=50):
    '''
    Return a beta profile upto r500c and a power law with index gamma upto r_max,
    or wherever the baryon fraction is reached.
    '''
    r_range = np.array([np.logspace(prms.r_min, np.log10(rm), prms.r_bins)
                        for i,rm in enumerate(r_max)])

    # the fit parameters are for the measured profile that assumed h=0.7
    prof_gas = dp.profile_beta_plaw(r_range,
                                    m_x=f_gas500 * prms.m500c / 0.7,
                                    r_x=prms.r500c / 0.7,
                                    rc=rc * prms.r500c / 0.7,
                                    beta=np.array([beta] * prms.r500c.shape[0]),
                                    r_y=r_max / 0.7,
                                    gamma=np.array([gamma] * prms.r500c.shape[0]),
                                    rho_x=rho_500c * (0.7)**2) * (0.7)**(-2.)

    profile_args =  {'m_x': f_gas500 * prms.m500c / 0.7,
                     'r_x': prms.r500c / 0.7,
                     'rc': rc * prms.r500c / 0.7,
                     'beta': np.array([beta] * prms.r500c.shape[0]),
                     'r_y': r_max / 0.7,
                     'gamma': np.array([gamma] * prms.r500c.shape[0]),
                     'rho_x': rho_500c * (0.7)**2}
    gas_kwargs = {'cosmo': prms.cosmo,
                  'r_h': r_max,
                  'profile': prof_gas,
                  'profile_args': profile_args,
                  'profile_mass': lambda r, **kwargs: dp.m_beta_plaw(r, **profile_args) * (0.7)**(-2.),
                  # compute FT in dens
                  'profile_f': None}

    dens_gas = dens.Profile(**gas_kwargs)
    return dens_gas

# ----------------------------------------------------------------------
# End of load_gas_plaw_r500c_rmax()
# ----------------------------------------------------------------------

def load_centrals_rmax(prms, r_max, m200m_dmo, f_comp='cen'):
    '''
    Return delta profiles with fstars_500c = f_obs

    '''
    # stellar fraction
    f_cen = d.f_stars(m200m_dmo / 0.7, comp=f_comp)

    profile_args = {'m_range': f_cen * m200m_dmo}
    cen_kwargs = {'cosmo': prms.cosmo,
                  'r_h': r_max,
                  'profile': dp.profile_delta,
                  'profile_args': profile_args,
                  'profile_mass': lambda r, **kwargs: dp.m_delta(r, **profile_args),
                  'profile_f': dp.profile_delta_f,
                  'profile_f_args': profile_args}

    dens_cen = dens.Profile(**cen_kwargs)
    return dens_cen

# ------------------------------------------------------------------------------
# End of load_centrals_rmax()
# ------------------------------------------------------------------------------

def load_satellites_rmax(prms, r_max, m200m_dmo, r200m_dmo, c200m_dmo, f_c=0.86):
    '''
    Return NFW profiles with fstars_500c = f_obs

    Parameters
    ----------
    f_c : float
      ratio between c_sat(m) and c_dm(m) used in iHOD
    prms : p.Parameters object
      contains relevant model info
    bar2dmo : bool
      specifies whether to carry out hmf conversion for missing m200m
    '''
    # stellar fraction
    f_sat = d.f_stars(m200m_dmo / 0.7, comp='sat')
    c_sat = f_c * c200m_dmo

    profile_args = {'m_x': f_sat * m200m_dmo,
                    'c_x': c_sat,
                    'r_x': r200m_dmo}
    sat_kwargs = {'cosmo': prms.cosmo,
                  'r_h': r_max,
                  'profile': dp.profile_NFW,
                  'profile_args': profile_args,
                  'profile_mass': lambda r, **kwargs: dp.m_NFW(r, **profile_args),
                  'profile_f': dp.profile_NFW_f,
                  'profile_f_args': profile_args}

    dens_sat = dens.Profile(**sat_kwargs)
    return dens_sat

# ------------------------------------------------------------------------------
# End of load_satellites_rmax()
# ------------------------------------------------------------------------------

@np.vectorize
def m500c_to_m200m_dmo(m500c, r500c, f_gas500c, f_c, cosmo):
    '''
    For a given measured halo mass, radius and gas fraction, compute the
    corresponding DMO equivalent mass taking into account the stellar
    contribution from HOD modeling

    Parameters
    ----------
    m500c : float or array
      halo masses
    r500c : float or array
      corresponding halo radii
    f_gas500c : float or array
      corresponding measured gas fractions
    f_c : float or array
      ratio between the satellite and DMO halo concentration
    cosmo : dict
      cosmology parameters

    Returns
    -------
    m500c_dmo : float or array
      resulting DMO equivalent halo masses
    '''
    def m_diff(m200m_dmo):
        # for a given halo mass, we know the concentration
        c200m_dmo = tools.c_duffy(m200m_dmo).reshape(-1)
        r200m_dmo = tools.mass_to_radius(m200m_dmo, 200 * cosmo.rho_m)

        # this give stellar fraction & concentration
        f_cen500c = d.f_stars(m200m_dmo / 0.7, 'cen') * m200m_dmo / m500c
        f_sat200m = d.f_stars(m200m_dmo / 0.7, 'sat')

        # which allows us to compute the stellar fraction at r500c
        f_sat500c = dp.m_NFW(r500c, m_x=f_sat200m*m200m_dmo, c_x=f_c*c200m_dmo,
                            r_x=r200m_dmo) / m500c

        # this is NOT m500c for our DMO halo, this is our DMO halo
        # evaluated at r500c for the observations, which when scaled
        # should match the observed m500c
        m_dmo_r500c = dp.m_NFW(r500c, m_x=m200m_dmo, c_x=c200m_dmo, r_x=r200m_dmo)

        m500c_cor = m_dmo_r500c * ((1 - cosmo.omegab / cosmo.omegam) /
                                   (1 - f_gas500c - f_cen500c - f_sat500c))

        return m500c_cor - m500c

    m200m_dmo = opt.brentq(m_diff, m500c / 10., 10. * m500c)
    return m200m_dmo

# ----------------------------------------------------------------------
# End of m500c_to_m200m_dmo()
# ----------------------------------------------------------------------

def m200m_obs_from_profs(profs, idcs):
    '''
    Fast way of getting the mass
    '''
    m200m = np.zeros_like(idcs)
    for idx in idcs:
        for prof in profs:
            args = prof.profile_args
            for k,v in args:
                args[k] = v[idx]
                

def load_gamma(prms, r_max,
               gamma=np.array([2.]),
               f_c=0.86,
               q_f=50, q_rc=50, q_beta=50,
               delta=False, bar2dmo=True):
    '''
    Load all of our different models, the ones upto r200m and the ones upto
    r_max.

    Parameters
    ----------
    prms : p.Parameters object
      model parameters
    r_max : float
      maximum radius to extend our profiles upto
    gamma : array
      values of the slope to compute for
    f_c : float
      ratio between satellite concentration and DM concentration
    q_f : int
      quantile for which to compute the f_gas,500c relation
    q_rc : int
      quantile for which to fit r_c
    q_beta : int
      quantile for which to fit beta
    delta : bool
      assume delta profile for the stellar contribution
    bar2dmo : bool
      convert halo mass in halo mass function to account for baryonic correction

    Returns
    -------
    results : dict
      dictionary containing the power.Power objects corresponding to our models
    '''
    # We assume for now that our masses are the X-ray derived quantities
    # gas fractions
    # h=0.7 needs to be converted here
    f_prms = d.f_gas_prms(prms.cosmo, q=50)
    f_gas500 = d.f_gas(prms.m500c / 0.7, cosmo=prms.cosmo, **f_prms)
    rc, beta = d.fit_prms(x=500, q_rc=50, q_beta=50)

    # we compute the DMO equivalent halo mass
    # all models will have the same DMO equivalent halo mass
    m200m_dmo = m500c_to_m200m_dmo(prms.m500c, prms.r500c, f_gas500, 0.86, prms.cosmo)
    r200m_dmo = tools.mass_to_radius(m200m_dmo, 200 * prms.cosmo.rho_m)
    c200m_dmo = tools.c_duffy(m200m_dmo).reshape(-1)
    r_max = r200m_dmo


    # the data assumed h=0.7, but resulting f_star is independent of h in our
    # model
    f_stars = d.f_stars(m200m_dmo / 0.7, 'all')
    f_cen = d.f_stars(m200m_dmo / 0.7, 'cen')
    f_sat = d.f_stars(m200m_dmo / 0.7, 'sat')
    c_sat = f_c * c200m_dmo

    # load stars
    if not delta:
        cen_rmax = load_centrals_rmax(prms=prms,
                                      r_max=r_max,
                                      m200m_dmo=m200m_dmo,
                                      f_comp='cen')
        sat_rmax = load_satellites_rmax(prms=prms,
                                        r_max=r_max,
                                        m200m_dmo=m200m_dmo,
                                        r200m_dmo=r200m_dmo,
                                        c200m_dmo=c200m_dmo,
                                        f_c=f_c)
        stars_rmax = cen_rmax + sat_rmax

    else:
        stars_rmax = load_centrals_rmax(prms=prms,
                                        r_max=r_max,
                                        m200m_dmo=m200m_dmo,
                                        f_comp='all')

    # now load dmo profiles
    dm_dmo_rmax = load_dm_dmo_rmax(prms=prms,
                                   r_max=r200m_dmo,
                                   m200m_dmo=m200m_dmo,
                                   r200m_dmo=r200m_dmo,
                                   c200m_dmo=c200m_dmo)

    pow_dm_dmo_rmax = power.Power(m200m_dmo=m200m_dmo,
                                  m200m_obs=m200m_dmo,
                                  prof=dm_dmo_rmax,
                                  bar2dmo=False)

    # compute density at r500c for the beta profile
    rho_500c = (dp.profile_beta(prms.r500c.reshape(-1,1),
                                m_x=f_gas500 * prms.m500c / 0.7,
                                r_x=prms.r500c / 0.7,
                                rc=rc * prms.r500c / 0.7,
                                beta=np.array([beta]*prms.r500c.shape[0])).reshape(-1) *
                (0.7)**(-2.))

    results = {}
    for idx_g, g in enumerate(gamma):
        gas_plaw_r500c_rmax = load_gas_plaw_r500c_rmax(prms=prms,
                                                       rho_500c=rho_500c,
                                                       r_max=r_max,
                                                       gamma=g,
                                                       rc=rc, beta=beta, f_gas500=f_gas500,
                                                       q_f=q_f, q_rc=q_rc, q_beta=q_beta)

        # now compute DM parameters
        f_stars500c = stars_rmax.m_r(prms.r500c) / prms.m500c
        f_gas500c = gas_plaw_r500c_rmax.m_r(prms.r500c) / prms.m500c
        m500c_dm = prms.m500c * (1 - f_gas500c - f_stars500c)
        c500c_dm = c200m_dmo * prms.r500c / r200m_dmo

        dm_plaw_r500c_rmax = load_dm_rmax(prms=prms,
                                          r_max=r_max,
                                          m500c=m500c_dm,
                                          r500c=prms.r500c,
                                          c500c=c500c_dm)

        prof_tot = dm_plaw_r500c_rmax + gas_plaw_r500c_rmax + stars_rmax

        # need to make the function specific to one mass!
        r200m_obs = np.zeros_like(prms.r500c)
        r_fb = np.zeros_like(prms.r500c)
        for idx in np.arange(0, prms.r500c.shape[0]):
            m_r = (lambda r: dp.m_NFW(r, **{k: v[idx] for k,v in
                                            dm_plaw_r500c_rmax.profile_args.iteritems()}) +
                   dp.m_beta_plaw(r, **{k: v[idx] for k,v in
                                        gas_plaw_r500c_rmax.profile_args.iteritems()}) +
                   dp.m_delta(r, **{k: v[idx] for k,v in cen_rmax.profile_args.iteritems()}) +
                   dp.m_NFW(r, **{k: v[idx] for k,v in sat_rmax.profile_args.iteritems()}))
            fb = (lambda r: (dp.m_beta_plaw(r, **{k: v[idx] for k,v in
                                                  gas_plaw_r500c_rmax.profile_args.iteritems()}) +
                             dp.m_delta(r, **{k: v[idx] for k,v in cen_rmax.profile_args.iteritems()}) +
                             dp.m_NFW(r, **{k: v[idx] for k,v in sat_rmax.profile_args.iteritems()})) /
                  (dp.m_NFW(r, **{k: v[idx] for k,v in
                                  dm_plaw_r500c_rmax.profile_args.iteritems()}) +
                   dp.m_beta_plaw(r, **{k: v[idx] for k,v in
                                        gas_plaw_r500c_rmax.profile_args.iteritems()}) +
                   dp.m_delta(r, **{k: v[idx] for k,v in cen_rmax.profile_args.iteritems()}) +
                   dp.m_NFW(r, **{k: v[idx] for k,v in sat_rmax.profile_args.iteritems()})))
            r200m_obs[idx] = dp.r200m_from_m(m_r, prms.cosmo)
            try:
                r_fb[idx] = dp.r_fb_from_m(fb, prms.cosmo)
            except ValueError:
                r_fb[idx] = np.inf

        m200m_obs = tools.radius_to_mass(r200m_obs, 200 * prms.cosmo.rho_m)
        m_obs_200m_dmo = prof_tot.m_r(r200m_dmo)

        pow_gas_plaw_r500c_rmax = power.Power(m200m_dmo=m200m_dmo,
                                              m200m_obs=m200m_obs,
                                              prof=prof_tot,
                                              bar2dmo=bar2dmo)

        results['{:d}'.format(idx_g)] = {'pow': pow_gas_plaw_r500c_rmax,
                                         'pow_dmo': pow_dm_dmo_rmax,
                                         'd_gas': gas_plaw_r500c_rmax,
                                         'd_dm': dm_plaw_r500c_rmax,
                                         'd_stars': stars_rmax,
                                         'd_tot': prof_tot,
                                         'm200m_obs': m200m_obs,
                                         'm_obs_200m_dmo': m_obs_200m_dmo,
                                         'm200m_dmo': m200m_dmo,
                                         'r_fb': r_fb,
                                         'r200m_obs': r200m_obs,
                                         'r_max': r_max}

    return results

# ----------------------------------------------------------------------
# End of load_gamma()
# ----------------------------------------------------------------------

def plot_profiles_paper(dens_gas,
                        dens_gas_r500c_r200m,
                        dens_gas_r200m_5r500c,
                        dens_stars_nodelta,
                        dens_dm,
                        rho_k=False,
                        prms=p.prms):
    '''
    Plot the density for our different gas profiles in one plot
    '''
    fig = plt.figure(figsize=(20,8))
    ax1 = fig.add_axes([0.1,0.1,0.266,0.8])
    ax2 = fig.add_axes([0.366,0.1,0.266,0.8])
    ax3 = fig.add_axes([0.632,0.1,0.266,0.8])

    idx_1 = 0
    idx_2 = 50
    idx_3 = -1
    r200m = prms.r200m
    reload(pl)
    pl.set_style('line')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if not rho_k:
        norm = prms.rho_crit
        # Plot idx_1
        # gas
        l2, = ax1.plot(dens_gas_r500c_r200m.r_range[idx_1] / r200m[idx_1],
                       (dens_gas_r500c_r200m.rho_r[idx_1] *
                        dens_gas_r500c_r200m.f_comp[idx_1] / norm),
                       lw=3, c=colors[1])
        l3, = ax1.plot(dens_gas_r200m_5r500c.r_range[idx_1] / r200m[idx_1],
                       (dens_gas_r200m_5r500c.rho_r[idx_1] *
                        dens_gas_r200m_5r500c.f_comp[idx_1] / norm),
                       lw=4, c=colors[0])
        l1, = ax1.plot(dens_gas.r_range[idx_1] / r200m[idx_1],
                       dens_gas.rho_r[idx_1] * dens_gas.f_comp[idx_1]/norm,
                       lw=2, c=colors[2])
        # stars
        markerson = 0.1
        ls, = ax1.plot(dens_stars_nodelta.r_range[idx_1] / r200m[idx_1],
                       dens_stars_nodelta.rho_r[idx_1] * dens_stars_nodelta.f_comp[idx_1]/norm,
                       c='k', marker='*',
                       markevery=markerson,
                       markersize=8)

        # dark matter
        ld, = ax1.plot(dens_dm.r_range[idx_1] / r200m[idx_1],
                       dens_dm.rho_r[idx_1] * dens_dm.f_comp[idx_1]/norm,
                       c='k', marker='o',
                       markevery=markerson,
                       markersize=8)

        # # dark matter
        # ld5r500c, = ax1.plot(dens_dm_5r500c.r_range[idx_1] / r200m[idx_1],
        #                      dens_dm_5r500c.rho_r[idx_1] * dens_dm_5r500c.f_comp[idx_1]/norm,
        #                      c='r', marker='o',
        #                      markevery=markerson,
        #                      markersize=8)

        ax1.axvline(x=prms.r500c[idx_1] / prms.r200m[idx_1], ls='--', c='k')
        ax1.text(x=prms.r500c[idx_1] / prms.r200m[idx_1], y=1e2, s=r'$r_\mathrm{500c}$',
                 ha='left', va='bottom')

        # Plot idx_2
        # gas
        ax2.plot(dens_gas_r500c_r200m.r_range[idx_2] / r200m[idx_2],
                 (dens_gas_r500c_r200m.rho_r[idx_2] *
                  dens_gas_r500c_r200m.f_comp[idx_2] / norm),
                 lw=3, c=colors[1])
        ax2.plot(dens_gas_r200m_5r500c.r_range[idx_2] / r200m[idx_2],
                 (dens_gas_r200m_5r500c.rho_r[idx_2] *
                  dens_gas_r200m_5r500c.f_comp[idx_2] / norm),
                 lw=4, c=colors[0])
        ax2.plot(dens_gas.r_range[idx_2] / r200m[idx_2],
                 dens_gas.rho_r[idx_2] * dens_gas.f_comp[idx_2]/norm,
                 lw=2, c=colors[2])

        # stars
        ls, = ax2.plot(dens_stars_nodelta.r_range[idx_2] / r200m[idx_2],
                       dens_stars_nodelta.rho_r[idx_2] * dens_stars_nodelta.f_comp[idx_2]/norm,
                       c='k', marker='*',
                       markevery=markerson,
                       markersize=8,
                       label=r'$\mathtt{\star\_NFW}$')

        # dark matter
        ld, = ax2.plot(dens_dm.r_range[idx_2] / r200m[idx_2],
                       dens_dm.rho_r[idx_2] * dens_dm.f_comp[idx_2]/norm,
                       c='k', marker='o',
                       markevery=markerson,
                       markersize=8,
                       label='dark matter')

        # # dark matter
        # ld5r500c, = ax2.plot(dens_dm_5r500c.r_range[idx_2] / r200m[idx_2],
        #                      dens_dm_5r500c.rho_r[idx_2] * dens_dm_5r500c.f_comp[idx_2]/norm,
        #                      c='r', marker='o',
        #                      markevery=markerson,
        #                      markersize=8)

        ax2.axvline(x=prms.r500c[idx_2] / prms.r200m[idx_2], ls='--', c='k')
        ax2.text(x=prms.r500c[idx_2] / prms.r200m[idx_2], y=1e2, s=r'$r_\mathrm{500c}$',
                 ha='left', va='bottom')

        # Plot idx_3
        # gas
        ax3.plot(dens_gas_r500c_r200m.r_range[idx_3] / r200m[idx_3],
                 (dens_gas_r500c_r200m.rho_r[idx_3] *
                  dens_gas_r500c_r200m.f_comp[idx_3] / norm),
                 lw=3, c=colors[1])
        ax3.plot(dens_gas_r200m_5r500c.r_range[idx_3] / r200m[idx_3],
                 (dens_gas_r200m_5r500c.rho_r[idx_3] *
                  dens_gas_r200m_5r500c.f_comp[idx_3] / norm),
                 lw=4, c=colors[0])
        ax3.plot(dens_gas.r_range[idx_3] / r200m[idx_3],
                 dens_gas.rho_r[idx_3] * dens_gas.f_comp[idx_3]/norm,
                 lw=2, c=colors[2])

        # stars
        ls, = ax3.plot(dens_stars_nodelta.r_range[idx_3] / r200m[idx_3],
                       dens_stars_nodelta.rho_r[idx_3] * dens_stars_nodelta.f_comp[idx_3]/norm,
                       c='k', marker='*',
                       markevery=markerson,
                       markersize=8)

        # dark matter
        ld, = ax3.plot(dens_dm.r_range[idx_3] / r200m[idx_3],
                       dens_dm.rho_r[idx_3] * dens_dm.f_comp[idx_3]/norm,
                       c='k', marker='o',
                       markevery=markerson,
                       markersize=8)

        # # dark matter
        # ld5r500c, = ax3.plot(dens_dm_5r500c.r_range[idx_3] / r200m[idx_3],
        #                      dens_dm_5r500c.rho_r[idx_3] * dens_dm_5r500c.f_comp[idx_3]/norm,
        #                      c='r', marker='o',
        #                      markevery=markerson,
        #                      markersize=8)

        ax3.axvline(x=prms.r500c[idx_3] / prms.r200m[idx_3], ls='--', c='k')
        ax3.text(x=prms.r500c[idx_3] / prms.r200m[idx_3], y=110, s=r'$r_\mathrm{500c}$',
                 ha='left', va='bottom')

        ax1.set_xlim(1e-2, 3)
        ax1.set_ylim(1e-1, 1e4)
        ax2.set_xlim(1e-2, 3)
        ax2.set_ylim(1e-1, 1e4)
        ax3.set_xlim(1e-2, 3)
        ax3.set_ylim(1e-1, 1e4)

        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax3.set_xscale('log')
        ax3.set_yscale('log')

        ax1.tick_params(axis='x', which='major', pad=6)
        ax2.tick_params(axis='x', which='major', pad=6)
        ax3.tick_params(axis='x', which='major', pad=6)

        ax2.set_xlabel(r'$r/r_\mathrm{200m}$', labelpad=-10)
        ax1.set_ylabel(r'$\rho(r)/\rho_\mathrm{c}$')
        ax2.set_yticklabels([])
        ax3.set_yticklabels([])
        # ticks2 = ax2.get_xticklabels()
        # ticks2[-6].set_visible(False)
        # ticks3 = ax3.get_xticklabels()
        # ticks3[-6].set_visible(False)

        ax1.set_title(r'$m_{\mathrm{200m}} = 10^{%.1f} \, h^{-1} \, \mathrm{M_\odot}$'%np.log10(prms.m200m[idx_1]), y=1.015, fontsize=28)
        ax2.set_title(r'$m_{\mathrm{200m}} = 10^{%.1f} \, h^{-1} \, \mathrm{M_\odot}$'%np.log10(prms.m200m[idx_2]), y=1.015, fontsize=28)
        ax3.set_title(r'$m_{\mathrm{200m}} = 10^{%.1f} \, h^{-1} \, \mathrm{M_\odot}$'%np.log10(prms.m200m[idx_3]), y=1.015, fontsize=28)


        leg1 = ax1.legend([l1, l2, l3],
                          [r'$\mathtt{\beta\_r200m\_nofb}$',
                           r'$\mathtt{\beta\_r500c\_fb\_r200m}$',
                           r'$\mathtt{\beta\_r200m\_fb\_5r500c}$'],
                          loc=2, fontsize=28, frameon=True, framealpha=0.8)
        leg1.get_frame().set_linewidth(0.0)

        leg2 = ax2.legend(loc=3, fontsize=28, frameon=True, framealpha=0.8)
        leg2.get_frame().set_linewidth(0.0)

        plt.savefig('obs_rho_extrapolated.pdf', transparent=True,
                    bbox_inches='tight')

    else:
        norm = 1.
        # Plot idx_1
        # gas
        ax1.plot(dens_gas.k_range,
                 dens_gas.rho_k[idx_1] * dens_gas.f_comp[idx_1]/norm,
                 lw=2, c=colors[2], label=r'$\mathtt{\beta\_r200m\_nofb}$')
        ax1.plot(dens_gas_r500c_r200m.k_range,
                 (dens_gas_r500c_r200m.rho_k[idx_1] *
                  dens_gas_r500c_r200m.f_comp[idx_1] / norm),
                 lw=3, c=colors[1], label=r'$\mathtt{\beta\_r500c\_fb\_r200m}$')
        ax1.plot(dens_gas_r200m_5r500c.k_range,
                 (dens_gas_r200m_5r500c.rho_k[idx_1] *
                  dens_gas_r200m_5r500c.f_comp[idx_1] / norm),
                 lw=4, c=colors[0], label=r'$\mathtt{\beta\_r200m\_fb\_5r500c}$')

        # stars
        markerson = 0.1
        ls, = ax1.plot(dens_stars_nodelta.k_range,
                       dens_stars_nodelta.rho_k[idx_1] * dens_stars_nodelta.f_comp[idx_1]/norm,
                       c='k', marker='*',
                       markevery=markerson,
                       markersize=8,
                       label=r'$\mathtt{\star\_NFW}$')

        # dark matter
        ld, = ax1.plot(dens_dm.k_range,
                       dens_dm.rho_k[idx_1] * dens_dm.f_comp[idx_1]/norm,
                       c='k', marker='o',
                       markevery=markerson,
                       markersize=8,
                       label='dark matter')

        # # dark matter
        # ld5r500c, = ax1.plot(dens_dm_5r500c.k_range,
        #                      dens_dm_5r500c.rho_k[idx_1] * dens_dm_5r500c.f_comp[idx_1]/norm,
        #                      c='r', marker='o',
        #                      markevery=markerson,
        #                      markersize=8,
        #                      label='dark matter')


        # Plot idx_2
        # gas
        ax2.plot(dens_gas.k_range,
                 dens_gas.rho_k[idx_2] * dens_gas.f_comp[idx_2]/norm,
                 lw=2, c=colors[2], label=r'$\mathtt{\beta\_r200m\_nofb}$')
        ax2.plot(dens_gas_r500c_r200m.k_range,
                 (dens_gas_r500c_r200m.rho_k[idx_2] *
                  dens_gas_r500c_r200m.f_comp[idx_2] / norm),
                 lw=3, c=colors[1], label=r'$\mathtt{\beta\_r500c\_fb\_r200m}$')
        ax2.plot(dens_gas_r200m_5r500c.k_range,
                 (dens_gas_r200m_5r500c.rho_k[idx_2] *
                  dens_gas_r200m_5r500c.f_comp[idx_2] / norm),
                 lw=4, c=colors[0], label=r'$\mathtt{\beta\_r200m\_fb\_5r500c}$')

        # stars
        markerson = 0.1
        ls, = ax2.plot(dens_stars_nodelta.k_range,
                       dens_stars_nodelta.rho_k[idx_2] * dens_stars_nodelta.f_comp[idx_2]/norm,
                       c='k', marker='*',
                       markevery=markerson,
                       markersize=8,
                       label=r'$\mathtt{\star\_NFW}$')

        # dark matter
        ld, = ax2.plot(dens_dm.k_range,
                       dens_dm.rho_k[idx_2] * dens_dm.f_comp[idx_2]/norm,
                       c='k', marker='o',
                       markevery=markerson,
                       markersize=8,
                       label='dark matter')

        # # dark matter
        # ld5r500c, = ax2.plot(dens_dm_5r500c.k_range,
        #                      dens_dm_5r500c.rho_k[idx_2] * dens_dm_5r500c.f_comp[idx_2]/norm,
        #                      c='r', marker='o',
        #                      markevery=markerson,
        #                      markersize=8,
        #                      label='dark matter')


        # Plot idx_3
        # gas
        ax3.plot(dens_gas.k_range,
                 dens_gas.rho_k[idx_3] * dens_gas.f_comp[idx_3]/norm,
                 lw=2, c=colors[2], label=r'$\mathtt{\beta\_r200m\_nofb}$')
        ax3.plot(dens_gas_r500c_r200m.k_range,
                 (dens_gas_r500c_r200m.rho_k[idx_3] *
                  dens_gas_r500c_r200m.f_comp[idx_3] / norm),
                 lw=3, c=colors[1], label=r'$\mathtt{\beta\_r500c\_fb\_r200m}$')
        ax3.plot(dens_gas_r200m_5r500c.k_range,
                 (dens_gas_r200m_5r500c.rho_k[idx_3] *
                  dens_gas_r200m_5r500c.f_comp[idx_3] / norm),
                 lw=4, c=colors[0], label=r'$\mathtt{\beta\_r200m\_fb\_5r500c}$')

        # stars
        markerson = 0.1
        ls, = ax3.plot(dens_stars_nodelta.k_range,
                       dens_stars_nodelta.rho_k[idx_3] * dens_stars_nodelta.f_comp[idx_3]/norm,
                       c='k', marker='*',
                       markevery=markerson,
                       markersize=8,
                       label=r'$\mathtt{\star\_NFW}$')

        # dark matter
        ld, = ax3.plot(dens_dm.k_range,
                       dens_dm.rho_k[idx_3] * dens_dm.f_comp[idx_3]/norm,
                       c='k', marker='o',
                       markevery=markerson,
                       markersize=8,
                       label='dark matter')

        # # dark matter
        # ld5r500c, = ax3.plot(dens_dm_5r500c.k_range,
        #                      dens_dm_5r500c.rho_k[idx_3] * dens_dm_5r500c.f_comp[idx_3]/norm,
        #                      c='r', marker='o',
        #                      markevery=markerson,
        #                      markersize=8,
        #                      label='dark matter')

        ax1.set_xscale('log')
        ax2.set_xscale('log')
        ax3.set_xscale('log')
        ax1.set_yscale('symlog',linthreshy=1e-4)
        ax2.set_yscale('symlog',linthreshy=1e-4)
        ax3.set_yscale('symlog',linthreshy=1e-4)

        ax1.set_xlim(1,100)
        ax2.set_xlim(1,100)
        ax3.set_xlim(1,100)
        ax1.set_ylim(-1, 1)
        ax2.set_ylim(-1, 1)
        ax3.set_ylim(-1, 1)

        ax1.tick_params(axis='x', which='major', pad=6)
        ax2.tick_params(axis='x', which='major', pad=6)
        ax3.tick_params(axis='x', which='major', pad=6)
        ax1.tick_params(axis='x', which='minor', bottom='on', top='on')
        ax2.tick_params(axis='x', which='minor', bottom='on', top='on')
        ax3.tick_params(axis='x', which='minor', bottom='on', top='on')

        ax2.set_xlabel(r'$k \, [h \, \mathrm{Mpc}^{-1}]$', labelpad=-10)
        ax1.set_ylabel(r'$f_\mathrm{i}(m)u(k|m)$')
        ax2.set_yticklabels([])
        ax3.set_yticklabels([])
        # ticks2 = ax2.get_xticklabels()
        # ticks2[-6].set_visible(False)
        # ticks3 = ax3.get_xticklabels()
        # ticks3[-6].set_visible(False)

        ax1.set_title(r'$m_{\mathrm{200m}} = 10^{%.1f} \, h^{-1} \, \mathrm{M_\odot}$'%np.log10(prms.m200m[idx_1]), y=1.015, fontsize=28)
        ax2.set_title(r'$m_{\mathrm{200m}} = 10^{%.1f} \, h^{-1} \, \mathrm{M_\odot}$'%np.log10(prms.m200m[idx_2]), y=1.015, fontsize=28)
        ax3.set_title(r'$m_{\mathrm{200m}} = 10^{%.1f} \, h^{-1} \, \mathrm{M_\odot}$'%np.log10(prms.m200m[idx_3]), y=1.015, fontsize=28)

        ax3.legend(loc='best', fontsize=28)
        plt.savefig('obs_rho_k_extrapolated.pdf', transparent=True,
                    bbox_inches='tight')

# ------------------------------------------------------------------------------
# End of plot_profiles_gas_paper()
# ------------------------------------------------------------------------------

# def plot_fgas200m_paper(comp_gas, comp_gas_r500c_5r500c, prms=p.prms):
#     '''
#     Plot gas mass fractions at r200m for our different models
#     '''
#     fig = plt.figure(figsize=(10,9))
#     ax = fig.add_subplot(111)

#     f_b = 1 - prms.f_dm

#     pl.set_style('line')
#     ax.plot(comp_gas.m200m, comp_gas.f_comp, label='model 1')
#     ax.plot(comp_gas_r500c_5r500c.m200m, comp_gas_r500c_5r500c.f_comp,
#             label='model 3')
#     ax.axhline(y=f_b, c='k', ls='--')

#     ax.tick_params(axis='x', which='major', pad=6)
#     text_props = ax.get_xticklabels()[0].get_font_properties()

#     # add annotation to f_bar
#     ax.annotate(r'$f_{\mathrm{b}}$',
#                  # xy=(1e14, 0.16), xycoords='data',
#                  # xytext=(1e14, 0.15), textcoords='data',
#                  xy=(10**(11), f_b), xycoords='data',
#                  xytext=(1.2 * 10**(11),
#                          f_b * 0.95), textcoords='data',
#                  fontproperties=text_props)

#     ax.set_xscale('log')
#     ax.set_xlabel('$m_\mathrm{200m} \, [\mathrm{M_\odot}/h]$')
#     ax.set_ylabel('$f_\mathrm{gas,200m}$')
#     ax.legend(loc=4)
#     plt.savefig('obs_fgas_extrapolated.pdf', transparent=True)

# # ------------------------------------------------------------------------------
# # End of plot_fgas200m_paper()
# # ------------------------------------------------------------------------------
