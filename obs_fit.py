import numpy as np
import scipy.optimize as opt
import scipy.interpolate as intp
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import sys
import os
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
import cProfile

def do_cprofile(func):
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats(sort='cumtime')
    return profiled_func

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

    profile_mass = lambda r, **kwargs: dp.m_NFW(r, **profile_args)

    m_h = profile_mass(r_max)
    r_h = r_max
    c_h = c200m_dmo * r_h / r200m_dmo
    profile_f_args = {'m_h': m_h,
                      'c_h': c_h,
                      'r_h': r_h}

    dm_kwargs = {'cosmo': prms.cosmo,
                 'r_h': r_max,
                 'profile': dp.profile_NFW,
                 'profile_args': profile_args,
                 'profile_mass': profile_mass,
                 'profile_f': dp.profile_NFW_f,
                 'profile_f_args': profile_f_args}

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

    profile_mass = lambda r, **kwargs: dp.m_NFW(r, **profile_args)

    m_h = profile_mass(r_max)
    r_h = r_max
    c_h = c500c * r_h / r500c
    profile_f_args = {'m_h': m_h,
                      'c_h': c_h,
                      'r_h': r_h}

    dm_kwargs = {'cosmo': prms.cosmo,
                 'r_h': r_max,
                 'profile': dp.profile_NFW,
                 'profile_args': profile_args,
                 'profile_mass': profile_mass,
                 'profile_f': dp.profile_NFW_f,
                 'profile_f_args': profile_f_args}

    dens_dm = dens.Profile(**dm_kwargs)
    return dens_dm

# ------------------------------------------------------------------------------
# End of load_dm_rmax()
# ------------------------------------------------------------------------------

def load_gas_plaw_r500c_rmax(prms, rho_500c, r_flat, r_max, gamma, rc=None,
                             beta=None, f_gas500=None, q_f=50,
                             q_rc=50, q_beta=50):
    '''
    Return a beta profile upto r500c and a power law with index gamma upto r_max,
    or wherever the baryon fraction is reached.
    '''
    r_range = np.array([np.logspace(prms.r_min, np.log10(rm), prms.r_bins)
                        for i,rm in enumerate(r_max)])

    # the fit parameters are for the measured profile that assumed h=0.7
    # HOWEVER, we do not need to correct for this, since all the scalings
    # just correspond to r and rho, which can be scaled afterwards anyway
    # prof_gas = dp.profile_beta_plaw(r_range,
    #                                 m_x=f_gas500 * prms.m500c,
    #                                 r_x=prms.r500c,
    #                                 rc=rc * prms.r500c,
    #                                 beta=np.array([beta] * prms.r500c.shape[0]),
    #                                 r_y=r_max,
    #                                 gamma=gamma,
    #                                 rho_x=rho_500c)
    prof_gas = dp.profile_beta_plaw_uni(r_range,
                                        m_x=f_gas500 * prms.m500c,
                                        r_x=prms.r500c,
                                        rc=rc * prms.r500c,
                                        beta=np.array([beta] * prms.r500c.shape[0]),
                                        r_y=r_flat * prms.r500c,
                                        gamma=gamma,
                                        rho_x=rho_500c)

    profile_args =  {'m_x': f_gas500 * prms.m500c,
                     'r_x': prms.r500c,
                     'rc': rc * prms.r500c,
                     'beta': np.array([beta] * prms.r500c.shape[0]),
                     'r_y': r_flat * prms.r500c,
                     'gamma': gamma,
                     'rho_x': rho_500c}
    gas_kwargs = {'cosmo': prms.cosmo,
                  'r_h': r_max,
                  'profile': prof_gas,
                  'profile_args': profile_args,
                  'profile_mass': lambda r, **kwargs: dp.m_beta_plaw_uni(r, **profile_args),
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

    profile_mass = lambda r, **kwargs: dp.m_NFW(r, **profile_args)

    m_h = profile_mass(r_max)
    r_h = r_max
    c_h = c200m_dmo * r_h / r200m_dmo
    profile_f_args = {'m_h': m_h,
                      'c_h': c_h,
                      'r_h': r_h}

    sat_kwargs = {'cosmo': prms.cosmo,
                  'r_h': r_max,
                  'profile': dp.profile_NFW,
                  'profile_args': profile_args,
                  'profile_mass': profile_mass,
                  'profile_f': dp.profile_NFW_f,
                  'profile_f_args': profile_f_args}

    dens_sat = dens.Profile(**sat_kwargs)
    return dens_sat

# ------------------------------------------------------------------------------
# End of load_satellites_rmax()
# ------------------------------------------------------------------------------

def m_dm_stars_from_m500c(m500c, cosmo, f_c=0.86, sigma_lnc=0., bias=False):
    '''
    Determine the enclosed mass profiles for the dark matter and stellar mass
    components

    Parameters
    ----------
    m500c : float
      halo masses
    cosmo : dict
      cosmological parameters
    f_c : float
      ratio between satellite concentration and DM concentration
    bias : bool
      include hydrostatic bias correction

    Returns
    -------
    m_dm, m_stars : functions with arg r
      function to compute halo mass
    '''
    r500c = tools.mass_to_radius(m500c, 500 * cosmo.rho_crit)

    # We assume for now that our masses are the X-ray derived quantities
    # gas fractions
    # h=0.7 needs to be converted here
    f_prms = d.f_gas_prms(cosmo, q=50, bias=bias)
    f_gas500 = d.f_gas(m500c / 0.7, cosmo=cosmo, **f_prms)

    # we compute the DMO equivalent halo mass
    # all models will have the same DMO equivalent halo mass
    m200m_dmo = m500c_to_m200m_dmo(m500c=m500c, r500c=r500c,
                                   f_gas500c=f_gas500, f_c=f_c,
                                   sigma_lnc=sigma_lnc,
                                   cosmo=cosmo)
    r200m_dmo = tools.mass_to_radius(m200m_dmo, 200 * cosmo.rho_m)
    c200m_dmo = tools.c_duffy(m200m_dmo, sigma_lnc=sigma_lnc).reshape(-1)

    # ##### #
    # STARS #
    # ##### #
    # the data assumed h=0.7, but resulting f_star is independent of h in our
    # model
    f_stars = d.f_stars(m200m_dmo / 0.7, 'all')
    f_cen = d.f_stars(m200m_dmo / 0.7, 'cen')
    f_sat = d.f_stars(m200m_dmo / 0.7, 'sat')
    c_sat = f_c * c200m_dmo

    cen_args = {'m_range': f_cen * m200m_dmo}
    sat_args = {'m_x': f_sat * m200m_dmo,
                'c_x': c_sat,
                'r_x': r200m_dmo}

    m_stars = lambda r, **kwargs: (dp.m_delta(r, **cen_args) +
                                   dp.m_NFW(r, **sat_args))

    # ## #
    # DM #
    # ## #
    f_stars500c = m_stars(r500c) / m500c
    m500c_dm = m500c * (1 - f_gas500 - f_stars500c)
    c500c_dm = c200m_dmo * r500c / r200m_dmo
    dm_args = {'m_x': m500c_dm,
               'c_x': c500c_dm,
               'r_x': r500c}

    m_dm = lambda r, **kwargs: dp.m_NFW(r, **dm_args)

    return m_dm, m_stars

# ----------------------------------------------------------------------
# End of m_dm_stars_from_m500c()
# ----------------------------------------------------------------------

def m_gas_from_m500c(m500c, cosmo, gamma, r_flat=None, q_f=50, q_rc=50,
                     q_beta=50, bias=False):
    '''
    Determine the enclosed mass profiles for the gas mass component
    in a power law with index gamma for r>r500c

    Parameters
    ----------
    m500c : float
      halo masses
    cosmo : dict
      cosmological parameters
    gamma : float
      power law slope for beta profile extension
    r_flat : float or None [units of r500c]
      values where gas profile goes flat,
      if None, profile follows power law
    q_f : int
      quantile for which to compute the f_gas,500c relation
    q_rc : int
      quantile for which to fit r_c
    q_beta : int
      quantile for which to fit beta
    bias : bool
      include hydrostatic bias correction

    Returns
    -------
    m_gas : functions with arg r
      function to compute halo mass
    '''
    r500c = tools.mass_to_radius(m500c, 500 * cosmo.rho_crit)

    # We assume for now that our masses are the X-ray derived quantities
    # gas fractions
    # h=0.7 needs to be converted here
    f_prms = d.f_gas_prms(cosmo, q=q_f, bias=bias)
    f_gas500 = d.f_gas(m500c / 0.7, cosmo=cosmo, **f_prms)

    # ### #
    # GAS #
    # ### #
    rc, beta = d.fit_prms(x=500, q_rc=q_rc, q_beta=q_beta)
    rho_500c = dp.profile_beta(r500c.reshape(-1,1),
                               m_x=np.array([f_gas500 * m500c]),
                               r_x=np.array([r500c]),
                               rc=np.array([rc * r500c]),
                               beta=np.array([beta])).reshape(-1)

    gas_args =  {'m_x': np.array([f_gas500 * m500c]),
                 'r_x': np.array([r500c]),
                 'rc': np.array([rc * r500c]),
                 'beta': np.array([beta]),
                 'gamma': np.array([gamma]),
                 'rho_x': np.array([rho_500c])}

    if r_flat is None:
        m_gas = lambda r, **kwargs: dp.m_beta_plaw(r, **gas_args)
    else:
        gas_args['r_y'] = np.array([r_flat * r500c]),
        m_gas = lambda r, **kwargs: dp.m_beta_plaw_uni(r, **gas_args)

    return m_gas

# ----------------------------------------------------------------------
# End of m_gas_from_m500c()
# ----------------------------------------------------------------------

def m_from_model(prms, m200m_dmo, r200m_dmo, c200m_dmo, gamma,
                 q_f=50, q_rc=50, q_beta=50, bias=False,
                 f_c=0.86):
    '''
    Determine the enclosed mass profiles for the baryonic and the total mass
    components

    Parameters
    ----------
    prms : Parameters object
      model parameters
    m200m_dmo : float or array
      DMO equivalent halo masses
    r200m_dmo : float or array
      DMO equivalent halo radii
    c200m_dmo : float or array
      DMO equivalent halo concentration
    gamma : float or array
      slope of power law extension to beta profile
    q_f : int
      quantile for which to compute the f_gas,500c relation
    q_rc : int
      quantile for which to fit r_c
    q_beta : int
      quantile for which to fit beta
    bias : bool
      take into account hydrostatic bias
    f_c : float
      ratio between the satellite and DMO halo concentration

    Returns
    -------
    m_b, m_tot : functions with args r & idx
      function to compute halo mass
    '''
    # ### #
    # GAS #
    # ### #
    # We assume for now that our masses are the X-ray derived quantities
    # gas fractions
    # h=0.7 needs to be converted here
    f_prms = d.f_gas_prms(prms.cosmo, q=q_f, bias=bias)
    f_gas500 = d.f_gas(prms.m500c / 0.7, cosmo=prms.cosmo, **f_prms)
    rc, beta = d.fit_prms(x=500, q_rc=q_rc, q_beta=q_beta)
    rho_500c = dp.profile_beta(prms.r500c.reshape(-1,1),
                               m_x=f_gas500 * prms.m500c,
                               r_x=prms.r500c,
                               rc=rc * prms.r500c,
                               beta=np.array([beta]*prms.r500c.shape[0])).reshape(-1)

    gas_args =  {'m_x': f_gas500 * prms.m500c,
                 'r_x': prms.r500c,
                 'rc': rc * prms.r500c,
                 'beta': np.array([beta] * prms.r500c.shape[0]),
                 'gamma': gamma,
                 'rho_x': rho_500c}
    m_gas = lambda r, sl, **kwargs: dp.m_beta_plaw(r, **{k: v[sl] for k,v in
                                                         gas_args.items()})

    # ##### #
    # STARS #
    # ##### #
    # the data assumed h=0.7, but resulting f_star is independent of h in our
    # model
    f_stars = d.f_stars(m200m_dmo / 0.7, 'all')
    f_cen = d.f_stars(m200m_dmo / 0.7, 'cen')
    f_sat = d.f_stars(m200m_dmo / 0.7, 'sat')
    c_sat = f_c * c200m_dmo

    cen_args = {'m_range': f_cen * m200m_dmo}
    sat_args = {'m_x': f_sat * m200m_dmo,
                'c_x': c_sat,
                'r_x': r200m_dmo}

    m_stars = lambda r, sl, **kwargs: (dp.m_delta(r, **{k: v[sl] for k,v in
                                                        cen_args.items()}) +
                                       dp.m_NFW(r, **{k: v[sl] for k,v in
                                                      sat_args.items()}))

    # ## #
    # DM #
    # ## #
    f_stars500c = m_stars(prms.r500c, np.arange(0,prms.r500c.shape[0])) / prms.m500c
    f_gas500c = m_gas(prms.r500c, np.arange(0,prms.r500c.shape[0])) / prms.m500c
    m500c_dm = prms.m500c * (1 - f_gas500c - f_stars500c)
    c500c_dm = c200m_dmo * prms.r500c / r200m_dmo
    dm_args = {'m_x': m500c_dm,
               'c_x': c500c_dm,
               'r_x': prms.r500c}

    m_dm = lambda r, sl, **kwargs: dp.m_NFW(r, **{k: v[sl] for k,v in
                                                  dm_args.items()})

    m_b = lambda r, sl, **kwargs: m_stars(r, sl) + m_gas(r, sl)
    m_tot = lambda r, sl, **kwargs: m_dm(r, sl) + m_stars(r, sl) + m_gas(r, sl)

    return m_dm, m_gas, m_stars, m_b, m_tot

# ----------------------------------------------------------------------
# End of m_from_model()
# ----------------------------------------------------------------------

def m_gas_model(prms, q_f, q_rc, q_beta, gamma, r_flat=None,
                bias=False):
    '''
    Return the gas mass profile
    '''
    # We assume for now that our masses are the X-ray derived quantities
    # gas fractions
    # h=0.7 needs to be converted here
    f_prms = d.f_gas_prms(prms.cosmo, q=q_f, bias=bias)
    f_gas500 = d.f_gas(prms.m500c / 0.7, cosmo=prms.cosmo, **f_prms)
    rc, beta = d.fit_prms(x=500, q_rc=q_rc, q_beta=q_beta)
    rho_500c = dp.profile_beta(prms.r500c.reshape(-1,1),
                               m_x=f_gas500 * prms.m500c,
                               r_x=prms.r500c,
                               rc=rc * prms.r500c,
                               beta=np.array([beta]*prms.r500c.shape[0])).reshape(-1)

    gas_args =  {'m_x': f_gas500 * prms.m500c,
                 'r_x': prms.r500c,
                 'rc': rc * prms.r500c,
                 'beta': np.array([beta] * prms.r500c.shape[0])}

    if r_flat is None:
        gas_args['gamma'] = gamma
        gas_args['rho_x'] = rho_500c
        m_gas = lambda r, sl, **kwargs: dp.m_beta_plaw(r, **{k: v[sl] for k,v in
                                                             gas_args.items()})
    else:
        gas_args['r_y'] = r_flat * prms.r500c
        gas_args['gamma'] = gamma
        gas_args['rho_x'] = rho_500c
        m_gas = lambda r, sl, **kwargs: dp.m_beta_plaw_uni(r, **{k: v[sl] for k,v in
                                                                 gas_args.items()})

    return m_gas

# ----------------------------------------------------------------------
# End of m_gas_model()
# ----------------------------------------------------------------------

def m_stars_model(prms, m200m_dmo, r200m_dmo, c200m_dmo, f_c=0.86):
    '''
    Return the stellar mass profile
    '''
    # ##### #
    # STARS #
    # ##### #
    # the data assumed h=0.7, but resulting f_star is independent of h in our
    # model
    f_stars = d.f_stars(m200m_dmo / 0.7, 'all')
    f_cen = d.f_stars(m200m_dmo / 0.7, 'cen')
    f_sat = d.f_stars(m200m_dmo / 0.7, 'sat')
    c_sat = f_c * c200m_dmo

    cen_args = {'m_range': f_cen * m200m_dmo}
    sat_args = {'m_x': f_sat * m200m_dmo,
                'c_x': c_sat,
                'r_x': r200m_dmo}

    m_stars = lambda r, sl, **kwargs: (dp.m_delta(r, **{k: v[sl] for k,v in
                                                        cen_args.items()}) +
                                       dp.m_NFW(r, **{k: v[sl] for k,v in
                                                      sat_args.items()}))

    return m_stars

# ----------------------------------------------------------------------
# End of m_stars_model()
# ----------------------------------------------------------------------

def m_dm_model(prms, m200m_dmo, r200m_dmo, c200m_dmo,
               f_stars500c, f_gas500c):
    '''
    Return the dark matter mass profile
    '''
    # f_stars500c = m_stars(prms.r500c, np.arange(0,prms.r500c.shape[0])) / prms.m500c
    # f_gas500c = m_gas(prms.r500c, np.arange(0,prms.r500c.shape[0])) / prms.m500c
    m500c_dm = prms.m500c * (1 - f_gas500c - f_stars500c)
    c500c_dm = c200m_dmo * prms.r500c / r200m_dmo
    dm_args = {'m_x': m500c_dm,
               'c_x': c500c_dm,
               'r_x': prms.r500c}

    m_dm = lambda r, sl, **kwargs: dp.m_NFW(r, **{k: v[sl] for k,v in
                                                  dm_args.items()})

    return m_dm

# ----------------------------------------------------------------------
# End of m_dm_model()
# ----------------------------------------------------------------------

def r200m_from_m(m_f, cosmo, **kwargs):
    '''
    For a given cumulative mass profile m_f that takes the radius as its first
    argument, compute the radius where the mean enclosed density is 200 rho_m

    Parameters
    ----------
    m_f : function
      function to compute cumulative mass profile, radius is its first arg
    kwargs : dict
      arguments for m_f

    Returns
    -------
    r200m : float
      radius where mean enclosed density is 200 rho_m
    '''
    def diff_m200m(r):
        m200m = 4. /3 * np.pi * 200 * cosmo.rho_m * r**3
        m_diff = m_f(r, **kwargs) - m200m
        return m_diff

    r200m = opt.brentq(diff_m200m, 0.1, 100)
    return r200m

# ----------------------------------------------------------------------
# End of r200m_from_m()
# ----------------------------------------------------------------------

@np.vectorize
def gamma_from_m500c(m500c, cosmo, f_c=0.86, r_flat=None, q_f=50,
                     q_rc=50, q_beta=50, sigma_lnc=0., bias=False):
    '''
    Determine the maximum power law slope gamma for which

        m_b / m_tot <= f_b = cosmo.omegab / cosmo.omegam

    at r200m_obs

    Parameters
    ----------
    m500c : float or array
      halo mass
    cosmo : dict
      dictionary with cosmological parameters
    f_c : float
      ratio between the satellite and DMO halo concentration
    r_flat : float or None
      values where gas profile goes flat in units of r500c,
      if None, r200m_obs will be assumed
    q_f : int
      quantile for which to compute the f_gas,500c relation
    q_rc : int
      quantile for which to fit r_c
    q_beta : int
      quantile for which to fit beta
    bias : bool
      take into account hydrostatic bias

    Returns
    -------
    gamma : maximum allowed gamma for each m500c
    '''
    def fb_diff(gamma, m500c, cosmo, r_flat, q_f, q_rc, q_beta, bias,
                m_dm, m_stars):
        m_gas = m_gas_from_m500c(m500c, cosmo, gamma, r_flat, q_f, q_rc, q_beta, bias)
        f_b = cosmo.omegab / cosmo.omegam
        
        m_b = lambda r: m_stars(r) + m_gas(r)
        m_tot = lambda r: m_dm(r) + m_stars(r) + m_gas(r)
        r200m_obs = r200m_from_m(m_tot, cosmo)
        return (m_gas(r200m_obs) + m_stars(r200m_obs)) / m_tot(r200m_obs) - f_b

    # First, we need to calculate the mass profiles for the stars & DM
    # that don't change.
    #
    # Then we iterate over gamma for each m200m_dmo to determine which
    # values of gamma keep the total mass in the baryonic component
    # below the cosmic baryon fraction at their r200m_obs
    m_dm, m_stars = m_dm_stars_from_m500c(m500c=m500c, cosmo=cosmo, f_c=f_c,
                                          sigma_lnc=sigma_lnc)

    try:
        gamma = opt.brentq(fb_diff, 0, 1000, args=(m500c, cosmo, r_flat,
                                                   q_f, q_rc, q_beta, bias,
                                                   m_dm, m_stars))
    # if not solution is found, set boundary to gamma = 0
    except ValueError:
        gamma = 0.

    return gamma

# ----------------------------------------------------------------------
# End of gamma_from_m500c()
# ----------------------------------------------------------------------

def gamma_max(m500c, cosmo, f_c=0.86, r_flat=None, q_f=50, q_rc=50,
              q_beta=50, sigma_lnc=0., bias=False):
    '''
    Return interpolated m500c-gamma_max relation

    Parameters
    ----------
    m500c : float or array
      halo mass
    cosmo : dict
      dictionary with cosmological parameters
    f_c : float
      ratio between the satellite and DMO halo concentration
    r_flat : float or None
      values where gas profile goes flat in units of r500c,
      if None, r200m_obs will be assumed
    q_f : int
      quantile for which to compute the f_gas,500c relation
    q_rc : int
      quantile for which to fit r_c
    q_beta : int
      quantile for which to fit beta
    bias : bool
      take into account hydrostatic bias

    Returns
    -------
    gamma : maximum allowed gamma for each m500c
    '''
    logm_min = 8
    logm_max = 17

    bias_str = str(bias).replace(".", "p")
    slnc_str = str(sigma_lnc).replace(".", "p")
    fname = 'tables/m500c_gamma_max_r_flat_{}_qf_{}_qrc_{}_qb_{}_slnc_{}_bias_{}_table.npy'.format(r_flat,
                                                                                                   q_f, q_rc,
                                                                                                   q_beta,
                                                                                                   slnc_str,
                                                                                                   bias_str)
    interp_file = '/'.join(__file__.split('/')[:-1] + [fname])
    if os.path.isfile(interp_file):
        m_interp, gamma_max_interp = np.load(interp_file)

    else:
        m_interp = np.logspace(logm_min, logm_max, 201)
        gamma_max_interp = gamma_from_m500c(m500c=m_interp, cosmo=cosmo, f_c=f_c,
                                            r_flat=r_flat, q_f=q_f, q_rc=q_rc,
                                            q_beta=q_beta, sigma_lnc=sigma_lnc, bias=bias)
        np.save(interp_file, (m_interp, gamma_max_interp))

    gamma_max = intp.interp1d(m_interp, gamma_max_interp,
                              bounds_error=False,
                              fill_value=(0., np.nan))

    return gamma_max(m500c)

# ----------------------------------------------------------------------
# End of gamma_max()
# ----------------------------------------------------------------------

@np.vectorize
def m500c_to_m200m_dmo(m500c, r500c, f_gas500c, f_c, sigma_lnc, cosmo):
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
        c200m_dmo = tools.c_duffy(m200m_dmo, sigma_lnc=sigma_lnc).reshape(-1)
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

def m200m_dmo(m500c, r500c, f_gas500c, f_c, cosmo):
    '''
    Return interpolated m200m_dmo(m500c) relation

    Parameters
    ----------
    m500c : float or array
      halo mass
    cosmo : dict
      dictionary with cosmological parameters
    f_c : float
      ratio between the satellite and DMO halo concentration
    r_flat : float or None
      values where gas profile goes flat in units of r500c,
      if None, r200m_obs will be assumed
    q_f : int
      quantile for which to compute the f_gas,500c relation
    q_rc : int
      quantile for which to fit r_c
    q_beta : int
      quantile for which to fit beta

    Returns
    -------
    m200m_dmo : DMO equivalent halo mass for each halo
    '''
    pass

# @do_cprofile
def load_gamma(prms, r_max,
               gamma=np.array([2.]),
               r_flat=np.array([None]),
               f_c=0.86,
               q_f=50,
               q_rc=50,
               q_beta=50,
               sigma_lnc=0.,
               delta=False,
               bar2dmo=True,
               f_b=True,
               bias=False):
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
    r_flat : array
      values where gas profile goes flat in units of r500c,
      if None, r200m_obs will be assumed
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
    f_b : bool
      if True, profiles get extended after r_flat up until r_max or the cosmic
      baryon fraction is reached, if False, profiles get extended up to r200m_obs
    bias : bool
      include a hydrostatic bias correction
    Returns
    -------
    results : dict
      dictionary containing the power.Power objects corresponding to our models
    '''
    # We assume for now that our masses are the X-ray derived quantities
    # gas fractions
    # h=0.7 needs to be converted here
    f_prms = d.f_gas_prms(prms.cosmo, q=q_f, bias=bias)
    f_gas500 = d.f_gas(prms.m500c / 0.7, cosmo=prms.cosmo, **f_prms)
    rc, beta = d.fit_prms(x=500, q_rc=q_rc, q_beta=q_beta)

    # we compute the DMO equivalent halo mass
    # all models will have the same DMO equivalent halo mass
    m200m_dmo = m500c_to_m200m_dmo(m500c=prms.m500c, r500c=prms.r500c,
                                   f_gas500c=f_gas500, f_c=f_c, sigma_lnc=sigma_lnc,
                                   cosmo=prms.cosmo)
    r200m_dmo = tools.mass_to_radius(m200m_dmo, 200 * prms.cosmo.rho_m)
    c200m_dmo = tools.c_duffy(m200m_dmo, sigma_lnc=sigma_lnc).reshape(-1)

    # the data assumed h=0.7, but resulting f_star is independent of h in our
    # model
    f_stars = d.f_stars(m200m_dmo / 0.7, 'all')
    f_cen = d.f_stars(m200m_dmo / 0.7, 'cen')
    f_sat = d.f_stars(m200m_dmo / 0.7, 'sat')
    c_sat = f_c * c200m_dmo

    # compute density at r500c for the beta profile
    rho_500c = dp.profile_beta(prms.r500c.reshape(-1,1),
                               m_x=f_gas500 * prms.m500c,
                               r_x=prms.r500c,
                               rc=rc * prms.r500c,
                               beta=np.array([beta]*prms.r500c.shape[0])).reshape(-1)


    results = {'m500c': prms.m500c,
               'm200m_dmo': m200m_dmo,
               'r200m_dmo': r200m_dmo,
               'c200m_dmo': c200m_dmo,
               'f_stars': f_stars,
               'f_cen': f_cen,
               'f_sat': f_sat,
               'c_sat': c_sat}

    for idx_r, r_fl in enumerate(r_flat):
        results['{:d}'.format(idx_r)] = {}

        # need to have a flag to change r_fl back if we change
        # it to r200m_obs later
        r_fl_changed = False
        # First, we need to see whether our gamma values would not exceed f_b at
        # the resuling r200m_obs
        gamma_mx = gamma_max(m500c=prms.m500c, cosmo=prms.cosmo, f_c=f_c,
                             r_flat=r_fl, q_f=q_f, q_rc=q_rc, q_beta=q_beta,
                             sigma_lnc=sigma_lnc, bias=bias)
        # # We do not use the interpolation for now, since it goes wrong at
        # # the turnover between gamma = 0 and gamma > 0
        # gamma_mx = gamma_from_m500c(prms.m500c, prms.cosmo, f_c, r_fl,
        #                             q_f, q_rc, q_beta)

        for idx_g, g in enumerate(gamma):
            g = np.where(gamma_mx > g, gamma_mx, g)
            # First get the enclosed mass profiles to determine r200m
            m_stars = m_stars_model(prms=prms,
                                    m200m_dmo=m200m_dmo,
                                    r200m_dmo=r200m_dmo,
                                    c200m_dmo=c200m_dmo,
                                    f_c=f_c)

            m_gas_200m = m_gas_model(prms=prms,
                                     q_f=q_f,
                                     q_rc=q_rc,
                                     q_beta=q_beta,
                                     gamma=g,
                                     r_flat=r_fl,
                                     bias=bias)

            f_stars500c = (m_stars(prms.r500c, np.arange(0, prms.r500c.shape[0])) /
                           prms.m500c)
            f_gas500c = (m_gas_200m(prms.r500c, np.arange(0, prms.r500c.shape[0])) /
                         prms.m500c)

            m_dm = m_dm_model(prms=prms,
                              m200m_dmo=m200m_dmo,
                              r200m_dmo=r200m_dmo,
                              c200m_dmo=c200m_dmo,
                              f_stars500c=f_stars500c,
                              f_gas500c=f_gas500c)

            m_b_200m = lambda r, sl, **kwargs: m_stars(r, sl) + m_gas_200m(r, sl)
            m_tot_200m = lambda r, sl, **kwargs: (m_dm(r, sl) + m_stars(r, sl)
                                                  + m_gas_200m(r, sl))
            # get r200m_obs
            r200m_obs = np.array([r200m_from_m(m_tot_200m, prms.cosmo, sl=i)
                                  for i in np.arange(0, prms.r500c.shape[0])])
            m200m_obs = tools.radius_to_mass(r200m_obs, 200 * prms.cosmo.rho_m)

            # now we need to correct the gas, baryonic and total masses in case
            # r_flat is None, since then we will MAKE r_flat = r200m_obs
            if r_fl is None:
                r_fl = r200m_obs / prms.r500c
                r_fl_changed = True

            m_gas_fb = m_gas_model(prms=prms,
                                   q_f=q_f,
                                   q_rc=q_rc,
                                   q_beta=q_beta,
                                   gamma=g,
                                   r_flat=r_fl,
                                   bias=bias)
            m_b_fb = lambda r, sl, **kwargs: m_stars(r, sl) + m_gas_fb(r, sl)
            m_tot_fb = lambda r, sl, **kwargs: (m_dm(r, sl) + m_stars(r, sl)
                                                + m_gas_fb(r, sl))

            if f_b:
                # put r_max to radius where f_b is reached
                fb = lambda r, sl, **kwargs: m_b_fb(r, sl) / m_tot_fb(r, sl)

                # now we determine radius at which plaw + uniform gives fb_universal
                r_max_fb = np.array([dp.r_fb_from_f(fb, cosmo=prms.cosmo, sl=sl,
                                                    r500c=r)
                                     for sl, r in enumerate(prms.r500c)])

                # sometimes, there are multiple values where f_b is reached. We
                # know that r200m_obs will also have f_b, since we determined
                # gamma_max this way so if r_max < r200m_obs, force r_max = r_200m_obs
                r_max_fb = np.where(r_max_fb < r200m_obs, r200m_obs, r_max_fb)

                # not all profiles will be baryonically closed, cut off at r_max
                r_max_fb[np.isinf(r_max_fb)] = r_max[np.isinf(r_max_fb)]

            else:
                # let's put r_max to r200m_obs
                r_max_fb = r200m_obs

            # load stars
            if not delta:
                cen_rmax = load_centrals_rmax(prms=prms,
                                              r_max=r_max_fb,
                                              m200m_dmo=m200m_dmo,
                                              f_comp='cen')
                sat_rmax = load_satellites_rmax(prms=prms,
                                                r_max=r_max_fb,
                                                m200m_dmo=m200m_dmo,
                                                r200m_dmo=r200m_dmo,
                                                c200m_dmo=c200m_dmo,
                                                f_c=f_c)
                stars_rmax = cen_rmax + sat_rmax

            else:
                stars_rmax = load_centrals_rmax(prms=prms,
                                                r_max=r_max_fb,
                                                m200m_dmo=m200m_dmo,
                                                f_comp='all')

            # load gas
            gas_plaw_r500c_rmax = load_gas_plaw_r500c_rmax(prms=prms,
                                                           rho_500c=rho_500c,
                                                           r_flat=r_fl,
                                                           r_max=r_max_fb,
                                                           gamma=g,
                                                           rc=rc, beta=beta,
                                                           f_gas500=f_gas500,
                                                           q_f=q_f, q_rc=q_rc,
                                                           q_beta=q_beta)

            # now compute DM parameters
            m_stars500c = stars_rmax.m_r(prms.r500c)
            m_gas500c = gas_plaw_r500c_rmax.m_r(prms.r500c)
            m500c_dm = prms.m500c - m_gas500c - m_stars500c
            c500c_dm = c200m_dmo * prms.r500c / r200m_dmo

            dm_plaw_r500c_rmax = load_dm_rmax(prms=prms,
                                              r_max=r_max_fb,
                                              m500c=m500c_dm,
                                              r500c=prms.r500c,
                                              c500c=c500c_dm)

            prof_tot = dm_plaw_r500c_rmax + gas_plaw_r500c_rmax + stars_rmax
            m_obs_200m_dmo = prof_tot.m_r(r200m_dmo)

            pow_gas_plaw_r500c_rmax = power.Power(m200m_dmo=m200m_dmo,
                                                  m200m_obs=m200m_obs,
                                                  prof=prof_tot,
                                                  bar2dmo=bar2dmo)

            # now load dmo profiles
            dm_dmo_rmax = load_dm_dmo_rmax(prms=prms,
                                           r_max=r_max_fb,
                                           m200m_dmo=m200m_dmo,
                                           r200m_dmo=r200m_dmo,
                                           c200m_dmo=c200m_dmo)

            pow_dm_dmo_rmax = power.Power(m200m_dmo=m200m_dmo,
                                          m200m_obs=m200m_dmo,
                                          prof=dm_dmo_rmax,
                                          bar2dmo=False)

            temp = {'pow': pow_gas_plaw_r500c_rmax,
                    'pow_dmo': pow_dm_dmo_rmax,
                    'd_gas': gas_plaw_r500c_rmax,
                    'd_dm': dm_plaw_r500c_rmax,
                    'd_stars': stars_rmax,
                    'd_tot': prof_tot,
                    'd_dmo': dm_dmo_rmax,
                    'gamma_max': gamma_mx,
                    'gamma': g,
                    'm200m_obs': m200m_obs,
                    'm_obs_200m_dmo': m_obs_200m_dmo,
                    'r200m_obs': r200m_obs,
                    'r_max': r_max_fb,
                    'r_flat': r_fl}
            results['{:d}'.format(idx_r)]['{:d}'.format(idx_g)] = temp

            if r_fl_changed is True:
                r_fl = r_flat[idx_r]

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
