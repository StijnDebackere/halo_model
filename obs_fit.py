import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

import halo.parameters as p
import halo.tools as tools
import halo.density_profiles as dp
import halo.model.density as dens
import halo.model.power as power
import halo.input.interpolators as inp_interp

import pdb
import cProfile

plt.style.use("paper")


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


def load_dm_rmax(prms, r_max, m500c_dm, r500c, c500c_dm):
    '''
    Return NFW profiles up to r_max
    '''
    profile_args = {'m_x': m500c_dm,
                    'c_x': c500c_dm,
                    'r_x': r500c}

    profile_mass = lambda r, **kwargs: dp.m_NFW(r, **profile_args)

    m_h = profile_mass(r_max)
    r_h = r_max
    c_h = c500c_dm * r_h / r500c
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


def load_gas_plaw_r500c_rmax(prms, m500c, r500c, rho_500c, r_flat, r_max,
                             gamma, r_c=None, beta=None, fgas_500c=None):
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
                                        m_x=fgas_500c * m500c,
                                        r_x=r500c,
                                        r_c=r_c * r500c,
                                        beta=np.array([beta] * r500c.shape[0]),
                                        r_y=r_flat * r500c,
                                        gamma=gamma,
                                        rho_x=rho_500c)

    profile_args =  {'m_x': fgas_500c * m500c,
                     'r_x': r500c,
                     'r_c': r_c * r500c,
                     'beta': np.array([beta] * r500c.shape[0]),
                     'r_y': r_flat * r500c,
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


def load_centrals_rmax(prms, r_max, m500c, fcen_500c):
    '''
    Return delta profiles with fstars_500c = f_obs

    '''
    profile_args = {'m_x': fcen_500c * m500c}
    cen_kwargs = {'cosmo': prms.cosmo,
                  'r_h': r_max,
                  'profile': dp.profile_delta,
                  'profile_args': profile_args,
                  'profile_mass': lambda r, **kwargs: dp.m_delta(r, **profile_args),
                  'profile_f': dp.profile_delta_f,
                  'profile_f_args': profile_args}

    dens_cen = dens.Profile(**cen_kwargs)
    return dens_cen


def load_satellites_rmax(prms, r_max, m500c, r500c, csat_500c, fsat_500c):
    '''
    Return NFW profiles with fstars_500c = f_obs

    Parameters
    ----------
    prms : p.Parameters object
      contains relevant model info
    '''
    profile_args = {'m_x': fsat_500c * m500c,
                    'c_x': csat_500c,
                    'r_x': r500c}

    profile_mass = lambda r, **kwargs: dp.m_NFW(r, **profile_args)

    m_h = profile_mass(r_max)
    r_h = r_max
    c_h = csat_500c * r_h / r500c
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


def m_gas_model(m500c_gas, r500c, r_c, beta, gamma,
                z=0.,
                r_flat=None,
                r_plaw=None,
                delta=None,
                rho_500c=None):
    '''
    Return the gas mass profile

    Parameters
    ----------
    m500c_gas : (m, ) array
      gas mass inside r500c
    r500c : (m, ) array
      radius for gas mass
    r_c : (m, ) array
      physical core radius r_c of the profile
    beta : (m, ) array
      beta slope of the profile
    gamma : (m, ) array
      power law slope of profile outside r500c
    r_flat : None or float in units of r500c
      radius at which profile becomes uniform
    r_plaw : (m, ) array or None
      radius at which delta slope sets in
    delta : (m, ) array
      slope outside r_plaw
    rho_500c : (m, ) array, optional
      gas density at r500c

    Returns
    -------
    m_gas : lambda function to compute gas masses
    '''
    if rho_500c is None:
        rho_500c = dp.profile_beta(r500c.reshape(-1,1),
                                   m_x=m500c_gas,
                                   r_x=r500c,
                                   r_c=r_c,
                                   beta=np.array([beta]*r500c.shape[0])).reshape(-1)

    gas_args =  {'m_x': m500c_gas,
                 'r_x': r500c,
                 'r_c': r_c,
                 'beta': beta}

    if r_flat is None:
        gas_args['gamma'] = gamma
        gas_args['rho_x'] = rho_500c
        if r_plaw is None:
            m_gas = lambda r, sl, **kwargs: dp.m_beta_plaw(r, **{k: v[sl]
                                                                 for k, v in
                                                                 gas_args.items()})
        else:
            gas_args['r_y'] = r_plaw
            gas_args['delta'] = np.array([delta] * r500c.shape[0])
            m_gas = lambda r, sl, **kwargs: dp.m_beta_gamma_plaw(r, **{k: v[sl]
                                                                       for k, v in
                                                                       gas_args.items()})
    else:
        gas_args['r_y'] = r_flat * r500c
        gas_args['gamma'] = gamma
        gas_args['rho_x'] = rho_500c
        m_gas = lambda r, sl, **kwargs: dp.m_beta_plaw_uni(r, **{k: v[sl] for k, v in
                                                                 gas_args.items()})

    return m_gas


def m_stars_model(m500c, r500c, csat_500c, fcen_500c, fsat_500c):
    '''
    Return the stellar mass profile
    '''
    cen_args = {'m_x': fcen_500c * m500c}
    sat_args = {'m_x': fsat_500c * m500c,
                'c_x': csat_500c,
                'r_x': r500c}

    m_stars = lambda r, sl, **kwargs: (dp.m_delta(r, **{k: v[sl] for k, v in
                                                        cen_args.items()}) +
                                       dp.m_NFW(r, **{k: v[sl] for k, v in
                                                      sat_args.items()}))

    return m_stars


def m_dm_model(m500c_dm, c500c_dm, r500c):
    '''
    Return the dark matter mass profile
    '''
    dm_args = {'m_x': m500c_dm,
               'c_x': c500c_dm,
               'r_x': r500c}

    m_dm = lambda r, sl, **kwargs: dp.m_NFW(r, **{k: v[sl] for k, v in
                                                  dm_args.items()})

    return m_dm


def r200m_from_m(m_f, r200m_dmo, cosmo, **kwargs):
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
        m200m = 4. / 3 * np.pi * 200 * cosmo.rho_m * r**3
        m_diff = m_f(r, **kwargs) - m200m
        return m_diff

    r200m = opt.brentq(diff_m200m, 0.2 * r200m_dmo, 2 * r200m_dmo)
    return r200m


def r200c_from_m(m_f, r200c_dmo, cosmo, **kwargs):
    '''
    For a given cumulative mass profile m_f that takes the radius as its
    first argument, compute the radius where the mean enclosed density
    is 200 rho_crit

    Parameters
    ----------
    m_f : function
      function to compute cumulative mass profile, radius is its first arg
    kwargs : dict
      arguments for m_f

    Returns
    -------
    r200c : float
      radius where mean enclosed density is 200 rho_crit
    '''
    def diff_m200c(r):
        m200c = 4. / 3 * np.pi * 200 * cosmo.rho_crit * r**3
        m_diff = m_f(r, **kwargs) - m200c
        return m_diff

    r200c = opt.brentq(diff_m200c, 0.2 * r200c_dmo, 2 * r200c_dmo)
    return r200c


def r_fb_from_f(f_b, cosmo, r500c, r_max, **kwargs):
    '''
    For a given cumulative mass profile m_f that takes the radius as its first
    argument, compute the radius where the mean enclosed density is 200 rho_m

    Parameters
    ----------
    m_f : function
      function to compute cumulative mass profile, radius is its first arg
    cosmo : cosmo.Cosmology object
      cosmology information
    r500c : float
      radius to start search at
    r_max : float
      radius to finish search at
    kwargs : dict
      arguments for m_f

    Returns
    -------
    r200m : float
      radius where mean enclosed density is 200 rho_m
    '''
    def diff_fb(r):
        fb = cosmo.omegab / cosmo.omegam
        f_diff = f_b(r, **kwargs) - fb
        return f_diff

    try:
        r_fb = opt.brentq(diff_fb, r500c, r_max)
    except ValueError:
        r_fb = np.inf

    return r_fb


def load_interp_prms(prms=p.prms):
    """
    Loading the interpolated parameters into a dict to have them in memory
    """
    prms_interp = {}
    # these ones already assume sigma_lnc in prms_interp
    prms_interp["m200m_dmo"] = prms.m200m_dmo
    prms_interp["r200m_dmo"] = prms.r200m_dmo
    prms_interp["c200m_dmo"] = prms.c200m_dmo

    # load gas prms_interp
    prms_interp["fgas_500c"] = prms.fgas_500c

    # load stellar prms_interp
    prms_interp["fcen_500c"] = prms.fcen_500c
    prms_interp["fsat_500c"] = prms.fsat_500c
    prms_interp["fstar_500c"] = prms_interp["fcen_500c"] + prms_interp["fsat_500c"]

    # get fit prms_interp for the satellite density profiles
    prms_interp["csat_500c"] = (prms.f_c * prms_interp["c200m_dmo"] *
                                prms.r500c / prms_interp["r200m_dmo"])
    prms_interp["c500c_dm"] = (prms_interp["c200m_dmo"] *
                               prms.r500c / prms_interp["r200m_dmo"])

    prms_interp["m500c_dm"] = prms.m500c * (1 - prms_interp["fgas_500c"] -
                                            prms_interp["fstar_500c"])
    prms_interp["m500c_dmo"] = prms_interp["m500c_dm"] / (1 - prms.f_bar)

    return prms_interp


# @do_cprofile
def load_gamma(prms=p.prms,
               prms_interp=None,
               r_max=20*p.prms.r500c,
               r_c=0.21,
               beta=0.71,
               gamma=np.array([2.]),
               r_flat=np.array([None]),
               # delta=False,
               bar2dmo=True,
               f_b=True,
               comps=False):
    '''
    Load all of our different models, the ones upto r200m and the ones upto
    r_max.

    Parameters
    ----------
    prms : p.Parameters object
      model parameters
    r_max : array
      maximum radius to extend our profiles up to for each mass
    gamma : array
      values of the slope to compute for
    r_flat : array
      values where gas profile goes flat in units of r500c,
      if None, r200m_obs will be assumed
    q_rc : int
      quantile for which to fit r_c
    q_beta : int
      quantile for which to fit beta
    delta : bool
      assume delta profile for the total stellar contribution
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
    if prms_interp is None:
        prms_interp = load_interp_prms(prms)

    # load in the basic model parameters
    z_range = prms.z_range
    if np.size(z_range) > 1:
        raise ValueError("redshift dependence not yet implemented")
    m500c = prms.m500c
    r500c = prms.r500c
    # these ones already assume sigma_lnc in parameters
    m200m_dmo = prms_interp["m200m_dmo"]
    r200m_dmo = prms_interp["r200m_dmo"]
    c200m_dmo = prms_interp["c200m_dmo"]

    # load gas parameters
    fgas_500c = prms_interp["fgas_500c"]
    f_prms = prms.fgas_500c_prms

    # compute density at r500c for the beta profile
    rho_500c = dp.profile_beta(prms.r500c.reshape(-1, 1),
                               m_x=fgas_500c * prms.m500c,
                               r_x=prms.r500c,
                               r_c=r_c * prms.r500c,
                               beta=np.array([beta]*prms.r500c.shape[0])).reshape(-1)

    # load stellar parameters, these have already been calculated for m500c
    # from m200m_dmo
    fcen_500c = prms_interp["fcen_500c"]
    fsat_500c = prms_interp["fsat_500c"]
    fstar_500c = prms_interp["fstar_500c"]
    f_c = prms.f_c
    sigma_lnc = prms.sigma_lnc

    # get fit parameters for the satellite density profiles
    csat_500c = prms_interp["csat_500c"]
    c500c_dm = prms_interp["c500c_dm"]

    m500c_dm = prms_interp["m500c_dm"]
    m500c_dmo = prms_interp["m500c_dmo"]

    # get r200c_dmo
    m_dmo = m_dm_model(m500c_dm=m500c_dmo,
                       c500c_dm=c500c_dm,
                       r500c=r500c)
    r200c_dmo = np.array([r200c_from_m(m_f=m_dmo,
                                       r200c_dmo=r200m_dmo[i],
                                       cosmo=prms.cosmo,
                                       sl=i)
                          for i in np.arange(0, r500c.shape[0])])

    m200c_dmo = tools.radius_to_mass(r200c_dmo, 200 * prms.cosmo.rho_crit)

    # add basic model parameters to the final dict
    results = {'prms': prms,
               'm500c': prms.m500c,
               'm200m_dmo': m200m_dmo,
               'r200m_dmo': r200m_dmo,
               'c200m_dmo': c200m_dmo,
               'm200c_dmo': m200c_dmo,
               'r200c_dmo': r200c_dmo,
               'fstar_500c': fstar_500c,
               'fcen_500c': fcen_500c,
               'fsat_500c': fsat_500c,
               'csat_500c': csat_500c,
               'c500c_dm': c500c_dm,
               'm500c_dm': m500c_dm,
               'm500c_dmo': m500c_dmo}

    # we can load the enclosed mass profiles for the DM and stars here,
    # since their parameters do not depend on gamma or r_flat
    m_stars = m_stars_model(m500c=m500c,
                            r500c=r500c,
                            csat_500c=csat_500c,
                            fcen_500c=fcen_500c,
                            fsat_500c=fsat_500c)

    m_dm = m_dm_model(m500c_dm=m500c_dm,
                      c500c_dm=c500c_dm,
                      r500c=r500c)

    for idx_r, r_fl in enumerate(r_flat):
        results['{:d}'.format(idx_r)] = {}

        # need to have a flag to change r_fl back if we change
        # it to r200m_obs later
        r_fl_changed = False
        # First, we need to see whether our gamma values would not exceed f_b
        # at the resuling r200m_obs
        gamma_mx = inp_interp.gamma_max_interp(f_c=f_c, sigma_lnc=sigma_lnc,
                                               r_c=r_c, beta=beta, r_flat=r_fl)

        # fgas_500c is along same dimension as m500c
        z_c, m500c_c = inp_interp.arrays_to_ogrid(z_range, np.log10(m500c))
        fgas_500c_c = fgas_500c.reshape(m500c_c.shape)
        coords = (z_c, m500c_c, fgas_500c_c)
        gamma_max = gamma_mx(coords).reshape(np.shape(z_range) +
                                             np.shape(m500c))

        for idx_g, g in enumerate(gamma):
            g = np.where(gamma_max > g, gamma_max, g)

            # First get the enclosed mass profiles to determine r200m
            m_gas_200m = m_gas_model(m500c_gas=fgas_500c * m500c,
                                     r500c=r500c,
                                     r_c=r_c * r500c,
                                     beta=np.array([beta] * r500c.shape[0]),
                                     gamma=g,
                                     r_flat=r_fl,
                                     z=z_range,
                                     rho_500c=rho_500c)

            m_b_200m = lambda r, sl, **kwargs: m_stars(r, sl) + m_gas_200m(r, sl)
            m_tot_200m = lambda r, sl, **kwargs: (m_dm(r, sl) + m_stars(r, sl)
                                                  + m_gas_200m(r, sl))
            # get r200m_obs
            r200m_obs = np.array([r200m_from_m(m_f=m_tot_200m,
                                               r200m_dmo=r200m_dmo[i],
                                               cosmo=prms.cosmo,
                                               sl=i)
                                  for i in np.arange(0, r500c.shape[0])])

            m200m_obs = tools.radius_to_mass(r200m_obs, 200 * prms.cosmo.rho_m)

            # get r200c_obs
            r200c_obs = np.array([r200c_from_m(m_f=m_tot_200m,
                                               r200c_dmo=r200c_dmo[i],
                                               cosmo=prms.cosmo,
                                               sl=i)
                                  for i in np.arange(0, r500c.shape[0])])

            m200c_obs = tools.radius_to_mass(r200c_obs, 200 * prms.cosmo.rho_crit)

            # # get r2500c_obs
            # r2500c_obs = np.array([tools.rx_from_m(m_f=m_tot_200m,
            #                                        rho_x=2500 * prms.rho_crit,
            #                                        sl=i)
            #                       for i in np.arange(0, r500c.shape[0])])

            # m2500c_obs = tools.radius_to_mass(r2500c_obs, 2500 * prms.cosmo.rho_crit)

            # now we need to correct the gas, baryonic and total masses in case
            # r_flat is None, since then we will MAKE r_flat = r200m_obs
            if r_fl is None:
                r_fl = r200m_obs / r500c
                r_fl_changed = True

            m_gas_fb = m_gas_model(m500c_gas=fgas_500c * m500c,
                                   r500c=r500c,
                                   r_c=r_c * r500c,
                                   beta=np.array([beta] * r500c.shape[0]),
                                   gamma=g,
                                   r_flat=r_fl,
                                   z=z_range,
                                   rho_500c=rho_500c)

            m_b_fb = lambda r, sl, **kwargs: m_stars(r, sl) + m_gas_fb(r, sl)
            m_tot_fb = lambda r, sl, **kwargs: (m_dm(r, sl) + m_stars(r, sl)
                                                + m_gas_fb(r, sl))

            if f_b:
                # put r_max to radius where f_b is reached
                fb = lambda r, sl, **kwargs: m_b_fb(r, sl, **kwargs) / m_tot_fb(r, sl, **kwargs)

                # now we determine radius at which plaw + uniform gives fb_universal
                r_max_fb = np.array([r_fb_from_f(fb, cosmo=prms.cosmo, sl=sl,
                                                 r500c=r, r_max=r_max[sl])
                                     for sl, r in enumerate(r500c)])

                # sometimes, there are multiple values where f_b is reached. We
                # know that r200m_obs will also have f_b, since we determined
                # gamma_max this way so if r_max < r200m_obs, force r_max = r_200m_obs
                r_max_fb = np.where(r_max_fb < r200m_obs, r200m_obs, r_max_fb)

                # not all profiles will be baryonically closed, cut off at r_max
                r_max_fb[np.isinf(r_max_fb)] = r_max[np.isinf(r_max_fb)]

            else:
                # let's put r_max to r200m_obs
                r_max_fb = r200m_obs

            # load dm
            dm_plaw_r500c_rmax = load_dm_rmax(prms=prms,
                                              r_max=r_max_fb,
                                              m500c_dm=m500c_dm,
                                              r500c=r500c,
                                              c500c_dm=c500c_dm)

            # load stars
            cen_rmax = load_centrals_rmax(prms=prms,
                                          r_max=r_max_fb,
                                          m500c=m500c,
                                          fcen_500c=fcen_500c)
            sat_rmax = load_satellites_rmax(prms=prms,
                                            r_max=r_max_fb,
                                            m500c=m500c,
                                            r500c=r500c,
                                            csat_500c=csat_500c,
                                            fsat_500c=fsat_500c)
            stars_rmax = cen_rmax + sat_rmax

            # load gas
            gas_plaw_r500c_rmax = load_gas_plaw_r500c_rmax(prms=prms,
                                                           m500c=m500c,
                                                           r500c=r500c,
                                                           rho_500c=rho_500c,
                                                           r_flat=r_fl,
                                                           r_max=r_max_fb,
                                                           gamma=g,
                                                           r_c=r_c, beta=beta,
                                                           fgas_500c=fgas_500c)

            prof_tot = dm_plaw_r500c_rmax + gas_plaw_r500c_rmax + stars_rmax
            m_obs_200m_dmo = prof_tot.m_r(r200m_dmo)

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

            if not comps:
                pow_gas_plaw_r500c_rmax = power.Power(m200m_dmo=m200m_dmo,
                                                      m200m_obs=m200m_obs,
                                                      prof=prof_tot,
                                                      bar2dmo=bar2dmo)

            else:
                profiles = [dm_plaw_r500c_rmax, gas_plaw_r500c_rmax, stars_rmax]
                names    = ["dm", "gas", "stars"]
                pow_gas_plaw_r500c_rmax = power.Power_Components(m200m_dmo=m200m_dmo,
                                                                 m200m_obs=m200m_obs,
                                                                 cosmo=prms.cosmo,
                                                                 k_range=prms.k_range,
                                                                 profiles=profiles,
                                                                 names=names,
                                                                 bar2dmo=bar2dmo)
                
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
                    'm200c_obs': m200c_obs,
                    'r200c_obs': r200c_obs,
                    # 'm2500c_obs': m2500c_obs,
                    # 'r2500c_obs': r2500c_obs,
                    'r_max': r_max_fb,
                    'r_flat': r_fl}
            results['{:d}'.format(idx_r)]['{:d}'.format(idx_g)] = temp

            # if r_flat was changed to r200m_obs, put it back to its original value
            # for the loop
            if r_fl_changed is True:
                r_fl = r_flat[idx_r]

    return results


# @do_cprofile
def load_gamma_profiles(prms=p.prms,
                        prms_interp=None,
                        r_max=20*p.prms.r500c,
                        r_c=0.21,
                        beta=0.71,
                        gamma=np.array([2.]),
                        bar2dmo=True,
                        f_b=True):
    '''
    Load all of our different models, the ones upto r_max and the ones upto
    r_max.

    Parameters
    ----------
    prms : p.Parameters object
      model parameters
    r_max : array
      maximum radius to extend our profiles up to for each mass
    gamma : array
      values of the slope to compute for
    r_flat : array
      values where gas profile goes flat in units of r500c,
      if None, r200m_obs will be assumed
    q_rc : int
      quantile for which to fit r_c
    q_beta : int
      quantile for which to fit beta
    delta : bool
      assume delta profile for the total stellar contribution
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
    if prms_interp is None:
        prms_interp = load_interp_prms(prms)

    # load in the basic model parameters
    z_range = prms.z_range
    if np.size(z_range) > 1:
        raise ValueError("redshift dependence not yet implemented")
    m500c = prms.m500c
    r500c = prms.r500c
    # these ones already assume sigma_lnc in parameters
    m200m_dmo = prms_interp["m200m_dmo"]
    r200m_dmo = prms_interp["r200m_dmo"]
    c200m_dmo = prms_interp["c200m_dmo"]

    # load gas parameters
    fgas_500c = prms_interp["fgas_500c"]
    f_prms = prms.fgas_500c_prms

    # compute density at r500c for the beta profile
    rho_500c = dp.profile_beta(prms.r500c.reshape(-1, 1),
                               m_x=fgas_500c * prms.m500c,
                               r_x=prms.r500c,
                               r_c=r_c * prms.r500c,
                               beta=np.array([beta]*prms.r500c.shape[0])).reshape(-1)

    # load stellar parameters, these have already been calculated for m500c
    # from m200m_dmo
    fcen_500c = prms_interp["fcen_500c"]
    fsat_500c = prms_interp["fsat_500c"]
    fstar_500c = prms_interp["fstar_500c"]
    f_c = prms.f_c
    sigma_lnc = prms.sigma_lnc

    # get fit parameters for the satellite density profiles
    csat_500c = prms_interp["csat_500c"]
    c500c_dm = prms_interp["c500c_dm"]

    m500c_dm = prms_interp["m500c_dm"]
    m500c_dmo = prms_interp["m500c_dmo"]

    # get r200c_dmo
    m_dmo = m_dm_model(m500c_dm=m500c_dmo,
                       c500c_dm=c500c_dm,
                       r500c=r500c)
    r200c_dmo = np.array([r200c_from_m(m_f=m_dmo,
                                       r200c_dmo=r200m_dmo[i],
                                       cosmo=prms.cosmo,
                                       sl=i)
                          for i in np.arange(0, r500c.shape[0])])

    m200c_dmo = tools.radius_to_mass(r200c_dmo, 200 * prms.cosmo.rho_crit)

    # we can load the enclosed mass profiles for the DM and stars here,
    # since their parameters do not depend on gamma or r_flat
    m_stars = m_stars_model(m500c=m500c,
                            r500c=r500c,
                            csat_500c=csat_500c,
                            fcen_500c=fcen_500c,
                            fsat_500c=fsat_500c)

    m_dm = m_dm_model(m500c_dm=m500c_dm,
                      c500c_dm=c500c_dm,
                      r500c=r500c)

    results = {}

    # a priori, profiles do not need flat extension
    r_flat = None
    # First, we need to see whether our gamma values would not exceed f_b
    # at the resuling r200m_obs
    gamma_mx = inp_interp.gamma_max_interp(f_c=f_c, sigma_lnc=sigma_lnc,
                                           r_c=r_c, beta=beta, r_flat=r_flat)

    # fgas_500c is along same dimension as m500c
    z_c, m500c_c = inp_interp.arrays_to_ogrid(z_range, np.log10(m500c))
    fgas_500c_c = fgas_500c.reshape(m500c_c.shape)
    coords = (z_c, m500c_c, fgas_500c_c)
    gamma_max = gamma_mx(coords).reshape(np.shape(z_range) +
                                         np.shape(m500c))

    # add basic model parameters to the final dict
    results = {'prms': prms,
               'm500c': prms.m500c,
               'm200m_dmo': m200m_dmo,
               'r200m_dmo': r200m_dmo,
               'c200m_dmo': c200m_dmo,
               'm200c_dmo': m200c_dmo,
               'r200c_dmo': r200c_dmo,
               'fstar_500c': fstar_500c,
               'fcen_500c': fcen_500c,
               'fsat_500c': fsat_500c,
               'csat_500c': csat_500c,
               'c500c_dm': c500c_dm,
               'm500c_dm': m500c_dm,
               'm500c_dmo': m500c_dmo,
               'gamma_max': gamma_max}

    for idx_g, g in enumerate(gamma):
        g = np.where(gamma_max > g, gamma_max, g)

        # First get the enclosed mass profiles to determine r200m
        m_gas_200m = m_gas_model(m500c_gas=fgas_500c * m500c,
                                 r500c=r500c,
                                 r_c=r_c * r500c,
                                 beta=np.array([beta] * r500c.shape[0]),
                                 gamma=g,
                                 r_flat=r_flat,
                                 z=z_range,
                                 rho_500c=rho_500c)

        m_b_200m = lambda r, sl, **kwargs: m_stars(r, sl) + m_gas_200m(r, sl)
        m_tot_200m = lambda r, sl, **kwargs: (m_dm(r, sl) + m_stars(r, sl)
                                              + m_gas_200m(r, sl))
        # get r200m_obs
        r200m_obs = np.array([r200m_from_m(m_f=m_tot_200m,
                                           r200m_dmo=r200m_dmo[i],
                                           cosmo=prms.cosmo,
                                           sl=i)
                              for i in np.arange(0, r500c.shape[0])])

        m200m_obs = tools.radius_to_mass(r200m_obs, 200 * prms.cosmo.rho_m)

        # get r200c_obs
        r200c_obs = np.array([r200c_from_m(m_f=m_tot_200m,
                                           r200c_dmo=r200c_dmo[i],
                                           cosmo=prms.cosmo,
                                           sl=i)
                              for i in np.arange(0, r500c.shape[0])])

        m200c_obs = tools.radius_to_mass(r200c_obs, 200 * prms.cosmo.rho_crit)

        # # get r2500c_obs
        # r2500c_obs = np.array([tools.rx_from_m(m_f=m_tot_200m,
        #                                        rho_x=2500 * prms.rho_crit,
        #                                        sl=i)
        #                       for i in np.arange(0, r500c.shape[0])])

        # m2500c_obs = tools.radius_to_mass(r2500c_obs, 2500 * prms.cosmo.rho_crit)

        # now we need to correct the gas, baryonic and total masses in case
        # r_flat is None, since then we will MAKE r_flat = r200m_obs
        if r_flat is None and f_b:
            r_flat = r200m_obs / r500c

        m_gas_fb = m_gas_model(m500c_gas=fgas_500c * m500c,
                               r500c=r500c,
                               r_c=r_c * r500c,
                               beta=np.array([beta] * r500c.shape[0]),
                               gamma=g,
                               r_flat=r_flat,
                               z=z_range,
                               rho_500c=rho_500c)

        m_b_fb = lambda r, sl, **kwargs: m_stars(r, sl) + m_gas_fb(r, sl)
        m_tot_fb = lambda r, sl, **kwargs: (m_dm(r, sl) + m_stars(r, sl)
                                            + m_gas_fb(r, sl))

        if f_b:
            # put r_max to radius where f_b is reached
            fb = lambda r, sl, **kwargs: m_b_fb(r, sl, **kwargs) / m_tot_fb(r, sl, **kwargs)

            # now we determine radius at which plaw + uniform gives fb_universal
            r_max_fb = np.array([r_fb_from_f(fb, cosmo=prms.cosmo, sl=sl,
                                             r500c=r, r_max=r_max[sl])
                                 for sl, r in enumerate(r500c)])

            # sometimes, there are multiple values where f_b is reached. We
            # know that r200m_obs will also have f_b, since we determined
            # gamma_max this way so if r_max < r200m_obs, force r_max = r_200m_obs
            r_max_fb = np.where(r_max_fb < r200m_obs, r200m_obs, r_max_fb)

            # not all profiles will be baryonically closed, cut off at r_max
            r_max_fb[np.isinf(r_max_fb)] = r_max[np.isinf(r_max_fb)]

        else:
            # let's put r_max to r_max
            r_flat = r_max
            r_max_fb = r_max

        # load dm
        dm_plaw_r500c_rmax = load_dm_rmax(prms=prms,
                                          r_max=r_max_fb,
                                          m500c_dm=m500c_dm,
                                          r500c=r500c,
                                          c500c_dm=c500c_dm)

        # load stars
        cen_rmax = load_centrals_rmax(prms=prms,
                                      r_max=r_max_fb,
                                      m500c=m500c,
                                      fcen_500c=fcen_500c)
        sat_rmax = load_satellites_rmax(prms=prms,
                                        r_max=r_max_fb,
                                        m500c=m500c,
                                        r500c=r500c,
                                        csat_500c=csat_500c,
                                        fsat_500c=fsat_500c)
        stars_rmax = cen_rmax + sat_rmax

        # load gas
        gas_plaw_r500c_rmax = load_gas_plaw_r500c_rmax(prms=prms,
                                                       m500c=m500c,
                                                       r500c=r500c,
                                                       rho_500c=rho_500c,
                                                       r_flat=r_flat,
                                                       r_max=r_max_fb,
                                                       gamma=g,
                                                       r_c=r_c, beta=beta,
                                                       fgas_500c=fgas_500c)

        prof_tot = dm_plaw_r500c_rmax + gas_plaw_r500c_rmax + stars_rmax
        m_obs_200m_dmo = prof_tot.m_r(r200m_dmo)

        # now load dmo profiles
        dm_dmo_rmax = load_dm_dmo_rmax(prms=prms,
                                       r_max=r_max_fb,
                                       m200m_dmo=m200m_dmo,
                                       r200m_dmo=r200m_dmo,
                                       c200m_dmo=c200m_dmo)


        temp = {'d_gas': gas_plaw_r500c_rmax,
                'd_dm': dm_plaw_r500c_rmax,
                'd_stars': stars_rmax,
                'd_tot': prof_tot,
                'd_dmo': dm_dmo_rmax,
                'gamma': g,
                'm200m_obs': m200m_obs,
                'm_obs_200m_dmo': m_obs_200m_dmo,
                'r200m_obs': r200m_obs,
                'm200c_obs': m200c_obs,
                'r200c_obs': r200c_obs,
                # 'm2500c_obs': m2500c_obs,
                # 'r2500c_obs': r2500c_obs,
                'r_max': r_max_fb,
                'r_flat': r_flat}
        results['{:d}'.format(idx_g)] = temp

    return results


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
    fig = plt.figure(figsize=(20, 8))
    ax1 = fig.add_axes([0.1, 0.1, 0.266, 0.8])
    ax2 = fig.add_axes([0.366, 0.1, 0.266, 0.8])
    ax3 = fig.add_axes([0.632, 0.1, 0.266, 0.8])

    idx_1 = 0
    idx_2 = 50
    idx_3 = -1
    r200m = prms.r200m
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
