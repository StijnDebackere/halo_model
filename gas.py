import numpy as np
import astropy.constants as const
import astropy.units as u
import scipy.optimize as opt
import scipy.stats as stats
import h5py
import sys

import matplotlib.pyplot as plt
# allow import of plot
sys.path.append('~/Documents/Universiteit/MR/code')
import plot as pl

import halo.parameters as p
import halo.density_profiles as profs
import halo.tools as tools

import pdb

ddir = '/Volumes/Data/stijn/Documents/Universiteit/MR/code/data/'
prms = p.prms

def T2Mgas(T):
    '''
    Returns M500 for gas associated to temperature T (in keV)

        log[M500 / h_70^-5/2] = log(12.22) + 2.02 log(T)

    Eq. 6 in Eckert et al. XXL XIII (2015)
    '''

    M500 = 10**(12.22) * (T**2.02)
    return M500

# ------------------------------------------------------------------------------
# End of T2Mgas()
# ------------------------------------------------------------------------------

def T2Mwl(T):
    '''
    Returns M500 for halo associated to temperature T (in keV)

        log[M500 / h^-5/2] = log(12.22) + 2.02 log(T)

    Eq. 6 in Eckert et al. XXL XIII (2015)
    '''
    M500 = 10**(13.57) * (T**1.67)
    # faulty measurement -> scale to match hydrostatic mass
    M500 /= 1.3
    return M500

# ------------------------------------------------------------------------------
# End of T2Mwl()
# ------------------------------------------------------------------------------

def Mwl2Mgas(mwl):
    '''
    Return m500_gas for m500_wl

        f500_gas = 0.055 * (m500_wl / 10^14)^0.21

    Eq. 7 in Eckert et al. XXL XIII (2015)
    '''
    f_gas = 0.055 * (mwl / 10**14)**0.21
    return f_gas * mwl

# ------------------------------------------------------------------------------
# End of Mwl2Mgas()
# ------------------------------------------------------------------------------

def rhogas_eckert():
    # r  : [r500]
    # ni : [cm^-3]
    # si : [cm^-3] -> error
    path = ddir + 'data_mccarthy/gas/ngas_profiles.txt'
    r, n1, s1, n2, s2, n3, s3, n4, s4 = np.loadtxt(path, unpack=True)
    # rescale to hydrostatic r500_wl/r500_hydro ~ (M_wl/M_hydro)^1/3 = 1.09
    r *= 1.09
    # rho = mu * m_p * n_gas -> fully ionised: mu=0.59 (X=0.75, Y=0.25)
    # n_gas = n_e + n_H + n_He = 2n_H + 3n_He (fully ionized)
    # n_He = Y/(4X) n_H
    # => n_gas = 2.25 n_H
    # seems to need an extra factor of 1/h^2 to match sun profiles?
    rho1 = 2.25 * 0.59 * const.m_p.cgs * n1 * 1/u.cm**3 # in cgs
    rho2 = 2.25 * 0.59 * const.m_p.cgs * n2 * 1/u.cm**3
    rho3 = 2.25 * 0.59 * const.m_p.cgs * n3 * 1/u.cm**3
    rho4 = 2.25 * 0.59 * const.m_p.cgs * n4 * 1/u.cm**3
    s1 = 2.25 * 0.59 * const.m_p.cgs * s1 * 1/u.cm**3
    s2 = 2.25 * 0.59 * const.m_p.cgs * s2 * 1/u.cm**3
    s3 = 2.25 * 0.59 * const.m_p.cgs * s3 * 1/u.cm**3
    s4 = 2.25 * 0.59 * const.m_p.cgs * s4 * 1/u.cm**3
    # data is in temperature bins, need to convert to mass bins
    T_bins = np.array([0.6, 2, 3, 4, 5])
    # halo mass bin edges! What are min and max mass?
    mwl_bins = T2Mwl(T_bins)
    mgas_bins = T2Mgas(T_bins)

    # change to ``cosmological'' coordinates
    cgs2cos = (1e6 * const.pc.cgs)**3 / const.M_sun.cgs
    rho = (np.vstack([rho1, rho2, rho3, rho4]) * cgs2cos).value
    s = (np.vstack([s1, s2, s3, s4]) * cgs2cos).value

    mwl = 0.5 * (mwl_bins[1:] + mwl_bins[:-1])
    mgas = 0.5 * (mgas_bins[1:] + mgas_bins[:-1])

    return  r, rho, s, mwl, mgas

# ------------------------------------------------------------------------------
# End of rhogas_eckert()
# ------------------------------------------------------------------------------

def rhogas_sun():
    # Gas
    # data for median halo mass M500 ~ 8.7e13 Msun
    # r500, rho/rho_crit(med, 16th, 84th percentiles)
    path = ddir + 'data_mccarthy/gas/Sun09_rhogas_profiles.dat'
    r500, rho_m, rho_16, rho_84 = np.loadtxt(path, unpack=True)

    rhogas_sun.M = 8.7e13
    rho = rho_m
    err = np.array([rho_m - rho_16, rho_84 - rho_m])

    return r500, rho, err

# ------------------------------------------------------------------------------
# End of rhogas_sun()
# ------------------------------------------------------------------------------

def rhogas_croston():
    # Gas
    # data for median halo mass M500 ~ 8.7e13 Msun
    # r500, rho/rho_crit(med, 16th, 84th percentiles)
    path = ddir + 'data_mccarthy/gas/Croston08_rhogas_profiles.dat'
    r500, rho_m, rho_16, rho_84 = np.loadtxt(path, unpack=True)

    rhogas_croston.M = 3e14
    rho = rho_m
    err = np.array([rho_m - rho_16, rho_84 - rho_m])

    return r500, rho, err

# ------------------------------------------------------------------------------
# End of rhogas_croston()
# ------------------------------------------------------------------------------

def fit_beta_eckert():
    r500, rho, s, mwl, mgas = rhogas_eckert()
    idx_500 = np.argmin(np.abs(r500 - 1))
    norm = tools.m_h(rho[:,:idx_500+1], r500[:idx_500+1].reshape(1,-1), axis=-1)
    rho_norm = rho / norm.reshape(-1,1)
    s_norm = s / norm.reshape(-1,1)

    fit_prms = []
    covs     = []
    profiles = []
    for idx, profile in enumerate(rho_norm):
        if idx == 0:
            # do not fit bump
            fit, cov, prof = profs.fit_profile_beta_plaw(r500[3:], 1,
                                                    r500[idx_500],
                                                    profile[3:],
                                                    err=s_norm[idx,3:])
        else:
            fit, cov, prof = profs.fit_profile_beta_plaw(r500, 1,
                                                    r500[idx_500],
                                                    profile,
                                                    err=s_norm[idx])
        fit_prms.append(fit)
        covs.append(cov)
        profiles.append(prof)

    return fit_prms, covs, profiles

# ------------------------------------------------------------------------------
# End of fit_beta_eckert()
# ------------------------------------------------------------------------------

def fit_beta_bahamas():
    '''
    Fit beta profiles to the bahamas bins
    '''
    binned = h5py.File('/Volumes/Data/stijn/Documents/Universiteit/MR/code/data/BAHAMAS/eagle_subfind_particles_032_profiles_binned_500.hdf5 ', 'r')

    rho = binned['PartType0/MedianDensity'][:]
    q16 = binned['PartType0/Q16'][:]
    q84 = binned['PartType0/Q84'][:]
    r500_inbin = binned['PartType0/R500'][:]
    numbin = binned['PartType0/NumBin'][:]
    to_slice = np.concatenate([[0], numbin])
    bin_slice = np.concatenate([np.cumsum(to_slice[:-1]).reshape(-1,1),
                                np.cumsum(to_slice[1:]).reshape(-1,1)], axis=-1)
    r500 = np.array([np.median(r500_inbin[sl[0]:sl[1]]) for sl in bin_slice])
    err = np.maximum(rho - 16, q84 - rho)

    r_bins = binned['RBins_R_Mean500'][:]
    r = tools.bins2center(r_bins)

    norm = tools.m_h(rho, r500.reshape(1,-1), axis=-1)
    rho_norm = rho / norm.reshape(-1,1)
    err_norm = err / norm.reshape(-1,1)

    fit_prms = []
    covs     = []
    profiles = []
    for idx, profile in enumerate(rho_norm):
        if idx == 0:
            # do not fit bump
            fit, cov, prof = profs.fit_profile_beta_plaw(r500[3:], 1,
                                                    r500[idx_500],
                                                    profile[3:],
                                                    err=s_norm[idx,3:])
        else:
            fit, cov, prof = profs.fit_profile_beta_plaw(r500, 1,
                                                    r500[idx_500],
                                                    profile,
                                                    err=s_norm[idx])
        fit_prms.append(fit)
        covs.append(cov)
        profiles.append(prof)

    return fit_prms, covs, profiles

# ------------------------------------------------------------------------------
# End of fit_beta_bahamas()
# ------------------------------------------------------------------------------

def f_gas(M_halo, M_trans, a):
    '''
    Returns parametrization for gas fraction.

    Based on sigmoid parametrization in Schneider & Teyssier (2015).

    Parameters
    ----------
    M_halo : array
      Halo mass in terms of M_500
    M_trans : float
      Mass of transition in units of M_500
    a : float
      power law slope of sigmoid

    Returns
    -------
    f_gas : array
      Gas fraction for halo of mass M_halo
    '''
    # baryon fraction
    f_b = p.prms.omegab / p.prms.omegam
    return f_b / (1 + (M_trans/M_halo)**a)

# ------------------------------------------------------------------------------
# End of f_gas()
# ------------------------------------------------------------------------------

def f_gas_fit(m_range=p.prms.m_range_lin, n_bins=10):
    '''
    Fit the f_gas-M_500 relation
    '''
    m500, f = np.loadtxt(ddir + 'data_mccarthy/gas/M500_fgas_BAHAMAS_data.dat',
                         unpack=True)
    f_b = p.prms.omegab / p.prms.omegam
    # bin f_gas
    f_med, edges, bin_idx = stats.binned_statistic(x=m500, values=f,
                                                   statistic='median',
                                                   bins=n_bins)
    f_std, edges, bin_idx = stats.binned_statistic(x=m500, values=f,
                                                   statistic=np.std,
                                                   bins=n_bins)
    m500 = np.power(10, m500)
    # c_x = profs.c_correa(m_range, z_range=0).reshape(-1)
    # m500_model = m_range * tools.Mx_to_My(1., 200, 500, c_x, p.prms.rho_m * 0.7**2)

    # m_idx = np.argmin(np.abs(m500.reshape(-1,1) - m500_model.reshape(1,-1)),
    #                   axis=-1)

    centers = np.power(10, 0.5*(edges[1:] + edges[:-1]))
    popt, pcov = opt.curve_fit(f_gas, centers, f_med, sigma=f_std,
                               bounds=([1e10, 0],[1e15, 2]))

    fit_prms = {'M_trans' : popt[0],
                'a' : popt[1]}
                # 'f_0' : popt[2]}
    fit = f_gas(centers, **fit_prms)

    # plt.plot(m500, f, label=r'data', lw=0, marker='o')
    # plt.plot(centers, f_med, label=r'median')
    # plt.plot(centers, fit, label=r'fit')
    # plt.axhline(y=f_b, c='k')
    # plt.xscale('log')
    # plt.xlabel(r'$\log_{10} M/M_\odot$')
    # plt.ylabel(r'$f_{\mathrm{gas}}$')
    # plt.legend(loc='best')
    # plt.show()

    return fit_prms

# ------------------------------------------------------------------------------
# End of f_gas_fit()
# ------------------------------------------------------------------------------

def plot_eckert_fits():
    '''
    Plot beta profile fits to Eckert profiles
    '''
    pl.set_style()
    r500, rho, s, mwl, mgas = rhogas_eckert()
    idx_500 = np.argmin(np.abs(r500 - 1))
    norm = tools.m_h(rho[:,:idx_500+1], r500[:idx_500+1].reshape(1,-1), axis=-1)
    rho_norm = rho / norm.reshape(-1,1)

    prms, fits = fit_beta_eckert()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_prop_cycle(pl.cycle_mark())
    marks = []
    for idx, prof in enumerate(rho_norm):
        mark, = ax.plot(r500, prof)
        marks.append(mark)

    ax.set_prop_cycle(pl.cycle_line())
    lines = []
    for idx, fit in enumerate(fits):
        if idx == 0:
            line, = ax.plot(r500[3:], fit)
        else:
            line, = ax.plot(r500, fit)
        lines.append(line)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$r/r_{500}$')
    ax.set_ylabel(r'$\rho(r)/M$')
    ax.legend([(line, mark) for line, mark in zip(lines, marks)],
              [r'$M=10^{%.1f}$'%np.log10(c) for c in mwl])
    plt.show()

# ------------------------------------------------------------------------------
# End of plot_eckert_fit()
# ------------------------------------------------------------------------------

def fit_sun_profile():
    r500, rho, err = rhogas_sun()
    rho *= p.prms.rho_crit * 0.7**2# / 3.3
    err *= p.prms.rho_crit * 0.7**2

    # bin profile
    r500 = r500.reshape(-1,4).mean(axis=1)
    rho = rho.reshape(-1,4).mean(axis=1)
    err = 1./2 * np.sqrt(np.sum(err.reshape(2,-1,4)**2, axis=-1))

    # match mass at r500c
    idx_500 = np.argmin(np.abs(r500 - 1))
    norm = tools.m_h(rho[:idx_500+1], r500[:idx_500+1])

    rho_norm = rho / norm
    err_norm = err / norm
    sigma = np.maximum(err_norm[0], err_norm[1])

    fit_plaw = profs.fit_profile_beta_plaw(r500, 1, r500[idx_500], rho_norm, sigma)
    fit_beta = profs.fit_profile_beta(r500, 1, r500[idx_500], rho_norm, sigma)

    # M = 2.5e14
    M = rhogas_sun.M
    r200m = m500c_to_r200m(M)
    m200m = tools.radius_to_mass(r200m, 200 * p.prms.rho_m * 0.7**2)
    r500c = tools.mass_to_radius(M, 500 * p.prms.rho_crit * 0.7**2)

    print 'r200m: ', r200m
    print 'r500c: ', r500c
    print 'ratio: ', r500c/r200m

    r_range = np.logspace(-4, np.log10(r200m / r500c), 100)
    u_beta = profs.profile_b(r_range, 1, 1, **fit_beta[0])
    # u_beta = profs.profile_beta_extra(r_range, u_beta, 1, 3)
    u_plaw = profs.profile_b_plaw(r_range, 1, 1, **fit_plaw[0])
    # u_plaw = profs.profile_beta_extra(r_range, u_plaw, 1, 3)

    # need to divide out r500c to match observed profiles
    prof_beta = (f_gas(M, **f_gas_fit()) * M * u_beta / r500c**3)
    prof_plaw = (f_gas(M, **f_gas_fit()) * M * u_plaw / r500c**3)

    idx_500 = np.argmin(np.abs(r_range - 1))
    mg500 = tools.m_h(prof_beta[:idx_500 + 1], r_range[:idx_500 + 1]) * r500c**3
    mg200 = tools.m_h(prof_beta, r_range) * r500c**3

    print 'fgas_500: ', mg500 / M
    print 'fgas_200: ', mg200 / m200m

    # pl.set_style()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    # ax.set_prop_cycle(pl.cycle_mark())
    # ax.errorbar(r500, rho, yerr=[err[0], err[1]], fmt='o',
    #             label=r'Sun+2009')

    # ax.set_prop_cycle(pl.cycle_line())
    # ax.plot(r_range, prof_plaw, label=r'$\beta$ + power law')
    # ax.plot(r_range, prof_beta, label=r'$\beta$')
    # # ax.axvline(x=r200m/r500c, c='k', ls='--')

    # ax.set_xlim([0.9*r_range.min(), 1.1*r_range.max()])
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_xlabel(r'$r/r_{500}$')
    # ax.set_ylabel(r'$\rho(r) = Mu(r)$')
    # ax.legend(loc='best', numpoints=1)
    # plt.show()

    return fit_plaw, fit_beta

# ------------------------------------------------------------------------------
# End of fit_sun_profile()
# ------------------------------------------------------------------------------

def fit_croston_profile():
    r500, rho, err = rhogas_croston()
    rho *= p.prms.rho_crit * 0.7**2
    err *= p.prms.rho_crit * 0.7**2

    r500 = r500[:-1]
    rho = rho[:-1]
    err = err[:,:-1]

    # match mass at r500c
    idx_500 = np.argmin(np.abs(r500 - 1))
    norm = tools.m_h(rho[:idx_500+1], r500[:idx_500+1])

    rho_norm = rho / norm
    err_norm = err / norm
    sigma = np.maximum(err_norm[0], err_norm[1])

    fit_plaw = profs.fit_profile_beta_plaw(r500, 1, r500[idx_500], rho_norm, sigma)
    fit_beta = profs.fit_profile_beta(r500, 1, r500[idx_500], rho_norm, sigma)

    M = rhogas_croston.M
    r200m = m500c_to_r200m(M)
    m200m = tools.radius_to_mass(r200m, 200 * p.prms.rho_m * 0.7**2)
    r500c = tools.mass_to_radius(M, 500 * p.prms.rho_crit * 0.7**2)

    print 'r200m: ', r200m
    print 'r500c: ', r500c
    print 'ratio: ', r500c/r200m

    r_range = np.logspace(-4, np.log10(r200m / r500c), 100)
    u_beta = profs.profile_b(r_range, 1, 1, **fit_beta[0])
    # u_beta = profs.profile_beta_extra(r_range, u_beta, 1, 3)
    u_plaw = profs.profile_b_plaw(r_range, 1, 1, **fit_plaw[0])
    # u_plaw = profs.profile_beta_extra(r_range, u_plaw, 1, 3)

    # need to divide out r500c to match observed profiles
    prof_beta = (f_gas(M, **f_gas_fit()) * M * u_beta / r500c**3)
    prof_plaw = (f_gas(M, **f_gas_fit()) * M * u_plaw / r500c**3)

    idx_500 = np.argmin(np.abs(r_range - 1))
    mg500 = tools.m_h(prof_beta[:idx_500 + 1], r_range[:idx_500 + 1]) * r500c**3
    mg200 = tools.m_h(prof_beta, r_range) * r500c**3

    print 'fgas_500: ', mg500 / M
    print 'fgas_200: ', mg200 / m200m

    # pl.set_style()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    # ax.set_prop_cycle(pl.cycle_mark())
    # ax.errorbar(r500, rho, yerr=[err[0], err[1]], fmt='o',
    #             label=r'Croston+2008')

    # ax.set_prop_cycle(pl.cycle_line())
    # ax.plot(r_range, prof_plaw, label=r'$\beta$ + power law')
    # ax.plot(r_range, prof_beta, label=r'$\beta$')
    # # ax.axvline(x=r200m/r500c, c='k', ls='--')

    # ax.set_xlim([0.9*r_range.min(), 1.2*r_range.max()])
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_xlabel(r'$r/r_{500}$')
    # ax.set_ylabel(r'$\rho(r) = Mu(r)$')
    # ax.legend(loc='best', numpoints=1)
    # plt.show()


    return fit_plaw[:-1], fit_beta[:-1]

# ------------------------------------------------------------------------------
# End of fit_croston_profile()
# ------------------------------------------------------------------------------
def massdiff_cm(m500c, m200m):
    '''
    Integrate an NFW halo with m200m up to r500c and return the mass difference
    between the integral and m500c
    '''
    r500c = (m500c / (4./3 * np.pi * 500 * p.prms.rho_crit * 0.7**2))**(1./3)
    r_range = np.logspace(-4, np.log10(r500c), 1000)

    r200m = tools.mass_to_radius(m200m, 200 * p.prms.rho_m * 0.7**2)
    dens = profs.profile_NFW(r_range, np.array([m200m]),
                             profs.c_correa(m200m, 0).reshape(-1),
                             r200m, p.prms.rho_m * 0.7**2, Delta=200)
    mass_int = tools.m_h(dens, r_range)

    return mass_int - m500c


def massdiff_mc(m200m, m500c):
    '''
    Integrate an NFW halo with m200m up to r500c and return the mass difference
    between the integral and m500c
    '''
    r500c = (m500c / (4./3 * np.pi * 500 * p.prms.rho_crit * 0.7**2))**(1./3)
    r_range = np.logspace(-4, np.log10(r500c), 1000)

    r200m = tools.mass_to_radius(m200m, 200 * p.prms.rho_m * 0.7**2)
    dens = profs.profile_NFW(r_range, np.array([m200m]),
                             profs.c_correa(m200m, 0).reshape(-1),
                             r200m, p.prms.rho_m * 0.7**2, Delta=200)
    mass_int = tools.m_h(dens, r_range)

    return mass_int - m500c

def m500c_to_m200m(m500c):
    '''
    Give the virial mass for the halo corresponding to m500c

    Parameters
    ----------
    m500c : float
      halo mass at 500 times the universe critical density

    Returns
    -------
    m200m : float
      corresponding halo model halo virial mass
    '''
    # 1e19 Msun is ~maximum for c_correa
    m200m = opt.brentq(massdiff_mc, 1e10, 1e19, args=(m500c))

    return m200m

# ------------------------------------------------------------------------------
# End of m500c_to_m200m()
# ------------------------------------------------------------------------------

def m200m_to_m500c(m200m):
    '''
    Give m500c for the an m200m virial mass halo

    Parameters
    ----------
    m200m : float
      halo virial mass

    Returns
    -------
    m500c : float
      halo mass at 500 times the universe critical density
    '''
    # 1e19 Msun is ~maximum for c_correa
    m500c = opt.brentq(massdiff_cm, 1e8, 1e19, args=(m200m))

    return m500c

# ------------------------------------------------------------------------------
# End of m500c_to_m200m()
# ------------------------------------------------------------------------------
