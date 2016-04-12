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

# ------------------------------------------------------------------------------
def Mgas_M500_lovisari(m500, h=0.7):
    '''
    Returns M_gas - M_500 (eq. 9) relation from Lovisari (2015)

        http://dx.doi.org/10.1051/0004-6361/201423954

    '''
    h /= 0.7
    mgas = 10**(-0.16) * (m500 / 5e13 * h**-1)**(1.22) * 5e12 * h**(-2.5)

    return mgas

# ------------------------------------------------------------------------------
# End of Mgas_M500_lovisari()
# ------------------------------------------------------------------------------

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

    return r, rho, s, mwl, mgas

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

    return r500, rho

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

    return r500, rho

# ------------------------------------------------------------------------------
# End of rhogas_croston()
# ------------------------------------------------------------------------------

def m200_eckert():
    def M2Mgas(M200, rho, r):
        c_x = profs.c_correa(M200, z_range=0).reshape(-1)
        M500 = tools.Mx_to_My(M200, 200, 500, c_x, p.prms.rho_m * 0.7**2)
        M_gas = f_gas(M500, **fit_prms) * M500
        r200 = tools.r_delta(M200, 200, p.prms.rho_m * 0.7**2)
        r500 = r200 * tools.rx_to_r200(500, c_x, p.prms.rho_m * 0.7**2)

        Mgas = tools.Integrate(rho, r*r500)
        # include some way to take difference between determined and guessed
        # mass
        diff = Mgas - M_gas #+ mass - M500
        return diff

    r500, rho, s, mwl, mgas = rhogas_eckert()
    fit_prms = f_gas_fit()

    m200 = []
    for prof, mass in zip(rho, mwl):
        m200.append(opt.brentq(M2Mgas, 1e10, 1e15,
                               args=(prof, r500)))

    return np.array(m200)

# ------------------------------------------------------------------------------
# End of m200_eckert()
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

def compare_eckert_fractions():
    '''
    Compare different methods to get gas fraction
    '''
    r_eckert, rho_eckert, s, mwl, mgas = rhogas_eckert()
    mgas_mwl = Mwl2Mgas(mwl)
    mgas_f = mwl * f_gas(mwl, **f_gas_fit())

    pl.set_style('mark')
    plt.plot(mwl, mgas/mwl, label=r'$M_{\mathrm{gas}}-T$')
    plt.plot(mwl, mgas_mwl/mwl, label=r'$f_{\mathrm{gas}}-M_{\mathrm{wl}}$')
    plt.plot(mwl, mgas_f/mwl, label=r'Observations')
    plt.xscale('log')
    plt.xlabel(r'$M_{\mathrm{wl},500}\,[M_\odot]$')
    plt.ylabel(r'$f_{\mathrm{gas}}$')
    plt.legend(loc='best')
    plt.show()

# ------------------------------------------------------------------------------
# End of compare_eckert_fractions()
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
