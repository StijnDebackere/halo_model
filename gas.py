import numpy as np
import astropy.constants as const
import astropy.units as u

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
    M500 *= 1.3
    return M500

# ------------------------------------------------------------------------------
# End of T2Mwl()
# ------------------------------------------------------------------------------

def rhogas_eckert():
    # r  : [r500]
    # ni : [cm^-3]
    # si : [cm^-3] -> error
    path = ddir + 'data_mccarthy/gas/ngas_profiles.txt'
    r, n1, s1, n2, s2, n3, s3, n4, s4 = np.loadtxt(path, unpack=True)
    # rescale to hydrostatic r500_hydro/r500_wl ~ (M_hydro/M_wl)^1/3 = 1.09
    r *= 1.09
    # rho = mu * m_p * n_gas -> fully ionised: mu=0.59 (X=0.75, Y=0.25)
    # n_gas = n_e + n_H + n_He = 2n_H + 3n_He (fully ionized)
    # n_gas = 2.25 n_H
    # seems to need an extra factor of 1/h^2 to match sun profiles?
    rho1 = 2.25 * 0.59 * const.m_p.cgs * n1 * 1/u.cm**3 # in cgs
    rho2 = 2.25 * 0.59 * const.m_p.cgs * n2 * 1/u.cm**3
    rho3 = 2.25 * 0.59 * const.m_p.cgs * n3 * 1/u.cm**3
    rho4 = 2.25 * 0.59 * const.m_p.cgs * n4 * 1/u.cm**3
    s1 *= 2.25 * 0.59 * const.m_p.cgs * s1 * 1/u.cm**3 
    s2 *= 2.25 * 0.59 * const.m_p.cgs * s2 * 1/u.cm**3
    s3 *= 2.25 * 0.59 * const.m_p.cgs * s3 * 1/u.cm**3
    s4 *= 2.25 * 0.59 * const.m_p.cgs * s4 * 1/u.cm**3
    # data is in temperature bins, need to convert to mass bins
    T_bins = np.array([0.6, 2, 3, 4, 7])
    # halo mass bin edges! What are min and max mass?
    m_bins = T2Mwl(T_bins)

    cgs2cos = (1e6 * const.pc.cgs)**3 / const.M_sun.cgs
    rho = (np.vstack([rho1, rho2, rho3, rho4]) * cgs2cos).value
    s = (np.vstack([s1, s2, s3, s4]) * cgs2cos).value

    return r, rho, s, m_bins

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
    rho = rho_m * prms.rho_crit

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
    rho = rho_m * prms.rho_crit

    return r500, rho
    
# ------------------------------------------------------------------------------
# End of rhogas_croston()
# ------------------------------------------------------------------------------

def fit_beta_eckert():
    r500, rho, s, m_edges = rhogas_eckert()
    r200 = r500 * tools.rx_to_r200(1., 500)
    m_bins = 0.5* (m_edges[1:] + m_edges[:-1])
    m_range = p.prms.m_range_lin

    # eckert 0 -> rho_eckert[0][3:] cutoff

    # fit_prms = np.empty((0,4), dtype=float)
    # profiles = np.empty((0,) + r200.shape, dtype=float)
    fit_prms = []
    profiles = []
    idx = 0
    for profile, m_bin in zip(rho, m_bins):
        m_idx = np.argmin(np.abs(m_range - m_bin))
        r_x = prms.r_h[m_idx]
        if idx == 0:
            fit, prof = profs.fit_profile_beta_plaw(r200[3:] * r_x,
                                                    r_x,
                                                    profile[3:])
        else:
            fit, prof = profs.fit_profile_beta_plaw(r200 * r_x,
                                                    r_x,
                                                    profile)
        fit_prms.append(fit)
        profiles.append(prof)

    return fit_prms, profiles
