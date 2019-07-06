import halo.parameters as p
import halo.tools as tools
import halo.input.interpolators as inp_interp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
import scipy.optimize as opt
import scipy.interpolate as interp
import astropy.constants as const
import astropy.units as u
import astropy.io.fits as fits
import glob
import re
import sys
import pickle

# allow import of plot
sys.path.append('/Users/stijn/Documents/MR/code')

import pdb

ddir = '/Users/stijn/Documents/MR/code/halo/data/'
prms = p.prms


def read_croston():
    # Load in croston metadata
    ddir = 'halo/data/data_mccarthy/gas/Croston_data/'
    files = glob.glob(ddir + 'Croston08*.dat')
    fnames = [f.split('/')[-1] for f in files]
    idcs = [re.search('[0-9]*.dat', f).span() for f in fnames]
    sysnum = [int(fnames[idx][i[0]:i[1]][:-4]) for idx, i in enumerate(idcs)]

    # ---------------------------------------- #
    # We leave all the original h_70 scalings, #
    # we scale our model when comparing        #
    # ---------------------------------------- #

    data = np.loadtxt(ddir + 'Pratt09.dat')
    z = data[:,0]
    # r500 = data[:,1] * 1e-3 * 0.7 # [Mpc/h]
    # mgas500 = np.power(10, data[:,2]) * (0.7)**(5./2) # [Msun/h^(5/2)]
    # mgas500_err = np.power(10, data[:,3]) * (0.7)**(5./2) # [Msun/h^(5/2)]
    r500 = data[:,1] * 1e-3 # [Mpc/h_70]
    mgas500 = np.power(10, data[:,2]) # [Msun/h_70^(5/2)]
    mgas500_err = np.power(10, data[:,3]) # [Msun/h_70^(5/2)]
    Z = data[:,4]

    # Load in croston data -> n = n_e
    # r = [np.loadtxt(f)[:,0] * 10**(-3) * 0.7 for idx, f in enumerate(files)] # [Mpc/h]
    r = [np.loadtxt(f)[:,0] * 10**(-3) for idx, f in enumerate(files)] # [Mpc/h_70]
    rx = [np.loadtxt(f)[:,1] for idx, f in enumerate(files)]

    # from Flux ~ EM ~ 1/d_A^2 * int n^2 dV ~ h^2 * h^-3 * [n^2]
    # where n is theoretical model and Flux is measurement, so n^2 ~ F * h
    # n ~ h^(1/2)
    # n = [np.loadtxt(f)[:,2] / (0.7)**(0.5) for idx, f in enumerate(files)] # [cm^-3 h^(1/2)]
    # n_err = [np.loadtxt(f)[:,3] / (0.7)**(0.5) for idx, f in enumerate(files)] # [cm^-3 h^(1/2)]
    n = [np.loadtxt(f)[:,2] for idx, f in enumerate(files)] # [cm^-3 h_70^(1/2)]
    n_err = [np.loadtxt(f)[:,3] for idx, f in enumerate(files)] # [cm^-3 h_70^(1/2)]

    # Convert electron densities to gas density
    # Interpolate (X_0, Y_0, Z_0) = (0.75, 0.25, 0)
    #          to (X_s, Y_s, Z_s) = (0.7133, 0.2735, 0.0132)
    #         for (X, Y, Z) = (0.73899, 0.25705, 0.00396) for Z = 0.3Z_s
    # mu = (1 + Y/X) / (2 + 3Y / (4X))
    # rho = mu * m_p * n_gas -> fully ionised: mu=0.60 (X=0.73899, Y=0.25705, Z=0.00396)
    # n_gas = n_e + n_H + n_He = 2n_H + 3n_He (fully ionized)
    #       = (2 + 3Y/(4X))n_H
    # n_He = Y/(4X) n_H
    # n_e = n_H + 2n_He = (1 + Y/(2X)) n_H
    # n_gas = (2 + 3Y/(4X)) / (1 + Y/(2X)) n_e
    # => n_gas = 1.93 n_e
    # calculate correct mu factor for Z metallicity gas
    n2rho = (1.93 * 0.6 * const.m_p.cgs * 1/u.cm**3).value # [g/cm^3]
    # change to ``cosmological'' coordinates
    cgs2cos = ((1e6 * const.pc.cgs)**3 / const.M_sun.cgs).value
    rho =  [(ne * n2rho * cgs2cos) for ne in n]
    rho_err = [(ne * n2rho * cgs2cos) for ne in n_err]

    mgas = np.empty((0,), dtype=float)
    m500gas = np.empty((0,), dtype=float)
    for idx, prof in enumerate(rho):
        idx_500 = np.argmin(np.abs(rx[idx] - 1))
        mgas = np.append(mgas,
                         tools.m_h(prof[:idx_500+1], rx[idx][:idx_500+1] *
                                   r500[sysnum[idx]-1]))
        m500gas = np.append(m500gas, mgas500[sysnum[idx] - 1])
        # print mgas[idx]
        # print mgas500[sysnum[idx]-1]
        # print '-----------'

    ratio = mgas/m500gas
    print('Croston derived gas masses:')
    print('median: ', np.median(ratio))
    print('q15:    ', np.percentile(ratio, q=15))
    print('q85:    ', np.percentile(ratio, q=85))

    # plt.plot(rx[idx], prof)

    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()

    # ratio = mgas/m500gas
    # pl.set_style('mark')
    # plt.hist(ratio)
    # # plt.plot(m500gas, ratio)
    # # plt.axhline(y=1, color='k')
    # plt.axvline(x=np.median(ratio), c='k', ls='-', label='median')
    # plt.axvline(x=np.percentile(ratio, 15), c='k', ls='--',
    #             label='$15-85^\mathrm{th}$ percentile')
    # plt.axvline(x=np.percentile(ratio, 85), c='k', ls='--')
    # # plt.xscale('log')
    # # plt.xlabel(r'$m_\mathrm{500c,gas-measured} \, [M_\odot/h_{70}^{5/2}]$')
    # # plt.xlabel(r'$\left(\int\,\rho(r)\,\mathrm{d}^3r\right)/m_\mathrm{500c,gas-measured}$',
    # #            labelpad=-10)
    # plt.xlabel(r'$m_\mathrm{500c,gas-integrated}/m_\mathrm{500c,gas-measured}$')
    # plt.ylabel('Frequency')
    # plt.title('Croston+08 mass determination')
    # plt.legend()
    # plt.show()

    m500c = tools.radius_to_mass(r500, 500 * prms.cosmo.rho_crit * 0.7**2)
    # we save our gas mass determinations!
    data = {'m500gas': mgas,
            'r500': r500,
            'm500c': m500c,
            'rx': rx,
            'rho': rho,
            'rho_err': rho_err}

    with open('data/croston.p', 'wb') as f:
        pickle.dump(data, f)

    return m500gas, r500, rx, rho, rho_err

# ------------------------------------------------------------------------------
# End of read_croston()
# ------------------------------------------------------------------------------

def read_eckert():
    ddir = 'halo/data/data_mccarthy/gas/Eckert_data/'
    files = glob.glob(ddir + 'XLSSC*.fits')

    # metadata
    mdata = fits.open(ddir + 'XXL100GC.fits')
    units = mdata[1].header

    # -------------------------------------------------------------------------- #
    # We leave all the original h_70 scalings, we scale our model when comparing #
    # -------------------------------------------------------------------------- #

    # number of cluster
    num = mdata[1].data['xlssc']
    # m500mt = mdata[1].data['M500MT'] * 0.7 # [Msun/h]
    # m500mt_err = mdata[1].data['M500MT_err'] * 0.7 # [Msun/h]
    # mgas500 = mdata[1].data['Mgas500'] * (0.7)**(5./2) # [Msun/h^(5/2)]
    # mgas500_err = mdata[1].data['Mgas500_err'] * (0.7)**(5./2) # [Msun/h^(5/2)]
    m500mt = mdata[1].data['M500MT'] # [Msun/h_70]
    m500mt_err = mdata[1].data['M500MT_err'] # [Msun/h_70]
    mgas500 = mdata[1].data['Mgas500'] # [Msun/h_70^(5/2)]
    mgas500_err = mdata[1].data['Mgas500_err'] # [Msun/h_70^(5/2)]
    mdata.close()

    m500mt *= 1e13
    m500mt_err *= 1e13
    mgas500 *= 1e13
    mgas500_err *= 1e13


    # system info
    fnames = [f.split('/')[-1] for f in files]
    idcs = [re.search('[0-9]*_nh.fits', f).span() for f in fnames]
    numdata = np.array([int(fnames[idx][i[0]:i[1]][:-8])
                        for idx,i in enumerate(idcs)])
    # Interpolate (X_0, Y_0, Z_0) = (0.75, 0.25, 0)
    #          to (X_s, Y_s, Z_s) = (0.7133, 0.2735, 0.0132)
    #         for (X, Y, Z) = (0.73899, 0.25705, 0.00396) for Z = 0.3Z_s
    # mu = (1 + Y/X) / (2 + 3Y / (4X))
    # rho = mu * m_p * n_gas -> fully ionised: mu=0.6 (X=0.73899, Y=0.25705, Z=0.00396)
    # n_gas = n_e + n_H + n_He = 2n_H + 3n_He (fully ionized)
    # n_He = Y/(4X) n_H
    # n_gas = 2 + 3Y/(4X) n_H
    # => n_gas = 2.26 n_H
    n2rho = (2.26 * 0.6 * const.m_p.cgs * 1./u.cm**3).value # [cm^-3]
    cgs2cos = ((1e6 * const.pc.cgs)**3 / const.M_sun.cgs).value

    r500 = np.empty((0,), dtype=float)
    z = np.empty((0,), dtype=float)
    rx = []
    rho = []
    rho_err = []
    for f in files:
        # actual data
        data = fits.open(f)
        z = np.append(z, data[1].header['REDSHIFT'])

        # !!!! check whether this still needs correction factor from weak lensing
        # Currently included it, but biases the masses, since this wasn't done in
        # measurement by Eckert, of course
        r500 = np.append(r500, data[1].header['R500'] * 1e-3 * (1.3)**(-1./3)) # [Mpc/h_70]
        # r500 = np.append(r500, data[1].header['R500'] * 1e-3) # [Mpc/h_70]

        rx.append(data[1].data['RADIUS'] * (1.3)**(1./3))
        # rx.append(data[1].data['RADIUS'])

        # rho.append(data[1].data['NH'] * n2rho * cgs2cos / (0.7)**(0.5)) # [cm^-3 h^(1/2)]
        # rho_err.append(data[1].data['ENH'] * n2rho * cgs2cos / (0.7)**(0.5)) # [cm^-3 h^(1/2)]
        rho.append(data[1].data['NH'] * n2rho * cgs2cos) # [cm^-3 h_70^(1/2)]
        rho_err.append(data[1].data['ENH'] * n2rho * cgs2cos) # [cm^-3 h_70^(1/2)]

        data.close()

    mgas = np.empty((0,), dtype=float)
    m500gas = np.empty((0,), dtype=float)
    mdata2data = np.empty((0,), dtype=int)
    for idx, prof in enumerate(rho):
        idx_500 = np.argmin(np.abs(rx[idx] - 1))
        mgas = np.append(mgas,
                         tools.m_h(prof[:idx_500+1], rx[idx][:idx_500+1] *
                                   r500[idx]))
        mdata2data = np.append(mdata2data, (num == numdata[idx]).nonzero()[0][0])
        # print mgas[idx]
        # print mgas500[mdata2data[idx]]
        # print '------------'

    m500gas = mgas500[mdata2data]
    m500 = m500mt[mdata2data] / 1.3

    ratio = mgas/m500gas
    print('Eckert derived gas masses, too high if corrected rx for bias!!!:')
    print('median: ', np.median(ratio))
    print('q15:    ', np.percentile(ratio, q=15))
    print('q85:    ', np.percentile(ratio, q=85))

    # print("Eckert: ", m500gas / m500)
    print("Gas fraction Ours / Eckert: ", ((mgas / (m500)) / (m500gas / (m500 * 1.3))))


    # pl.set_style('mark')
    # plt.plot(m500, mgas/m500)
    # plt.xlabel(r'$m_\mathrm{500c,M-T}$')
    # plt.ylabel(r'$f_\mathrm{gas,500c}$')
    # plt.ylim([0.02,0.18])
    # plt.xscale('log')
    # plt.show()

    # pl.set_style('mark')
    # plt.plot(m500gas, ratio)
    # plt.axhline(y=1, color='k')
    # plt.xscale('log')
    # plt.xlabel(r'$m_\mathrm{500c,gas-measured} \, [M_\odot/h_{70}^{5/2}]$')
    # plt.ylabel(r'$\left(\int\,\rho(r)\,\mathrm{d}^3r\right)/m_\mathrm{500c,gas-measured}$',
    #            labelpad=-5)

    # plt.hist(ratio)
    # plt.axvline(x=np.median(ratio), c='k', ls='-', label='median')
    # plt.axvline(x=np.percentile(ratio, 15), c='k', ls='--',
    #             label='$15-85^\mathrm{th}$ percentile')
    # plt.axvline(x=np.percentile(ratio, 85), c='k', ls='--')
    # plt.xlabel(r'$m_\mathrm{500c,gas-integrated}/m_\mathrm{500c,gas-measured}$')
    # plt.ylabel('Frequency')

    # plt.title('Eckert+16 mass determination')
    # plt.legend()
    # plt.show()

    m500c = tools.radius_to_mass(r500, 500 * prms.cosmo.rho_crit * 0.7**2)
    # we save our gas mass determinations!
    data = {'m500gas': mgas,
            'r500': r500,
            'm500c': m500c,
            'rx': rx,
            'rho': rho,
            'rho_err': rho_err}

    with open('data/eckert.p', 'wb') as f:
        pickle.dump(data, f)

    return m500gas, m500, r500, rx, rho, rho_err

# ------------------------------------------------------------------------------
# End of read_eckert()
# ------------------------------------------------------------------------------

def bin_eckert(n=20):
    '''
    Bin the Eckert profiles into n mass bins

    Parameters
    ----------
    n : int
      number of bins
    '''
    # these are all assuming h_70!!
    m500g, m500, r500, rx, rho, rho_err = read_eckert()

    # number of points to mass bin
    n_m = n + 1 # bin edges, not centers -> +1
    m_bins = np.logspace(np.log10(m500g).min(), np.log10(m500g).max(), n_m)
    m_bin_idx = np.digitize(m500g, m_bins)

    r_min = 0.
    r_max = 3.
    # NUMBER of points in new profile
    n_r = 20
    r_ranges = np.empty((n_m-1, n_r), dtype=float)
    rho_med = np.empty((n_m-1, n_r), dtype=float)
    rho_std = np.empty((n_m-1, 2, n_r), dtype=float)
    m500_med = np.empty((n_m-1), dtype=float)
    r500_med = np.empty((n_m-1), dtype=float)
    m500c_med = np.empty((n_m-1), dtype=float)

    for idx_m, m_bin in enumerate(np.arange(1, len(m_bins))):
        idx_in_bin = (m_bin_idx == m_bin)
        m500_med[idx_m] = np.median(m500g[idx_in_bin])
        r500_med[idx_m] = np.median(r500[idx_in_bin])
        m500c_med[idx_m] = tools.radius_to_mass(np.median(r500[idx_in_bin]),
                                                500 * prms.cosmo.rho_crit *
                                                0.7**2)
        for idx in idx_in_bin.nonzero()[0]:
            # find maximum allowed rx range in bin
            if r_min < rx[idx].min():
                r_min = rx[idx].min()
            if r_max > rx[idx].max():
                r_max = rx[idx].max()

        # need to add small offsets for interpolation, can go wrong otherwise
        r_range = np.logspace(np.log10(r_min+0.001), np.log10(r_max-0.001), n_r)
        # need another loop to extrapolate function
        rho_new = np.empty((0, n_r))
        for idx in idx_in_bin.nonzero()[0]:
            f_rho = interp.interp1d(rx[idx], rho[idx])
            rho_new = np.concatenate([rho_new, f_rho(r_range).reshape(1,-1)],
                                     axis=0)

        r_ranges[idx_m] = r_range
        rho_med[idx_m] = np.median(rho_new, axis=0)
        rho_std[idx_m][0] = np.percentile(rho_new, 15, axis=0)
        rho_std[idx_m][1] = np.percentile(rho_new, 85, axis=0)

        # for t in rho_new:
        #     print t
        #     t[t <= 0] = np.nan
        #     if (t <= 0).sum() == 0:
        #         plt.plot(r_range, t, c='k')
        # plt.plot(r_range, rho_med[idx_m], label=r'median')
        # plt.fill_between(r_range, rho_std[idx_m][0], rho_std[idx_m][1],
        #                  color='r', alpha=0.2,
        #                  label=r'$15^\mathrm{th}-85^\mathrm{th}$ percentile')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.legend(loc='best')
        # plt.show()

    return r_ranges, rho_med, rho_std, m500_med, r500_med

# ------------------------------------------------------------------------------
# End of bin_eckert()
# ------------------------------------------------------------------------------

def prof_gas_hot(x, sl, a, b, m_sl, r500):
    '''beta profile'''
    profile = (1 + (x/a)**2)**(-3*b/2)
    mass = tools.m_h(profile[sl], x[sl] * r500)
    profile *= m_sl/mass

    return profile

def fit_croston():
    '''
    Fit profiles to the observations
    '''
    # these are all assuming h_70!
    m500g, r500, rx, rho, rho_err = read_croston()

    # we need to plug in the 0.7^2
    m500 = tools.radius_to_mass(r500, 500 * p.prms.cosmo.rho_crit * 0.7**2)

    a = np.empty((0,), dtype=float)
    b = np.empty((0,), dtype=float)
    m_sl = np.empty((0,), dtype=float)
    aerr = np.empty((0,), dtype=float)
    berr = np.empty((0,), dtype=float)
    for idx, prof in enumerate(rho):
        sl = ((prof > 0) & (rx[idx] >= 0.15) & (rx[idx] <= 1.))
        sl_500 = ((prof > 0) & (rx[idx] <= 1.))

        r = rx[idx]

        # Determine different profile masses
        mass = tools.m_h(prof[sl], r[sl] * r500[idx])
        m500gas = tools.m_h(prof[sl_500], r[sl_500] * r500[idx])

        # Need to perform the fit for [0.15,1] r500c -> mass within this region
        # need to match
        sl_fit = np.ones(sl.sum(), dtype=bool)
        popt, pcov = opt.curve_fit(lambda r, a, b: \
                                   prof_gas_hot(r, sl_fit, a, b,
                                                mass, r500[idx]),
                                   r[sl], prof[sl],
                                   sigma=rho_err[idx][sl],
                                   bounds=([0, 0], [1, 5]))

        # print('f_gas,500c_actual :', m500g[idx] / m500[idx])
        # print('f_gas,500c_fitted :', m500gas / m500[idx])
        # print('m_gas,500c_fitted / m_gas,500c_actual', m500gas / m500g[idx])
        # print('-------------------')
        plt.plot(r, prof, label='obs')
        plt.plot(r, prof_gas_hot(r, sl_500, popt[0], popt[1], # , popt[2],
                                 m500gas, r500[idx]),
                 label='fit')
        plt.title('%i'%idx)
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.show()

        # Final fit will need to reproduce the m500gas mass
        a = np.append(a, popt[0])
        b = np.append(b, popt[1])
        m_sl = np.append(m_sl, m500gas)
        aerr = np.append(aerr, np.sqrt(np.diag(pcov))[0])
        berr = np.append(berr, np.sqrt(np.diag(pcov))[1])

    return a, aerr, b, berr, m_sl, m500g, r500  # c, cerr, m500g


def fit_eckert():
    # these are all assuming h_70!!
    rx, rho, rho_std, m500g, r500 = bin_eckert(n=20)
    rho_err = np.mean(rho_std, axis=1)
    # m500g, m500, r500, rx, rho, rho_err = read_eckert()

    # we need to plug in the 0.7^2
    m500 = tools.radius_to_mass(r500, 500 * p.prms.cosmo.rho_crit * 0.7**2)

    a = np.empty((0,), dtype=float)
    b = np.empty((0,), dtype=float)
    m_sl = np.empty((0,), dtype=float)
    aerr = np.empty((0,), dtype=float)
    berr = np.empty((0,), dtype=float)

    for idx, prof in enumerate(rho):
        sl = ((prof > 0) & (rx[idx] >= 0.15) & (rx[idx] <= 1.))
        sl_500 = ((prof > 0) & (rx[idx] <= 1.))

        r = rx[idx]

        # Determine different profile masses
        mass = tools.m_h(prof[sl], r[sl] * r500[idx])
        m500gas = tools.m_h(prof[sl_500], r[sl_500] * r500[idx])

        # Need to perform the fit for [0.15,1] r500c -> mass within this region
        # need to match
        sl_fit = np.ones(sl.sum(), dtype=bool)
        popt, pcov = opt.curve_fit(lambda r, a, b: \
                                   prof_gas_hot(r, sl_fit, a, b,
                                                mass, r500[idx]),
                                   r[sl], prof[sl],
                                   sigma=rho_err[idx][sl],
                                   bounds=([0, 0], [1, 5]))

        # get gas fractions for binned Eckert profiles
        prof_fit = prof_gas_hot(r, sl_500, popt[0], popt[1],
                                m500gas, r500[idx])
        m500gas_fit = tools.m_h(prof_fit[sl_500], r[sl_500] * r500[idx])
        print("{:.6f}          {:.6f}".format(np.log10(m500)[idx], m500gas_fit / m500[idx]))

        # print('f_gas,500c_actual :', m500g[idx] / m500[idx])
        # print('f_gas,500c_fitted :', m500gas / m500[idx])
        # print('m_gas,500c_fitted / m_gas,500c_actual', m500gas / m500g[idx])
        # print('-------------------')
        # plt.clf()
        # plt.plot(r, prof, label='obs')
        # plt.plot(r, prof_gas_hot(r, sl_500, popt[0], popt[1], # popt[2],
        #                          m500gas, r500[idx]),
        #          label=r'$r_c={:.2f}, \beta={:.2f}$'.format(popt[0],popt[1]))
        # plt.title('{:d}: gas fraction = {:.2f}'.format(idx, m500gas / m500[idx]))
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.legend()
        # plt.show()
        # plt.pause(1)
        
        # Final fit will need to reproduce the m500gas mass
        a = np.append(a, popt[0])
        b = np.append(b, popt[1])
        m_sl = np.append(m_sl, m500gas)
        aerr = np.append(aerr, np.sqrt(np.diag(pcov))[0])
        berr = np.append(berr, np.sqrt(np.diag(pcov))[1])

    return a, aerr, b, berr, m_sl, m500g, r500 # c, cerr, m500g

# ------------------------------------------------------------------------------
# End of fit_eckert()
# ------------------------------------------------------------------------------

def convert_hm(r200m=True):
    '''
    Save fit parameters and masses for both m500c & m200m. All of the saved
    data assumes h=0.7, so convert theoretical fits accordingly!!!

    Running this redoes the whole analysis, i.e. reading in all the data
    and analyzing it.

    '''
    # All of these assume h=0.7
    rcc, rccerr, bc, bcerr, mslc, mgc, r500c = fit_croston()
    rce, rceerr, be, beerr, msle, mge, r500e = fit_eckert()

    # we thus need to convert our critical density as well
    m500cc = tools.radius_to_mass(r500c, 500 * prms.cosmo.rho_crit * 0.7**2)
    m500ce = tools.radius_to_mass(r500e, 500 * prms.cosmo.rho_crit * 0.7**2)

    # # Get masses from inverted mgas-m500c relation, these assume h=0.7!!!!
    # m500cgc = mgas_to_m500c(mgc, p.prms.cosmo)
    # m500cge = mgas_to_m500c(mge, p.prms.cosmo)

    # we thus need to convert our critical density as well
    r500cc = tools.mass_to_radius(m500cc, 500 * prms.cosmo.rho_crit * 0.7**2)
    r500ce = tools.mass_to_radius(m500ce, 500 * prms.cosmo.rho_crit * 0.7**2)

    m_range = np.logspace(np.log10(np.min(m500ce)),
                          np.log10(np.max(m500cc)), 100)

    data_croston = {'rc': rcc,
                    'rcerr': rccerr,
                    'b': bc,
                    'berr': bcerr,
                    # 'm500c_gas': m500cgc,
                    'm500c': m500cc}
    data_eckert = {'rc': rce,
                   'rcerr': rceerr,
                   'b': be,
                   'berr': beerr,
                   # 'm500c_gas': m500cge,
                   'm500c': m500ce}

    with open('data/croston_500.p', 'wb') as f:
        pickle.dump(data_croston, f)

    with open('data/eckert_500.p', 'wb') as f:
        pickle.dump(data_eckert, f)

    if r200m:
        # assume h=1, since we already give correctly scaled h=0.7 values to the fit
        # hence we do not need to rescale the mass for c_correa
        m200mc = np.array([tools.m500c_to_m200m_duffy(m, prms.cosmo.rho_crit * 0.7**2,
                                                      prms.cosmo.rho_m * 0.7**2)
                           for m in m500cc])
        m200me = np.array([tools.m500c_to_m200m_duffy(m, prms.cosmo.rho_crit * 0.7**2,
                                                      prms.cosmo.rho_m * 0.7**2)
                           for m in m500ce])
        r200mc = tools.mass_to_radius(m200mc, 200 * prms.cosmo.rho_m * 0.7**2)
        r200me = tools.mass_to_radius(m200me, 200 * prms.cosmo.rho_m * 0.7**2)

        rcc *= r500cc/r200mc
        rccerr *= r500cc/r200mc
        rce *= r500ce/r200me
        rceerr *= r500ce/r200me

        data_croston = {'rc': rcc,
                        'rcerr': rccerr,
                        'b': bc,
                        'berr': bcerr,
                        'm200m': m200mc}
        data_eckert = {'rc': rce,
                       'rcerr': rceerr,
                       'b': be,
                       'berr': beerr,
                       'm200m': m200me}

        with open('data/croston_200.p', 'wb') as f:
            pickle.dump(data_croston, f)

        with open('data/eckert_200.p', 'wb') as f:
            pickle.dump(data_eckert, f)

# ------------------------------------------------------------------------------
# End of convert_hm()
# ------------------------------------------------------------------------------

def f_gas(m, log10mt, a, fgas_500c_max, cosmo, f_c=0.86, sigma_lnc=0.0,
          z=0., norm=None, **kwargs):
    '''
    Return the f_gas(m500c) relation for m. The relation cannot exceed 
    f_b,500c - f_stars,500c

    This function assumes h=0.7 for everything!

    Parameters
    ----------
    m : array [M_sun / h_70]
      values of m500c to compute f_gas for
    log10mt : float
      the turnover mass for the relation in log10([M_sun/h_70])
    a : float
      the strength of the transition
    fgas_500c_max : interpolator or function of z and m
      maximum gas fraction for which fgas_500c + fstars_500c = f_bar
    cosmo : hmf.cosmo.Cosmology object
      relevant cosmological parameters

    Returns
    -------
    f_gas : array [h_70^(-3/2)]
      gas fraction at r500c for m
    '''
    x = np.log10(m) - log10mt

    if norm is None:
        norm = (cosmo.omegab/cosmo.omegam)
    # gas fractions
    fgas_fit = norm * (0.5 * (1 + np.tanh(x / a)))

    coords = inp_interp.arrays_to_coords(z, np.log10(m))
    if np.size(z) > 1:
        raise ValueError("redshift dependence not yet implemented")
    fgas_500c_mx = fgas_500c_max(f_c=f_c, sigma_lnc=sigma_lnc)(coords).reshape(m.shape)
    
    # gas fractions that will cause halo to exceed cosmic baryon fraction
    cb_exceeded = (fgas_fit >= fgas_500c_mx)
    fgas_fit[cb_exceeded] = fgas_500c_mx[cb_exceeded]

    return fgas_fit

# ------------------------------------------------------------------------------
# End of f_gas()
# ------------------------------------------------------------------------------

# def logmgas(logm500c, cosmo, q=50, bias=False, fstar_500c=0.):
#     return np.log10(np.power(10, logm500c) * f_gas(np.power(10, logm500c),
#                                                    cosmo=cosmo,
#                                                    fstar_500c=fstar_500c,
#                                                    **f_gas_prms(cosmo, q=q, bias=bias)))

# def mgas_to_m500c(mgas500, cosmo, q=50, fstar_500c=0.):
#     '''
#     Invert f_gas relation to find m500c belonging to mgas
#     '''
#     logmgas_prms = {'cosmo': cosmo, 'q': q}
#     f1 = tools.inverse(logmgas, start=1., **logmgas_prms)
#     m500c = np.ones_like(mgas500)
#     # do everything logarithmically for speed, results in errors at .01% level
#     for idx, m in enumerate(mgas500):
#         m500c[idx] = np.power(10, f1(np.log10(m), fstar_500c=f_star500c, **logmgas_prms))

#     return m500c

# # ------------------------------------------------------------------------------
# # End of mgas_to_m500c()
# # ------------------------------------------------------------------------------
    
def f_gas_prms(cosmo, z=0., q=50, f_c=0.86, sigma_lnc=0.0, bias=False):
    '''
    Compute best fit parameters to the f_gas(m500c) relation with both f_gas and
    m500c assuming h=0.7
    '''
    if bias == False:
        m500_obs, f_obs = np.loadtxt(ddir +
                                     'data_mccarthy/gas/m500_fgas_hydrostatic.dat',
                                     unpack=True)

    else:
        fname = 'data_mccarthy/gas/m500_fgas_bias_{}_corrected.dat'.format(str(bias).replace(".", "p"))
        m500_obs, f_obs = np.loadtxt(ddir + fname,
                                     unpack=True)
        

    # do not include Eckert data
    m500_obs = m500_obs[:185]
    f_obs = f_obs[:185]

    n_m = 15

    m_bins = np.logspace(m500_obs.min(), m500_obs.max(), n_m)
    m = tools.bins2center(m_bins)

    if np.size(z) > 1:
        raise ValueError("redshift dependence not yet implemented")
    # get the coordinate arrays for interpolation of the maximum stellar fractions
    coords = inp_interp.arrays_to_coords(z, np.log10(m))
    fgas_500c_max = inp_interp.fgas_500c_max_interp

    # m_bins is in Hubble units
    m_bin_idx = np.digitize(10**(m500_obs), m_bins)

    f_q = np.array([np.percentile(f_obs[m_bin_idx == m_bin], q)
                      for m_bin in np.arange(1, len(m_bins))])

    f_gas_fit = lambda m, log10mt, a: f_gas(m=m, log10mt=log10mt,
                                            a=a, cosmo=cosmo,
                                            f_c=f_c, sigma_lnc=sigma_lnc,
                                            fgas_500c_max=fgas_500c_max)

    fqopt, fqcov = opt.curve_fit(f_gas_fit, m[m>1e14], f_q[m>1e14],
                                 bounds=([10, 0],
                                         [20, 20]))

    fq_prms = {"log10mt": fqopt[0],
               "a": fqopt[1],
               "fgas_500c_max": fgas_500c_max}

    return fq_prms

# ------------------------------------------------------------------------------
# End of f_gas_prms()
# ------------------------------------------------------------------------------

def f_stars(m200m, comp='all'):
    '''
    Return the stellar fraction as a function of halo mass as found by
    Zu & Mandelbaum (2015).

    For m200m < 1e10 M_sun/h we return f_stars=0
    For m200m > 1e16 M_sun/h we return f_stars=1.41e-2

    THIS FUNCTION ASSUMES h=0.7 FOR EVERYTHING!!

    Parameters
    ----------
    m200m : (m,) array or float [M_sun/h_70]
      halo mass with respect to mean density of universe

    Returns
    -------
    f_stars : (m,) array [h_70^(-1)]
      total stellar fraction for the halo mass
    '''
    comp_options = ['all', 'cen', 'sat']
    if comp not in comp_options:
        raise ValueError('comp needs to be in {}'.format(comp_options))

    # m_h is in Hubble units
    # all the fractions have assumed h=0.7
    m_h, f_stars, f_cen, f_sat = np.loadtxt(ddir +
                                            'data_mccarthy/stars/StellarFraction-Mh.txt',
                                            unpack=True)

    # we need to convert the halo mass to h=0.7 as well
    m_h = m_h / 0.7

    if comp == 'all':
        f_stars_interp = interp.interp1d(m_h, f_stars, bounds_error=False,
                                         fill_value=(0,f_stars[-1]))
    elif comp == 'cen':
        f_stars_interp = interp.interp1d(m_h, f_cen, bounds_error=False,
                                         fill_value=(0,f_cen[-1]))
    else:
        f_stars_interp = interp.interp1d(m_h, f_sat, bounds_error=False,
                                         fill_value=(0,f_sat[-1]))

    return f_stars_interp(m200m)

# ------------------------------------------------------------------------------
# End of f_stars()
# ------------------------------------------------------------------------------

def f_stars_interp(comp='all'):
    '''
    Return the stellar fraction interpolator as a function of halo mass as found 
    by Zu & Mandelbaum (2015).

    For m200m < 1e10 M_sun/h we return f_stars=0
    For m200m > 1e16 M_sun/h we return f_stars=1.41e-2

    THIS INTERPOLATOR ASSUMES h=0.7 FOR EVERYTHING!!

    Returns
    -------
    f_stars : (m,) array [h_70^(-1)]
      total stellar fraction for the halo mass
    '''
    comp_options = ['all', 'cen', 'sat']
    if comp not in comp_options:
        raise ValueError('comp needs to be in {}'.format(comp_options))

    # m_h is in Hubble units
    # all the fractions have assumed h=0.7
    m_h, f_stars, f_cen, f_sat = np.loadtxt(ddir +
                                              'data_mccarthy/stars/StellarFraction-Mh.txt',
                                              unpack=True)

    # we need to convert the halo mass to h=0.7 as well
    m_h = m_h / 0.7

    if comp == 'all':
        f_stars_interp = interp.interp1d(m_h, f_stars, bounds_error=False,
                                         fill_value=(0,f_stars[-1]))
    elif comp == 'cen':
        f_stars_interp = interp.interp1d(m_h, f_cen, bounds_error=False,
                                         fill_value=(0,f_cen[-1]))
    else:
        f_stars_interp = interp.interp1d(m_h, f_sat, bounds_error=False,
                                         fill_value=(0,f_sat[-1]))

    return f_stars_interp

# ------------------------------------------------------------------------------
# End of f_stars_interp()
# ------------------------------------------------------------------------------

def fit_prms(x=500, q_rc=50, q_beta=50):
    '''
    Return observational beta profile fit parameters
    '''
    if x == 500:
        with open('data/croston_500.p', 'rb') as f:
            data_c = pickle.load(f)
        with open('data/eckert_500.p', 'rb') as f:
            data_e = pickle.load(f)

        m500c_e = data_e['m500c']
        m500c_c = data_c['m500c']
        m500c = np.append(m500c_e, m500c_c)

    elif x == 200:
        with open('data/croston_200.p', 'rb') as f:
            data_c = pickle.load(f)
        with open('data/eckert_200.p', 'rb') as f:
            data_e = pickle.load(f)

        m200m_e = data_e['m200m']
        m200m_c = data_c['m200m']
        m200m = np.append(m200m_e, m200m_c)


    else:
        raise ValueError('x must be 500 or 200')

    rc_e = data_e['rc']
    rc_c = data_c['rc']
    rc = np.append(rc_e, rc_c)

    b_e = data_e['b'] # equal to literature standard
    b_c = data_c['b']
    beta = np.append(b_e, b_c)

    return np.percentile(rc, q_rc), np.percentile(beta, q_beta)

# ------------------------------------------------------------------------------
# End of fit_prms()
# ------------------------------------------------------------------------------

def plot_profiles_paper(prms=prms):
    with open('data/croston.p', 'rb') as f:
        data_c = pickle.load(f)
    with open('data/eckert.p', 'rb') as f:
        data_e = pickle.load(f)

    # with open('data/croston_500.p', 'rb') as f:
    #     data_hm_c = pickle.load(f)
    # with open('data/eckert_500_all.p', 'rb') as f:
    #     data_hm_e = pickle.load(f)

    # pl.set_style()
    fig = plt.figure(figsize=(20,9))
    ax1 = fig.add_axes([0.1,0.1,0.4,0.8])
    ax2 = fig.add_axes([0.5,0.1,0.4,0.8])
    ax3 = fig.add_axes([0.9,0.1,0.05,0.8])
    # fig = plt.figure(figsize=(18,8))
    # ax1 = fig.add_axes([0.1,0.1,0.266,0.8])
    # ax2 = fig.add_axes([0.366,0.1,0.266,0.8])
    # ax3 = fig.add_axes([0.632,0.1,0.266,0.8])

    rx_c = np.array(data_c['rx'])
    rx_e = np.array(data_e['rx'])

    # these go as h_70^(1/2)
    rho_c = np.array(data_c['rho'])
    rho_e = np.array(data_e['rho'])

    # this now goes as h_70^2
    rho_crit = prms.cosmo.rho_crit * 0.7**2


    # now load halo model info
    m500c_c = np.array(data_c['m500c'])
    m500c_e = np.array(data_e['m500c'])

    # mass bins for the profiles
    m_bins = np.arange(12, 16, 0.5)
    m = tools.bins2center(m_bins)
    # create colorbar for massbins
    cmap = pl.discrete_cmap(m.shape[0], 'plasma')

    # define croston bins
    m_bin_idx_c = np.digitize(np.log10(m500c_c), m_bins)

    lines_c = []
    for idx_m, m_bin in enumerate(np.arange(1, len(m_bins))):
        idx_in_bin = (m_bin_idx_c == m_bin)
        for r, prof in zip(rx_c[idx_in_bin], rho_c[idx_in_bin]):
            l, = ax1.plot(r, prof / rho_crit, ls='-', lw=1, c=cmap(idx_m))
            lines_c.append(l)

    segs = [np.array([l.get_xdata(), l.get_ydata()]).T for l in lines_c]
    lws = [l.get_lw() for l in lines_c]
    lss = [l.get_ls() for l in lines_c]
    cs = [l.get_c() for l in lines_c]

    lc_c = LineCollection(segs, linewidths=lws, linestyles=lss, colors=cs, cmap=cmap)
    lc_c.set_array(m)

    # define the bins and normalize
    norm = mpl.colors.BoundaryNorm(m_bins, cmap.N)

    axcb = fig.colorbar(lc_c, cax=ax3,
                        norm=norm, ticks=m_bins[:-1], boundaries=m_bins[:-1])
    axcb.set_label(r'$\log_{10}(m_\mathrm{500c}) \, [h_{70}^{-1} \, \mathrm{M_\odot}]$',
                   rotation=270, labelpad=50)

    # define eckert bins
    m_bin_idx_e = np.digitize(np.log10(m500c_e), m_bins)

    lines_e = []
    for idx_m, m_bin in enumerate(np.arange(1, len(m_bins))):
        idx_in_bin = (m_bin_idx_e == m_bin)
        for r, prof in zip(rx_e[idx_in_bin], rho_e[idx_in_bin]):
            l, = ax2.plot(r, prof / rho_crit, ls='-', lw=1, c=cmap(idx_m))
            lines_e.append(l)

    # segs = [np.array([l.get_xdata(), l.get_ydata()]).T for l in lines_e]
    # lws = [l.get_lw() for l in lines_e]
    # lss = [l.get_ls() for l in lines_e]
    # cs = [l.get_c() for l in lines_e]

    # lc_e = LineCollection(segs, linewidths=lws, linestyles=lss, colors=cs)
    # lc_e.set_array(m)

    ax1.set_xlim([1e-3, 1e1])
    ax1.set_ylim([1e1, 1e5])
    ticks = ax1.get_xticklabels()
    ticks[0].set_visible(False)

    ax1.xaxis.set_tick_params(pad=8)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$r/r_\mathrm{500c}$', labelpad=-5)
    ax1.set_ylabel(r'$\rho(r) / \rho_\mathrm{c} \, [h_{70}^{-3/2}]$')
    ax1.set_title(r'Croston+08')

    ax2.set_xlim([1e-3, 1e1])
    ax2.set_ylim([1e1, 1e5])
    ticks = ax2.get_xticklabels()
    ticks[-5].set_visible(False)

    ax2.xaxis.set_tick_params(pad=8)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_yticklabels([])
    ax2.set_xlabel(r'$r/r_\mathrm{500c}$', labelpad=-5)
    text = ax2.set_title(r'Eckert+16')
    title_props = text.get_fontproperties()

    plt.savefig('figures/obs_profiles.pdf', transparent=True,
                bbox_inches='tight')
    plt.show()
# ------------------------------------------------------------------------------
# End of plot_profiles_paper()
# ------------------------------------------------------------------------------

def plot_profiles_fit_paper():
    with open('data/croston.p', 'rb') as f:
        data_c = pickle.load(f)
    with open('data/eckert.p', 'rb') as f:
        data_e = pickle.load(f)

    with open('data/croston_500.p', 'rb') as f:
        data_hm_c = pickle.load(f)
    with open('data/eckert_500.p', 'rb') as f:
        data_hm_e = pickle.load(f)

    a_e, aerr_e, b_e, berr_e, msl_e, m500g_e, r500_e = fit_eckert()
    a_c, aerr_c, b_c, berr_c, msl_c, m500g_c, r500_c = fit_croston()

    fig = plt.figure(figsize=(20,9))
    ax1 = fig.add_axes([0.1,0.1,0.4,0.4])
    ax2 = fig.add_axes([0.5,0.1,0.4,0.4])
    ax3 = fig.add_axes([0.1,0.5,0.4,0.4])
    ax4 = fig.add_axes([0.5,0.5,0.4,0.4])
    ax5 = fig.add_axes([0.9,0.1,0.05,0.8])

    rx_c = np.array(data_c['rx'])
    rx_e = np.array(data_e['rx'])

    # these go as h_70^(1/2)
    r500_c = np.array(data_c['r500'])
    r500_e = np.array(data_e['r500'])

    rho_c = np.array(data_c['rho'])
    rho_e = np.array(data_e['rho'])

    # now load halo model info
    m500c_c = np.array(data_hm_c['m500c'])
    m500c_e = np.array(data_hm_e['m500c'])

    # mass bins for the profiles
    m_bins = np.arange(12, 16, 0.5)
    m = tools.bins2center(m_bins)
    # create colorbar for massbins
    cmap = pl.discrete_cmap(m.shape[0], 'plasma')

    # define croston bins
    m_bin_idx_c = np.digitize(np.log10(m500c_c), m_bins)

    lines_c = []
    m500_fit_c = np.empty_like(r500_c, dtype=float)
    m500_prof_c = np.empty_like(r500_c, dtype=float)
    for idx_m, m_bin in enumerate(np.arange(1, len(m_bins))):
        idx_in_bin = (m_bin_idx_c == m_bin)
        for idx, r, prof in zip(np.arange(len(r500_c))[idx_in_bin],
                                rx_c[idx_in_bin],
                                rho_c[idx_in_bin]):
            sl = ((prof > 0) & (r <= 1.))
            fit = prof_gas_hot(r, sl, a_c[idx], b_c[idx], msl_c[idx], r500_c[idx])
            l, = ax3.plot(r[1:], (prof[1:] - fit[1:]) / fit[1:], ls='-', lw=0.5, c=cmap(idx_m))
            lines_c.append(l)

            cum_mass_fit = np.array([tools.m_h(fit[:i], r[:i] * r500_c[idx])
                                     for i in np.arange(1, r.shape[0])])
            cum_mass_prof = np.array([tools.m_h(prof[:i], r[:i] * r500_c[idx])
                                      for i in np.arange(1, r.shape[0])])
            m500_fit_c[idx] =  tools.m_h(fit, r * r500_c[idx])
            m500_prof_c[idx] = tools.m_h(prof, r * r500_c[idx])

            ax1.plot(r[1:], (cum_mass_prof - cum_mass_fit) / m500_fit_c[idx],
                     ls='-', lw=0.5, c=cmap(idx_m))

    segs = [np.array([l.get_xdata(), l.get_ydata()]).T for l in lines_c]
    lws = [l.get_lw() for l in lines_c]
    lss = [l.get_ls() for l in lines_c]
    cs = [l.get_c() for l in lines_c]

    lc_c = LineCollection(segs, linewidths=lws, linestyles=lss, colors=cs, cmap=cmap)
    lc_c.set_array(m)

    # define the bins and normalize
    norm = mpl.colors.BoundaryNorm(m_bins, cmap.N)

    axcb = fig.colorbar(lc_c, cax=ax5,
                        norm=norm, ticks=m_bins[:-1], boundaries=m_bins[:-1])
    axcb.set_label(r'$\log_{10}(m_\mathrm{500c}) \, [h_{70}^{-1} \, \mathrm{M_\odot}]$',
                   rotation=270, labelpad=50)

    ax3.set_ylim([-0.5, 0.5])
    ax3.minorticks_on()
                       
    # ticks = ax3.get_yticklabels()
    # ticks[-6].set_visible(False)

    ax3.set_xscale('log')
    ax3.set_ylabel(r'$\rho_\mathrm{obs}(r)/\rho_\mathrm{fit}(r) - 1$')
    ax3.set_xticklabels([])
    ax3.set_title(r'Croston+08')

    ax1.set_ylim([-0.1,0.1])
    ax1.minorticks_on()
    ticks = ax1.get_yticklabels()
    ticks[-1].set_visible(False)

    ax1.xaxis.set_tick_params(pad=8)
    ax1.set_xscale('log')
    ax1.set_xlabel(r'$r/r_\mathrm{500c}$', labelpad=-8)
    ax1.set_ylabel(r'$\frac{m_\mathrm{obs}(<r) - m_\mathrm{fit}(<r)}{m_\mathrm{fit,500c}}$')

    # get median binned profiles
    r_med, rho_med, rho_std, m500_med, r500_med = bin_eckert()

    # define croston bins
    m_bin_idx_e = np.digitize(np.log10(m500c_e), m_bins)

    lines_e = []
    m500_fit_e = np.empty_like(r500_med, dtype=float)
    m500_prof_e = np.empty_like(r500_med, dtype=float)
    for idx_m, m_bin in enumerate(np.arange(1, len(m_bins))):
        idx_in_bin = (m_bin_idx_e == m_bin)
        for idx, r, prof in zip(np.arange(len(r500_med))[idx_in_bin],
                                r_med[idx_in_bin],
                                rho_med[idx_in_bin]):
            sl = ((prof > 0) & (r <= 1.))
            fit = prof_gas_hot(r, sl, a_e[idx], b_e[idx], msl_e[idx], r500_med[idx])
            l, = ax4.plot(r[1:], (prof[1:] - fit[1:]) / fit[1:], ls='-', lw=0.5, c=cmap(idx_m))
            lines_e.append(l)

            cum_mass_fit = np.array([tools.m_h(fit[:i], r[:i] * r500_e[idx])
                                     for i in np.arange(1, r.shape[0])])
            cum_mass_prof = np.array([tools.m_h(prof[:i], r[:i] * r500_e[idx])
                                      for i in np.arange(1, r.shape[0])])
            m500_fit_e[idx] = tools.m_h(fit, r * r500_e[idx])
            m500_prof_e[idx] = tools.m_h(prof, r * r500_e[idx])

            ax2.plot(r[1:], (cum_mass_prof - cum_mass_fit) / m500_fit_e[idx],
                     ls='-', lw=0.5, c=cmap(idx_m))

    ax4.set_ylim([-0.5, 0.5])
    ax4.minorticks_on()
    ax4.set_xscale('log')
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])
    ax4.set_title(r'Eckert+16')


    ax2.set_ylim([-0.1,0.1])
    ax2.minorticks_on()
    # ticks = ax2.get_xticklabels()
    # ticks[-3].set_visible(False)

    ax2.xaxis.set_tick_params(pad=8)
    ax2.set_xscale('log')
    ax2.set_yticklabels([])
    ax2.set_xlabel(r'$r/r_\mathrm{500c}$', labelpad=-8)
    plt.savefig('figures/obs_profiles_fit.pdf', transparent=True, bbox_inches='tight')
    plt.show()

# ------------------------------------------------------------------------------
# End of plot_profiles_fit()
# ------------------------------------------------------------------------------

def plot_beta_prof_fit_prms_paper(prms=prms):
    with open('data/croston_500.p', 'rb') as f:
        data_c = pickle.load(f)
    with open('data/eckert_500.p', 'rb') as f:
        data_e = pickle.load(f)

    with open('data/croston_200.p', 'rb') as f:
        data200m_c = pickle.load(f)
    with open('data/eckert_200.p', 'rb') as f:
        data200m_e = pickle.load(f)

    rc_e = data_e['rc']
    rc_c = data_c['rc']
    rc_e_err = data_e['rcerr']
    rc_c_err = data_c['rcerr']

    # one of the points blows up in the error
    rc_e[rc_e < 1e-3] = 0
    rc_e_err[rc_e < 1e-3] = 0

    rc = np.append(rc_e, rc_c)
    rc_err = np.append(rc_e_err, rc_c_err)

    b_e = data_e['b']
    b_c = data_c['b']
    b_e_err = data_e['berr']
    b_c_err = data_c['berr']
    beta = np.append(b_e, b_c)
    beta_err = np.append(b_e_err, b_c_err)

    m500c_e = data_e['m500c']
    m500c_c = data_c['m500c']
    m500c = np.append(m500c_e, m500c_c)

    # m200m_e = data200m_e['m200m']
    # m200m_c = data200m_c['m200m']
    # m200m = np.append(m200m_e, m200m_c)
    # m200m = np.array([tools.m500c_to_m200m(m500c.min(), prms.rho_crit, prms.rho_m, prms.h),
    #                   tools.m500c_to_m200m(m500c.max(), prms.rho_crit, prms.rho_m, prms.h)])

    # pdb.set_trace()
    
    pl.set_style('line')

    # fig = plt.figure(figsize=(30,9))
    # grid = plt.GridSpec(1,2, hspace=0.2)
    # ax_rc = fig.add_subplot(grid[0,0])
    # ax_beta = fig.add_subplot(grid[0,1])

    fig = plt.figure(figsize=(10,9))
    ax_rc = fig.add_subplot(111)
    
    ax_rc.errorbar(m500c_c, rc_c, yerr=rc_c_err, lw=0, elinewidth=1, marker='s',
                   label='Croston+08')
    ax_rc.errorbar(m500c_e, rc_e, yerr=rc_e_err, lw=0, elinewidth=1, marker='o',
                label='Eckert+16')
    ax_rc.axhline(y=np.median(rc), ls='-', c='k')
    ax_rc.axhspan(np.percentile(rc, 15), np.percentile(rc, 85),
               facecolor='k', alpha=0.3)
    ax_rc.annotate('median', xy=(1.05 * m500c.min(), 1.05 * np.median(rc)))

    # ##########################################################################
    # # Get shadow twiny instance for ratio plot, to also have m200m
    # # need it in this order to get correct yticks with log scale, since
    # # twin instance seems to mess things up...
    # axs = ax.twiny()
    # axs.plot([m200m.min(), m200m.max()], [m200m.min(), m200m.max()])
    # axs.cla()

    ax_rc.set_xlim(m500c.min(), m500c.max())
    ax_rc.minorticks_on()
    # axs.set_xlim(m200m.min(), m200m.max())

    # ax.set_ylim(0,0.5)
    ax_rc.set_xscale('log')
    # axs.set_xscale('log')

    ax_rc.xaxis.set_tick_params(pad=10)
    # axs.xaxis.set_tick_params(pad=5)

    ax_rc.set_xlabel(r'$m_{\mathrm{500c}}\, [h_{70}^{-1} \, \mathrm{M_\odot}]$')
    # axs.set_xlabel(r'$m_{\mathrm{200m}}\, [h_{70}^{-1} \, \mathrm{M_\odot}]$',
    #                labelpad=15)
    ax_rc.set_ylabel('$r_\mathrm{c}/r_{\mathrm{500c}}$')
    leg = ax_rc.legend(loc='best', frameon=True, framealpha=0.8)
    leg.get_frame().set_linewidth(0.0)

    plt.savefig('figures/obs_rc_fit.pdf', transparent=True, bbox_inches='tight')

    # ##########################################################################
    # Plot beta
    plt.clf()
    fig = plt.figure(figsize=(10,9))
    ax_beta = fig.add_subplot(111)

    ax_beta.errorbar(m500c_c, b_c, yerr=b_c_err, lw=0, elinewidth=1, marker='s',
                     label='Croston+08')
    ax_beta.errorbar(m500c_e, b_e, yerr=b_e_err, lw=0, elinewidth=1, marker='o',
                     label='Eckert+16')
    ax_beta.axhline(y=np.median(beta), ls='-', c='k')
    ax_beta.axhspan(np.percentile(beta, 15), np.percentile(beta, 85),
               facecolor='k', alpha=0.3)
    ax_beta.annotate('median', xy=(1.05 * m500c.min(), 1.05 * np.median(beta)))

    # ##########################################################################
    # # Get shadow twiny instance for ratio plot, to also have m200m
    # # need it in this order to get correct yticks with log scale, since
    # # twin instance seems to mess things up...
    # axs = ax_beta.twiny()
    # axs.plot([m200m.min(), m200m.max()], [m200m.min(), m200m.max()])
    # axs.cla()

    ax_beta.set_xlim(m500c.min(), m500c.max())
    ax_beta.minorticks_on()
    # axs.set_xlim(m200m.min(), m200m.max())

    # ax_beta.set_ylim(0.04, 1.7)
    ax_beta.set_xscale('log')
    # axs.set_xscale('log')

    ax_beta.xaxis.set_tick_params(pad=10)
    # axs.xaxis.set_tick_params(pad=5)

    ax_beta.set_xlabel('$m_{\mathrm{500c}}\, [h_{70}^{-1} \, \mathrm{M_\odot}]$')
    # axs.set_xlabel(r'$m_{\mathrm{200m}}\, [h_{70}^{-1} \, \mathrm{M_\odot}]$',
    #                labelpad=15)
    ax_beta.set_ylabel(r'$\beta$')
    # ax_beta.legend(loc='best')

    leg = ax_beta.legend(loc='best', frameon=True, framealpha=0.8)
    leg.get_frame().set_linewidth(0.0)

    plt.savefig('figures/obs_beta_fit.pdf', transparent=True, bbox_inches='tight')
    # plt.savefig('figures/obs_rc+beta_fit.pdf', transparent=True, bbox_inches='tight')

# ------------------------------------------------------------------------------
# End of plot_beta_prof_fit_prms_paper()
# ------------------------------------------------------------------------------

def plot_masses_hist_paper():
    with open('data/croston.p', 'rb') as f:
        data_c = pickle.load(f)
    with open('data/eckert.p', 'rb') as f:
        data_e = pickle.load(f)

    m500c_e = data_e['m500c']
    m500c_c = data_c['m500c']
    m500c = np.append(m500c_e, m500c_c)

    # m200m_e = data200m_e['m200m']
    # m200m_c = data200m_c['m200m']
    # m200m = np.append(m200m_e, m200m_c)

    # mn = np.min(np.hstack([m200m_c, m200m_e]))
    # mx = np.max(np.hstack([m200m_c, m200m_e]))
    mn = np.min(np.hstack([m500c_c, m500c_e]))
    mx = np.max(np.hstack([m500c_c, m500c_e]))

    bins = np.logspace(np.log10(mn), np.log10(mx), 10)

    pl.set_style('line')
    fig = plt.figure(figsize=(10,9))
    ax = fig.add_subplot(111)

    ax.hist([m500c_c, m500c_e], bins=bins, stacked=True, histtype='barstacked',
            label=['Croston+08', 'Eckert+16'])

    # axs = ax.twiny()
    # axs.plot([m200m.min(), m200m.max()], [0, 0])
    # axs.cla()

    ax.set_xlim(m500c.min(), m500c.max())
    # axs.set_xlim(m500c.min(), m500c.max())
    ax.set_ylim(0,41)

    ax.minorticks_on()
    ax.set_xlabel(r'$m_{\mathrm{500c}}\, [h_{70}^{-1} \, \mathrm{M_\odot}]$')
    # axs.set_xlabel(r'$m_\mathrm{200m} \, [h_{70}^{-1} \, \mathrm{M_\odot}]$',
    #                labelpad=15)

    ax.set_ylabel(r'Frequency')

    ax.set_xscale('log')
    # axs.set_xscale('log')

    ax.xaxis.set_tick_params(pad=10)
    # axs.xaxis.set_tick_params(pad=5)

    ax.legend(loc='best')

    plt.savefig('figures/obs_masses_hist_stacked.pdf',
                transparent=True, bbox_inches='tight')
    plt.show()

# ------------------------------------------------------------------------------
# End of plot_masses_hist_paper()
# ------------------------------------------------------------------------------
