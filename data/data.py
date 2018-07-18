import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from cycler import cycler
import scipy.optimize as opt
import scipy.special as spec
import scipy.interpolate as interp
import astropy.constants as const
import astropy.units as u
import astropy.io.fits as fits
import glob
import copy
import re
import sys
import pickle

# allow import of plot
sys.path.append('/Users/stijn/Documents/Universiteit/MR/code')
import plot as pl

import halo.parameters as p
import halo.tools as tools
import halo.gas as gas

import pdb

ddir = '/Users/stijn/Documents/Leiden/MR/code/halo/data/'
prms = p.prms


def read_croston():
    # Load in croston metadata
    ddir = 'halo/data/data_mccarthy/gas/Croston_data/'
    files = glob.glob(ddir + 'Croston08*.dat')
    fnames = [f.split('/')[-1] for f in files]
    idcs = [re.search('[0-9]*.dat', f).span() for f in fnames]
    sysnum = [int(fnames[idx][i[0]:i[1]][:-4]) for idx,i in enumerate(idcs)]

    # -------------------------------------------------------------------------- #
    # We leave all the original h_70 scalings, we scale our model when comparing #
    # -------------------------------------------------------------------------- #

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
    # rho = mu * m_p * n_gas -> fully ionised: mu=0.59 (X=0.75, Y=0.2461, Z=0.0039)
    # n_gas = n_e + n_H + n_He = 2n_H + 3n_He (fully ionized)
    #       = (2 + 3Y/(4X))n_H
    # n_He = Y/(4X) n_H
    # n_e = (2 + 3Y/(4X)) / (1 + Y/(2X)) n
    # => n_gas = 1.93 n_e
    # calculate correct mu factor for Z metallicity gas
    n2rho = 1.93 * 0.59 * const.m_p.cgs * 1/u.cm**3 # in cgs
    # change to ``cosmological'' coordinates
    cgs2cos = (1e6 * const.pc.cgs)**3 / const.M_sun.cgs
    rho =  [(ne * n2rho * cgs2cos).value for ne in n]
    rho_err = [(ne * n2rho * cgs2cos).value for ne in n_err]

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

    # we save our gas mass determinations!
    data = {'m500gas': mgas,
            'r500': r500,
            'rx': rx,
            'rho': rho}

    with open('data/croston.p', 'wb') as f:
        pickle.dump(data, f)

    return m500gas, r500, rx, rho

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
    # rho = mu * m_p * n_gas -> fully ionised: mu=0.59 (X=0.75, Y=0.2461, Z=0.0039)
    # n_gas = n_e + n_H + n_He = 2n_H + 3n_He (fully ionized)
    # n_He = Y/(4X) n_H
    # => n_gas = 2.25 n_H
    n2rho = 2.25 * 0.59 * const.m_p.cgs * 1./u.cm**3 # [cm^-3]
    cgs2cos = (1e6 * const.pc.cgs)**3 / const.M_sun.cgs

    r500 = np.empty((0,), dtype=float)
    z = np.empty((0,), dtype=float)
    rx = []
    rho = []
    rho_err = []
    for f in files:
        # actual data
        data = fits.open(f)
        z = np.append(z, data[1].header['REDSHIFT'])

        # r500 = np.append(r500, data[1].header['R500'] * 1e-3 * 0.7) # [Mpc/h]
        r500 = np.append(r500, data[1].header['R500'] * 1e-3) # [Mpc/h_70]

        # !!!! check whether this still needs correction factor from weak lensing
        # Currently included it, but biases the masses, since this wasn't done in
        # measurement by Eckert, of course
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
    m500 = m500mt[mdata2data]

    ratio = mgas/m500gas
    print('Eckert derived gas masses, too high if corrected rx for bias!!!:')
    print('median: ', np.median(ratio))
    print('q15:    ', np.percentile(ratio, q=15))
    print('q85:    ', np.percentile(ratio, q=85))

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

    # we save our gas mass determinations!
    data = {'m500gas': mgas,
            'r500': r500,
            'rx': rx,
            'rho': rho}

    with open('data/eckert.p', 'wb') as f:
        pickle.dump(data, f)

    return m500gas, r500, rx, rho

# ------------------------------------------------------------------------------
# End of read_eckert()
# ------------------------------------------------------------------------------

def bin_eckert(n=10):
    '''
    Bin the Eckert profiles into n mass bins

    Parameters
    ----------
    n : int
      number of bins
    '''
    # these are all assuming h_70!!
    m500g, r500, rx, rho = read_eckert()

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

    pl.set_style('line')
    for idx_m, m_bin in enumerate(np.arange(1, len(m_bins))):
        idx_in_bin = (m_bin_idx == m_bin)
        m500_med[idx_m] = np.median(m500g[idx_in_bin])
        r500_med[idx_m] = np.median(r500[idx_in_bin])
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
    m500g, r500, rx, rho = read_croston()

    # we need to plug in the 0.7^2
    m500 = tools.radius_to_mass(r500, 500 * p.prms.rho_crit * 0.7**2)

    pl.set_style()
    a = np.empty((0,), dtype=float)
    b = np.empty((0,), dtype=float)
    m_sl = np.empty((0,), dtype=float)
    # c = np.empty((0,), dtype=float)
    aerr = np.empty((0,), dtype=float)
    berr = np.empty((0,), dtype=float)
    # cerr = np.empty((0,), dtype=float)
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
                                   # , c:\
                                   prof_gas_hot(r, sl_fit, a, b,
                                                # , c, \
                                                mass, r500[idx]),
                                   # r[sl], prof[sl], bounds=([0, 0, 0.5],
                                   #                          [1, 5, 10]))
                                   r[sl], prof[sl], bounds=([0, 0],
                                                            [1, 5]))

        # print('f_gas,500c_actual :', m500g[idx] / m500[idx])
        # print('f_gas,500c_fitted :', m500gas / m500[idx])
        # print('m_gas,500c_fitted / m_gas,500c_actual', m500gas / m500g[idx])
        # print('-------------------')
        # plt.plot(r, prof, label='obs')
        # plt.plot(r, prof_gas_hot(r, sl_500, popt[0], popt[1], # , popt[2],
        #                          m500gas, r500[idx]),
        #          label='fit')
        # plt.title('%i'%idx)
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.legend()
        # plt.show()

        # Final fit will need to reproduce the m500gas mass
        a = np.append(a, popt[0])
        b = np.append(b, popt[1])
        m_sl = np.append(m_sl, m500gas)
        # c = np.append(c, popt[2])
        aerr = np.append(aerr, np.sqrt(np.diag(pcov))[0])
        berr = np.append(berr, np.sqrt(np.diag(pcov))[1])
        # cerr = np.append(cerr, np.sqrt(np.diag(pcov)))[2]

    return a, aerr, b, berr, m_sl, m500g, r500 # c, cerr, m500g

# ------------------------------------------------------------------------------
# End of fit_croston()
# ------------------------------------------------------------------------------

def fit_eckert():
    # these are all assuming h_70!!
    rx, rho, rho_std, m500g, r500 = bin_eckert()
    # m500g, r500, rx, rho = read_eckert()

    # we need to plug in the 0.7^2
    m500 = tools.radius_to_mass(r500, 500 * p.prms.rho_crit * 0.7**2)

    pl.set_style()
    a = np.empty((0,), dtype=float)
    b = np.empty((0,), dtype=float)
    m_sl = np.empty((0,), dtype=float)
    # c = np.empty((0,), dtype=float)
    aerr = np.empty((0,), dtype=float)
    berr = np.empty((0,), dtype=float)
    # cerr = np.empty((0,), dtype=float)
    for idx, prof in enumerate(rho):
        sl = ((prof > 0) & (rx[idx] >= 0.15) & (rx[idx] <= 1.))
        sl_500 = ((prof > 0) & (rx[idx] <= 1.))

        r = rx[idx]

        # Determine different profile masses
        mass = tools.m_h(prof[sl], r[sl] * r500[idx])
        m500gas = tools.m_h(prof[sl_500], r[sl_500] * r500[idx])
        # print 'm_gas500_actual/m_gas500_fit - 1 =', (m500g - m500gas) / m500gas
        # print 'f_gas,500c_actual :', m500g / m500[idx]
        # print 'f_gas,500c_fitted :', m500gas / m500[idx]
        # print '-------------------'

        # Need to perform the fit for [0.15,1] r500c -> mass within this region
        # need to match
        sl_fit = np.ones(sl.sum(), dtype=bool)
        popt, pcov = opt.curve_fit(lambda r, a, b: \
                                   # , c:\
                                   prof_gas_hot(r, sl_fit, a, b, # , c,
                                                mass, r500[idx]),
                                   # r[sl], prof[sl], bounds=([0, 0, 0.5],
                                   #                          [1, 5, 10]))
                                   r[sl], prof[sl], bounds=([0, 0],
                                                            [1, 5]))

        # plt.clf()
        # plt.plot(r, prof, label='obs')
        # plt.plot(r, prof_gas_hot(r, sl_500, popt[0], popt[1], # popt[2],
        #                          m500gas, r500[idx]),
        #          label='fit')
        # plt.title('%i'%idx)
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.legend()
        # plt.show()

        # Final fit will need to reproduce the m500gas mass
        a = np.append(a, popt[0])
        b = np.append(b, popt[1])
        m_sl = np.append(m_sl, m500gas)
        # c = np.append(c, popt[2])
        aerr = np.append(aerr, np.sqrt(np.diag(pcov))[0])
        berr = np.append(berr, np.sqrt(np.diag(pcov))[1])
        # cerr = np.append(cerr, np.sqrt(np.diag(pcov))[2])

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
    m500c = tools.radius_to_mass(r500c, 500 * p.prms.rho_crit * 0.7**2)
    m500e = tools.radius_to_mass(r500e, 500 * p.prms.rho_crit * 0.7**2)

    # Get 500crit values
    m500cc = gas.mgas_to_m500c(mgc)
    m500ce = gas.mgas_to_m500c(mge)

    # we thus need to convert our critical density as well
    r500cc = tools.mass_to_radius(m500cc, 500 * p.prms.rho_crit * 0.7**2)
    r500ce = tools.mass_to_radius(m500ce, 500 * p.prms.rho_crit * 0.7**2)

    m_range = np.logspace(np.log10(np.min(m500e)),
                          np.log10(np.max(m500c)), 100)

    data_croston = {'rc': rcc,
                    'rcerr': rccerr,
                    'b': bc,
                    'berr': bcerr,
                    'm500c': m500cc}
    data_eckert = {'rc': rce,
                   'rcerr': rceerr,
                   'b': be,
                   'berr': beerr,
                   'm500c': m500ce}

    with open('data/croston_500.p', 'wb') as f:
        pickle.dump(data_croston, f)

    with open('data/eckert_500.p', 'wb') as f:
        pickle.dump(data_eckert, f)

    if r200m:
        # assume h=1, since we already give correctly scaled h=0.7 values to the fit
        # hence we do not need to rescale the mass for c_correa
        m200mc = np.array([tools.m500c_to_m200m(m, prms.rho_crit * 0.7**2,
                                                prms.rho_m * 0.7**2, h=1.)
                           for m in m500cc])
        m200me = np.array([tools.m500c_to_m200m(m, prms.rho_crit * 0.7**2,
                                                prms.rho_m * 0.7**2, h=1.)
                           for m in m500ce])
        r200mc = tools.mass_to_radius(m200mc, 200 * p.prms.rho_crit *
                                      p.prms.omegam * 0.7**2)
        r200me = tools.mass_to_radius(m200me, 200 * p.prms.rho_crit *
                                      p.prms.omegam * 0.7**2)

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

def plot_parameters(mean=False):
    if mean:
        with open('data/croston_200.p', 'rb') as f:
            data_c = pickle.load(f)
        with open('data/eckert_200.p', 'rb') as f:
            data_e = pickle.load(f)
    else:
        with open('data/croston_500.p', 'rb') as f:
            data_c = pickle.load(f)
        with open('data/eckert_500.p', 'rb') as f:
            data_e = pickle.load(f)

    m500c = np.append(data_c['m500c'], data_e['m500c'])
    rc = np.append(data_c['rc'], data_e['rc'])
    rcerr = np.append(data_c['rcerr'], data_e['rcerr'])
    b = np.append(data_c['b'], data_e['b'])
    berr = np.append(data_c['berr'], data_e['berr'])

    m = np.logspace(np.log10(m500c.min()), np.log10(m500c.max()), 20)
    rc_med = np.median(rc) * np.ones_like(m)
    rc_q16 = np.percentile(rc, 16) * np.ones_like(m)
    rc_q84 = np.percentile(rc, 84) * np.ones_like(m)
    b_med = np.median(b) * np.ones_like(m)
    b_q16 = np.percentile(b, 16) * np.ones_like(m)
    b_q84 = np.percentile(b, 84) * np.ones_like(m)

    pl.set_style()
    fig = plt.figure(figsize=(20,6))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    # ax4 = fig.add_subplot(144)

    # Core radius rc
    ax1.set_prop_cycle(pl.cycle_mark())
    if mean:
        ax1.errorbar(data_c['m200m'], data_c['rc'], yerr=data_c['rcerr'],
                     marker='o', label=r'Croston+08')
        ax1.errorbar(data_e['m200m'], data_e['rc'], yerr=data_e['rcerr'],
                     marker='x', label=r'Eckert+16')
        ax1.set_xlabel(r'$m_{200\mathrm{m}} \, [\mathrm{M}_\odot]$')
        ax1.set_title(r'$r_c/r_{200\mathrm{m}}$')
    else:
        ax1.errorbar(data_c['m500c'], data_c['rc'], yerr=data_c['rcerr'],
                     marker='o', label=r'Croston+08')
        ax1.errorbar(data_e['m500c'], data_e['rc'], yerr=data_e['rcerr'],
                     marker='x', label=r'Eckert+16')
        ax1.set_prop_cycle(pl.cycle_line())
        ax1.plot(m, rc_med, ls='-', c='k', lw=2)
        ax1.plot(m, rc_q16, ls='--', c='k', lw=2)
        ax1.plot(m, rc_q84, ls='--', c='k', lw=2)
        ax1.set_xlabel(r'$m_{500\mathrm{c}} \, [\mathrm{M}_\odot]$')
        ax1.set_title(r'$r_c/r_{500\mathrm{c}}$')
    ax1.legend(loc='best')
    ax1.set_ylim([1e-3, 1])
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Slope beta
    ax2.set_prop_cycle(pl.cycle_mark())
    if mean:
        ax2.errorbar(data_c['m200m'], data_c['b'], yerr=data_c['berr'], marker='o',
                     label=r'Croston+08')
        ax2.errorbar(data_e['m200m'], data_e['b'], yerr=data_e['berr'], marker='x',
                     label=r'Eckert+16')
        ax2.set_xlabel(r'$m_{200\mathrm{m}} \, [\mathrm{M}_\odot]$')
    else:
        ax2.errorbar(data_c['m500c'], data_c['b'], yerr=data_c['berr'], marker='o',
                     label=r'Croston+08')
        ax2.errorbar(data_e['m500c'], data_e['b'], yerr=data_e['berr'], marker='x',
                     label=r'Eckert+16')
        ax2.set_prop_cycle(pl.cycle_line())
        ax2.plot(m, b_med, ls='-', c='k', lw=2)
        ax2.plot(m, b_q16, ls='--', c='k', lw=2)
        ax2.plot(m, b_q84, ls='--', c='k', lw=2)
        ax2.set_xlabel(r'$m_{500\mathrm{c}} \, [\mathrm{M}_\odot]$')
    ax2.legend(loc='best')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title(r'$\beta$')

    # # Cut-off radius rx
    # ax3.set_prop_cycle(pl.cycle_mark())
    # if mean:
    #     ax3.errorbar(data_c['m200m'], data_c['rx'], yerr=data_c['rxerr'],
    #                  marker='o', label=r'Croston+08')
    #     ax3.errorbar(data_e['m200m'], data_e['rx'], yerr=data_e['rxerr'],
    #                  marker='x', label=r'Eckert+16')
    #     ax3.set_xlabel(r'$m_{200\mathrm{m}} \, [\mathrm{M}_\odot]$')
    #     ax3.set_title(r'$r_x/r_{200\mathrm{m}}$')
    # else:
    #     ax3.errorbar(data_c['m500c'], data_c['rx'], yerr=data_c['rxerr'],
    #                  marker='o', label=r'Croston+08')
    #     ax3.errorbar(data_e['m500c'], data_e['rx'], yerr=data_e['rxerr'],
    #                  marker='x', label=r'Eckert+16')
    #     ax3.set_xlabel(r'$m_{500\mathrm{c}} \, [\mathrm{M}_\odot]$')
    #     ax3.set_title(r'$r_x/r_{500\mathrm{c}}$')
    # ax3.legend(loc='best')
    # ax3.set_xscale('log')
    # ax3.set_yscale('log')

    # Gas fractions
    m500_obs, f_obs = np.loadtxt(ddir +
                                 'data_mccarthy/gas/M500_fgas_BAHAMAS_data.dat',
                                 unpack=True)
    n_m = 15
    m_bins = np.logspace(m500_obs.min(), m500_obs.max(), n_m)
    m = tools.bins2center(m_bins)
    m_bin_idx = np.digitize(10**(m500_obs), m_bins)

    f_med = np.array([np.median(f_obs[m_bin_idx == m_bin])
                      for m_bin in np.arange(1, len(m_bins))])
    f_q16 = np.array([np.percentile(f_obs[m_bin_idx == m_bin], 16)
                      for m_bin in np.arange(1, len(m_bins))])
    f_q84 = np.array([np.percentile(f_obs[m_bin_idx == m_bin], 84)
                      for m_bin in np.arange(1, len(m_bins))])

    fmopt, fmcov = opt.curve_fit(lambda m, mc, a: f_gas(m, mc, a, 0.04),
                                 m[m>1e14], f_med[m>1e14],
                                 bounds=([1e10, 0],
                                         [1e15, 10]))
    f1opt, f1cov = opt.curve_fit(lambda m, mc, a: f_gas(m, mc, a, 0.02),
                                 m[m>1e14], f_q16[m>1e14],
                                 bounds=([1e10, 0],
                                         [1e15, 10]))
    f2opt, f2cov = opt.curve_fit(lambda m, mc, a: f_gas(m, mc, a, 0.06),
                                 m[m>1e14], f_q84[m>1e14],
                                 bounds=([1e10, 0],
                                         [1e15, 10]))

    fm_prms = {"mc": fmopt[0],
               "a": fmopt[1],
               "f0": 0.04}
    f1_prms = {"mc": f1opt[0],
               "a": f1opt[1],
               "f0": 0.02}
    f2_prms = {"mc": f2opt[0],
               "a": f2opt[1],
               "f0": 0.06}

    ax3.set_prop_cycle(pl.cycle_mark())
    ax3.plot(10**(m500_obs[23:33]), f_obs[23:33], marker='o',
             label=r'Vikhlinin+2006')
    ax3.plot(10**(m500_obs[128:165]), f_obs[128:165],
             marker='^', label=r'Maughan+2008')
    ax3.plot(10**(m500_obs[:23]), f_obs[:23], marker='v',
             label=r'Sun+2009')
    ax3.plot(10**(m500_obs[33:64]), f_obs[33:64], marker='<',
             label=r'Pratt+2009')
    ax3.plot(10**(m500_obs[64:128]), f_obs[64:128], marker='>',
             label=r'Lin+2012')
    ax3.plot(10**(m500_obs[165:]), f_obs[165:], marker='D',
             label=r'Lovisari+2015')
    ax3.set_prop_cycle(pl.cycle_line())
    ax3.plot(m, f_med, ls='-', c='k', lw=2)
    ax3.plot(m, f_q16, ls='--', c='k', lw=2)
    ax3.plot(m, f_q84, ls='--', c='k', lw=2)
    ax3.plot(m, f_gas(m, **fm_prms), ls='-', c='r', lw=1)
    ax3.plot(m, f_gas(m, **f1_prms), ls='--', c='r', lw=1)
    ax3.plot(m, f_gas(m, **f2_prms), ls='--', c='r', lw=1)
    ax3.yaxis.tick_right()
    ax3.yaxis.set_ticks_position('both')
    ax3.yaxis.set_label_position("right")
    ax3.set_xlim([1e13, 10**(15.5)])
    ax3.set_ylim([0.01, 0.145])
    ax3.set_xscale('log')
    ax3.set_xlabel(r'$m_{500\mathrm{c}} \, [\mathrm{M_\odot}]$')
    ax3.set_ylabel(r'$f_{\mathrm{gas}}$', rotation=270, labelpad=20)
    ax3.set_title(r'$f_{\mathrm{gas,500c}}$')
    leg = ax3.legend(loc='best', numpoints=1, frameon=True, framealpha=0.8)
    leg.get_frame().set_linewidth(0.0)

    plt.show()

# ------------------------------------------------------------------------------
# End of plot_parameters()
# ------------------------------------------------------------------------------

def f_gas(m, log10mc, a, prms):
    '''
    Return the f_gas(m500c) relation for m. The relation cannot exceed f_b

    This function assumes h=0.7 for everything!

    Parameters
    ----------
    m : array [M_sun / h_70]
      values of m500c to compute f_gas for
    log10mc : float
      the turnover mass for the relation in log10([M_sun/h_70])
    a : float
      the strength of the transition
    prms : Parameters object
      relevant cosmological parameters

    Returns
    -------
    f_gas : array [h_70^(-3/2)]
      gas fraction at r500c for m
    '''
    x = np.log10(m) - log10mc
    return (prms.omegab/prms.omegam) * (0.5 * (1 + np.tanh(x / a)))

# ------------------------------------------------------------------------------
# End of f_gas()
# ------------------------------------------------------------------------------

def f_gas_prms(prms, q=50):
    '''
    Compute best fit parameters to the f_gas(m500c) relation with both f_gas and
    m500c assuming h=0.7
    '''
    m500_obs, f_obs = np.loadtxt(ddir +
                                 'data_mccarthy/gas/M500_fgas_BAHAMAS_data.dat',
                                 unpack=True)

    n_m = 15

    m_bins = np.logspace(m500_obs.min(), m500_obs.max(), n_m)
    m = tools.bins2center(m_bins)

    # m_bins is in Hubble units
    m_bin_idx = np.digitize(10**(m500_obs), m_bins)

    f_q = np.array([np.percentile(f_obs[m_bin_idx == m_bin], q)
                      for m_bin in np.arange(1, len(m_bins))])

    fqopt, fqcov = opt.curve_fit(lambda m, log10mc, a: f_gas(m, log10mc, a, prms),
                                 m[m>1e14], f_q[m>1e14],
                                 bounds=([10, 0],
                                         [15, 10]))

    fq_prms = {"log10mc": fqopt[0],
               "a": fqopt[1]}

    return fq_prms

# ------------------------------------------------------------------------------
# End of f_gas_prms()
# ------------------------------------------------------------------------------

def f_gas_prms_debiased(prms):
    m500_obs, f_obs = np.loadtxt(ddir +
                                 'data_mccarthy/gas/M500_fgas_bias_corrected.dat',
                                 unpack=True)

    m500 = m500_obs

    fqopt, fqcov = opt.curve_fit(lambda m, log10mc, a: f_gas(m, log10mc, a, prms),
                                 m500, f_obs,
                                 bounds=([10, 0],
                                         [15, 10]))

    fq_prms = {"log10mc": fqopt[0],
               "a": fqopt[1]}

    return fq_prms

# ------------------------------------------------------------------------------
# End of f_gas_prms_debiased()
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

def massdiff_dmo2bar(m_bar, m_dmo, f_b, prms, fm):
    '''
    Return the mass difference between the measured halo mass and the
    dark matter only equivalent mass according to the relation

        m_dmo(m_b) = m_b / ( 1 - (f_b - f_gas(m_b)) )

    '''
    f_g = f_gas(m_bar, prms=prms, **fm)

    return (1 - (f_b - f_g)) * m_dmo - m_bar

def m200dmo_to_m200b(m_dmo, prms):
    '''
    Invert the relation between the measured halo mass and the dark matter only
    equivalent mass according to the relation

        m_dmo(m_b) = m_b / ( 1 - (f_b - f_gas(m_b)) )
    '''
    f_b = 1 - prms.f_dm
    fm = f_gas_prms(prms, q=50)
    m200_b = opt.brentq(massdiff_dmo2bar, m_dmo / 10., m_dmo,
                        args=(m_dmo, f_b, prms, fm))

    return m200_b

# ------------------------------------------------------------------------------
# End of m200dmo_to_m200b()
# ------------------------------------------------------------------------------

def m200b_to_m200dmo(m_b, f_model, prms):
    '''
    Get the relation between the measured halo mass and the dark matter only
    equivalent mass according to the relation

        m_dmo(m_b) = m_b / ( 1 - (f_b - f_model(m_b)) )
    '''
    f_b = 1 - prms.f_dm
    m200dmo = m_b / (1 - (f_b - f_model))

    return m200dmo

# ------------------------------------------------------------------------------
# End of m200b_to_m200dmo()
# ------------------------------------------------------------------------------

def plot_profiles_paper(prms=prms):
    with open('data/croston.p', 'rb') as f:
        data_c = pickle.load(f)
    with open('data/eckert.p', 'rb') as f:
        data_e = pickle.load(f)

    with open('data/croston_500.p', 'rb') as f:
        data_hm_c = pickle.load(f)
    with open('data/eckert_500_all.p', 'rb') as f:
        data_hm_e = pickle.load(f)

    # pl.set_style()
    fig = plt.figure(figsize=(16,8))
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
    rho_crit = prms.rho_crit * 0.7**2


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

    plt.savefig('obs_profiles.pdf', transparent=True,
                bbox_inches='tight')
    plt.show()
# ------------------------------------------------------------------------------
# End of plot_profiles()
# ------------------------------------------------------------------------------

def plot_fit_profiles_paper():
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

    fig = plt.figure(figsize=(18,8))
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
    ticks = ax3.get_yticklabels()
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
    ticks = ax2.get_xticklabels()
    ticks[-3].set_visible(False)

    ax2.xaxis.set_tick_params(pad=8)
    ax2.set_xscale('log')
    ax2.set_yticklabels([])
    ax2.set_xlabel(r'$r/r_\mathrm{500c}$', labelpad=-8)
    plt.savefig('obs_profiles_fit.pdf', transparent=True, bbox_inches='tight')
    plt.show()

# ------------------------------------------------------------------------------
# End of plot_fit_profiles()
# ------------------------------------------------------------------------------

def plot_fit_prms_paper(prms=prms):
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

    m200m_e = data200m_e['m200m']
    m200m_c = data200m_c['m200m']
    m200m = np.append(m200m_e, m200m_c)
    # m200m = np.array([tools.m500c_to_m200m(m500c.min(), prms.rho_crit, prms.rho_m, prms.h),
    #                   tools.m500c_to_m200m(m500c.max(), prms.rho_crit, prms.rho_m, prms.h)])

    pl.set_style('line')

    fig = plt.figure(figsize=(10,9))
    ax = fig.add_subplot(111)

    ax.errorbar(m500c_c, rc_c, yerr=rc_c_err, lw=0, elinewidth=1, marker='s',
                label='Croston+08')
    ax.errorbar(m500c_e, rc_e, yerr=rc_e_err, lw=0, elinewidth=1, marker='o',
                label='Eckert+16')
    ax.axhline(y=np.median(rc), ls='-', c='k')
    ax.axhspan(np.percentile(rc, 15), np.percentile(rc, 85),
               facecolor='k', alpha=0.3)
    ax.annotate('median', xy=(1.05 * m500c.min(), 1.05 * np.median(rc)))

    ##########################################################################
    # Get shadow twiny instance for ratio plot, to also have m200m
    # need it in this order to get correct yticks with log scale, since
    # twin instance seems to mess things up...
    axs = ax.twiny()
    axs.plot([m200m.min(), m200m.max()], [m200m.min(), m200m.max()])
    axs.cla()

    ax.set_xlim(m500c.min(), m500c.max())
    axs.set_xlim(m200m.min(), m200m.max())

    ax.set_ylim(0,0.5)
    ax.set_xscale('log')
    axs.set_xscale('log')

    ax.xaxis.set_tick_params(pad=10)
    axs.xaxis.set_tick_params(pad=5)

    ax.set_xlabel(r'$m_{\mathrm{500c}}\, [h_{70}^{-1} \, \mathrm{M_\odot}]$')
    axs.set_xlabel(r'$m_{\mathrm{200m}}\, [h_{70}^{-1} \, \mathrm{M_\odot}]$',
                   labelpad=15)
    ax.set_ylabel('$r_\mathrm{c}/r_{\mathrm{500c}}$')
    ax.legend(loc='best')
    plt.savefig('obs_rc_fit.pdf', transparent=True, bbox_inches='tight')

    plt.clf()
    fig = plt.figure(figsize=(10,9))
    ax = fig.add_subplot(111)

    ax.errorbar(m500c_c, b_c, yerr=b_c_err, lw=0, elinewidth=1, marker='s',
                label='Croston+08')
    ax.errorbar(m500c_e, b_e, yerr=b_e_err, lw=0, elinewidth=1, marker='o',
                label='Eckert+16')
    ax.axhline(y=np.median(beta), ls='-', c='k')
    ax.axhspan(np.percentile(beta, 15), np.percentile(beta, 85),
               facecolor='k', alpha=0.3)
    ax.annotate('median', xy=(1.05 * m500c.min(), 1.05 * np.median(beta)))

    ##########################################################################
    # Get shadow twiny instance for ratio plot, to also have m200m
    # need it in this order to get correct yticks with log scale, since
    # twin instance seems to mess things up...
    axs = ax.twiny()
    axs.plot([m200m.min(), m200m.max()], [m200m.min(), m200m.max()])
    axs.cla()

    ax.set_xlim(m500c.min(), m500c.max())
    axs.set_xlim(m200m.min(), m200m.max())

    ax.set_ylim(0.04, 1.7)
    ax.set_xscale('log')
    axs.set_xscale('log')

    ax.xaxis.set_tick_params(pad=10)
    axs.xaxis.set_tick_params(pad=5)

    ax.set_xlabel('$m_{\mathrm{500c}}\, [h_{70}^{-1} \, \mathrm{M_\odot}]$')
    axs.set_xlabel(r'$m_{\mathrm{200m}}\, [h_{70}^{-1} \, \mathrm{M_\odot}]$',
                   labelpad=15)
    ax.set_ylabel(r'$\beta$')
    ax.legend(loc='best')
    plt.savefig('obs_beta_fit.pdf', transparent=True, bbox_inches='tight')

# ------------------------------------------------------------------------------
# End of plot_fit_prms_paper()
# ------------------------------------------------------------------------------

def plot_missing_mass_paper(comp_gas):
    '''
    Plot the total missing mass by assuming beta profile fit to gas up to r200m
    '''
    set_style('line')

    fig = plt.figure(figsize=(10,9))
    ax = fig.add_subplot(111)

    m_range = comp_gas.m_range
    f_dm = p.prms.f_dm * np.ones_like(m_range)
    f_gas = comp_gas.m_h / m_range

    m500c = np.array([tools.m200m_to_m500c(mass) for mass in m_range[::5]])
    f500c = d.f_gas(m500c, **d.f_gas_prms(prms, q=50))

    ax.plot(m_range, f_dm, label='$f_{\mathrm{dm}}$')
    ax.plot(m_range, f_dm + f_gas, label='$f_{\mathrm{dm}} + f_{\mathrm{gas,200m}}$')
    ax.plot(m_range[::5], f_dm[0] + f500c, label='$f_{\mathrm{dm}} + f_{\mathrm{gas,500c}}$')
    ax.set_xscale('log')
    ax.set_xlabel('$m_{\mathrm{200m}} \, [\mathrm{M_\odot}]$')
    ax.set_ylabel('$f(m)$')
    ax.xaxis.set_tick_params(pad=10)
    ax.legend(loc='best')

    plt.show()

# ------------------------------------------------------------------------------
# End of plot_missing_mass_paper()
# ------------------------------------------------------------------------------

def plot_masses_hist_paper():
    with open('data/croston_500.p', 'rb') as f:
        data_c = pickle.load(f)
    with open('data/eckert_500_all.p', 'rb') as f:
        data_e = pickle.load(f)
    with open('data/croston_200.p', 'rb') as f:
        data200m_c = pickle.load(f)
    with open('data/eckert_200_all.p', 'rb') as f:
        data200m_e = pickle.load(f)

    m500c_e = data_e['m500c']
    m500c_c = data_c['m500c']
    m500c = np.append(m500c_e, m500c_c)

    m200m_e = data200m_e['m200m']
    m200m_c = data200m_c['m200m']
    m200m = np.append(m200m_e, m200m_c)

    mn = np.min(np.hstack([m200m_c, m200m_e]))
    mx = np.max(np.hstack([m200m_c, m200m_e]))

    bins = np.logspace(np.log10(mn), np.log10(mx), 10)

    pl.set_style('line')
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.hist([m200m_c, m200m_e], bins=bins, stacked=True, histtype='barstacked',
            label=['Croston+08', 'Eckert+16'])

    axs = ax.twiny()
    axs.plot([m500c.min(), m500c.max()], [0, 0])
    axs.cla()

    ax.set_xlim(m200m.min(), m200m.max())
    axs.set_xlim(m500c.min(), m500c.max())
    ax.set_ylim(0,41)

    ax.minorticks_on()
    ax.set_xlabel(r'$m_\mathrm{200m} \, [h_{70}^{-1} \, \mathrm{M_\odot}]$')
    axs.set_xlabel(r'$m_{\mathrm{500c}}\, [h_{70}^{-1} \, \mathrm{M_\odot}]$',
                   labelpad=15)

    ax.set_ylabel(r'Frequency')

    ax.set_xscale('log')
    axs.set_xscale('log')

    ax.xaxis.set_tick_params(pad=10)
    axs.xaxis.set_tick_params(pad=5)

    ax.legend(loc='best')

    plt.savefig('obs_masses_hist_stacked.pdf',
                transparent=True, bbox_inches='tight')
    plt.show()

# ------------------------------------------------------------------------------
# End of plot_masses_hist_paper()
# ------------------------------------------------------------------------------
