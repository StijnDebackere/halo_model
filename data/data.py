import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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
import cPickle

# allow import of plot
sys.path.append('~/Documents/Universiteit/MR/code')
import plot as pl

import halo.parameters as p
import halo.tools as tools
import halo.gas as gas

import pdb

ddir = '/Volumes/Data/stijn/Documents/Leiden/MR/code/halo/data/'
prms = p.prms

def h(z):
    return (0.3 * (1+z)**3 + 0.7)**0.5

# ------------------------------------------------------------------------------
# End of h()
# ------------------------------------------------------------------------------

def read_croston():
    # Load in croston metadata
    ddir = 'halo/data/data_mccarthy/gas/Croston_data/'
    files = glob.glob(ddir + 'Croston08*.dat')
    fnames = [f.split('/')[-1] for f in files]
    idcs = [re.search('[0-9]*.dat', f).span() for f in fnames]
    sysnum = [int(fnames[idx][i[0]:i[1]][:-4]) for idx,i in enumerate(idcs)]

    data = np.loadtxt(ddir + 'Pratt09.dat')
    z = data[:,0]
    r500 = data[:,1] * 1e-3 # [Mpc/h70]
    mgas500 = np.power(10, data[:,2]) # [Msun/h70^(5/2)]
    mgas500_err = data[:,3]
    Z = data[:,4]

    # Load in croston data -> n = n_e
    rx = [np.loadtxt(f)[:,1] for idx, f in enumerate(files)]
    n = [np.loadtxt(f)[:,2] for idx, f in enumerate(files)]
    n_err = [np.loadtxt(f)[:,3] for idx, f in enumerate(files)]

    # Convert electron densities to gas density
    # rho = mu * m_p * n_gas -> fully ionised: mu=0.61 (X=0.707, Y=0.290)
    # n_gas = n_e + n_H + n_He = 2n_H + 3n_He (fully ionized)
    #       = (2 + 3Y/(4X))n_H
    # n_He = Y/(4X) n_H
    # n_e = (2 + 3Y/(4X)) / (1 + Y/(4X)) n
    # => n_gas = 1.93 n_e
    # calculate correct mu factor for Z metallicity gas
    n2rho = 1.93 * 0.61 * const.m_p.cgs * 1/u.cm**3 # in cgs
    # change to ``cosmological'' coordinates
    cgs2cos = (1e6 * const.pc.cgs)**3 / const.M_sun.cgs
    rho =  [(ne * n2rho * cgs2cos).value for ne in n]
    rho_err = [(ne * n2rho * cgs2cos).value for ne in n_err]

    mgas = np.empty((0,), dtype=float)
    m500gas = np.empty((0,), dtype=float)
    for idx, prof in enumerate(rho):
        idx_500 = np.argmin(np.abs(rx[idx] - 1))
        mgas = np.append(mgas,
                         tools.m_h(prof[:idx_500+1],
                                   rx[idx][:idx_500+1]*r500[sysnum[idx]-1]))
        m500gas = np.append(m500gas, mgas500[sysnum[idx] - 1])
        # print mgas[idx]
        # print mgas500[sysnum[idx]-1]
        # print '-----------'

    #     plt.plot(rx[idx], prof)

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

    data = {'m500gas': m500gas,
            'r500': r500,
            'rx': rx,
            'rho': rho}

    with open('croston.p', 'wb') as f:
        cPickle.dump(data, f)

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

    # number of cluster
    num = mdata[1].data['xlssc']
    z = mdata[1].data['z']
    # z_err = mdata[1].data['ez'] #?
    # r500mt = mdata[1].data['r500mt'] # same as in fits files
    m500mt = mdata[1].data['M500MT']
    m500mt_err = mdata[1].data['M500MT_err']
    mgas500 = mdata[1].data['Mgas500']
    mgas500_err = mdata[1].data['Mgas500_err']
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
    # rho = mu * m_p * n_gas -> fully ionised: mu=0.61 (X=0.707, Y=0.290)
    # n_gas = n_e + n_H + n_He = 2n_H + 3n_He (fully ionized)
    # n_He = Y/(4X) n_H
    # => n_gas = 2.31 n_H
    n2rho = 2.21 * 0.61 * const.m_p.cgs * 1/u.cm**3 # in cgs
    cgs2cos = (1e6 * const.pc.cgs)**3 / const.M_sun.cgs

    r500 = np.empty((0,), dtype=float)
    z = np.empty((0,), dtype=float)
    rx = []
    rho = []
    rho_err = []
    for f in files:
        # actual data
        data = fits.open(f)
        r500 = np.append(r500, data[1].header['R500'] * 1e-3) # [Mpc]
        z = np.append(z, data[1].header['REDSHIFT'])
        # !!!! check whether this still needs correction factor from weak lensing
        # Currently included it, but biases the masses, since this wasn't done in
        # measurement by Eckert, of course
        # rx.append(data[1].data['RADIUS'] * (1.3)**(1./3))
        rx.append(data[1].data['RADIUS'])
        rho.append(data[1].data['NH'] * n2rho * cgs2cos)
        rho_err.append(data[1].data['ENH'] * n2rho * cgs2cos)
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
        # print '------------'-

    m500gas = mgas500[mdata2data]

    ratio = mgas/m500gas
    print 'median: ', np.median(ratio)
    print 'q15:    ', np.percentile(ratio, q=15)
    print 'q85:    ', np.percentile(ratio, q=85)
    # pl.set_style('mark')
    # plt.plot(m500gas, ratio)
    # plt.axhline(y=1, color='k')
    # plt.xscale('log')
    # plt.xlabel(r'$m_\mathrm{500c,gas-measured} \, [M_\odot/h_{70}^{5/2}]$')
    # plt.ylabel(r'$\left(\int\,\rho(r)\,\mathrm{d}^3r\right)/m_\mathrm{500c,gas-measured}$',
    #            labelpad=-5)

    plt.hist(ratio)
    plt.axvline(x=np.median(ratio), c='k', ls='-', label='median')
    plt.axvline(x=np.percentile(ratio, 15), c='k', ls='--',
                label='$15-85^\mathrm{th}$ percentile')
    plt.axvline(x=np.percentile(ratio, 85), c='k', ls='--')
    plt.xlabel(r'$m_\mathrm{500c,gas-integrated}/m_\mathrm{500c,gas-measured}$')
    plt.ylabel('Frequency')

    plt.title('Eckert+16 mass determination')
    plt.legend()
    plt.show()

    data = {'m500gas': m500gas,
            'r500': r500,
            'rx': rx,
            'rho': rho}

    with open('eckert.p', 'wb') as f:
        cPickle.dump(data, f)

    return m500gas, r500, rx, rho

# ------------------------------------------------------------------------------
# End of read_eckert()
# ------------------------------------------------------------------------------

def bin_croston():
    with open('croston.p', 'rb') as f:
        data_croston = cPickle.load(f)

    r500 = data_croston['r500']
    m500g = data_croston['m500gas']
    rx = data_croston['rx']
    rho = data_croston['rho']

    # number of points to mass bin
    n_m = 10
    m_bins = np.logspace(np.log10(m500g).min(), np.log10(m500g).max(), n_m)
    m_bin_idx = np.digitize(m500g, m_bins)

    r_min = 0.
    r_max = 3.
    # number of points in new profile
    n_r = 20
    r_ranges = np.empty((n_m-1, n_r), dtype=float)
    rho_med = np.empty((n_m-1, n_r), dtype=float)
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

        # for t in rho_new:
        #     print t
        #     t[t <= 0] = np.nan
        #     if (t <= 0).sum() == 0:
        #         plt.plot(r_range, t)
        # plt.plot(r_range, rho_med[idx_m], label=r'median')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.legend(loc='best')
        # plt.show()

    return r_ranges, rho_med, m500_med, r500_med

# ------------------------------------------------------------------------------
# End of bin_croston()
# ------------------------------------------------------------------------------

def bin_eckert():
    with open('eckert.p', 'rb') as f:
        data_eckert = cPickle.load(f)

    r500 = data_eckert['r500']
    m500g = data_eckert['m500gas']
    rx = data_eckert['rx']
    rho = data_eckert['rho']

    # number of points to mass bin
    n_m = 15
    m_bins = np.logspace(np.log10(m500g).min(), np.log10(m500g).max(), n_m)
    m_bin_idx = np.digitize(m500g, m_bins)

    r_min = 0.
    r_max = 3.
    # number of points in new profile
    n_r = 20
    r_ranges = np.empty((n_m-1, n_r), dtype=float)
    rho_med = np.empty((n_m-1, n_r), dtype=float)
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

        # for t in rho_new:
        #     print t
        #     t[t <= 0] = np.nan
        #     if (t <= 0).sum() == 0:
        #         plt.plot(r_range, t)
        # plt.plot(r_range, rho_med[idx_m], label=r'median')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.legend(loc='best')
        # plt.show()

    return r_ranges, rho_med, m500_med, r500_med

# ------------------------------------------------------------------------------
# End of bin_eckert()
# ------------------------------------------------------------------------------

# def prof_gas_hot(x, sl, a, b, c, m_sl, r500):
#     '''beta profile'''
#     profile = (1 + (x/a)**2)**(-b/2) * np.exp(-(x/c)**2)
#     mass = tools.m_h(profile[sl], x[sl] * r500)
#     profile *= m_sl/mass

#     return profile

def prof_gas_hot(x, sl, a, b, m_sl, r500):
    '''beta profile'''
    profile = (1 + (x/a)**2)**(-b/2)
    mass = tools.m_h(profile[sl], x[sl] * r500)
    profile *= m_sl/mass

    return profile

def prof_gas_plaw(x, sl, rho_0, a):
    '''power law extension for r>r500c'''
    profile = rho_0 * x[sl]**a
    return profile

def fit_croston():
    '''
    Fit profiles to the observations
    '''
    with open('croston.p', 'rb') as f:
        data_croston = cPickle.load(f)

    r500 = data_croston['r500']
    m500 = tools.radius_to_mass(r500, 500 * p.prms.rho_crit)
    m500g = data_croston['m500gas']
    rx = data_croston['rx']
    rho = data_croston['rho']
    # rx, rho, m500g, r500 = bin_croston()

    pl.set_style()
    a = np.empty((0,), dtype=float)
    b = np.empty((0,), dtype=float)
    m_sl = np.empty((0,), dtype=float)
    # c = np.empty((0,), dtype=float)
    aerr = np.empty((0,), dtype=float)
    berr = np.empty((0,), dtype=float)
    # cerr = np.empty((0,), dtype=float)
    for idx, prof in enumerate(rho):
        # sl = (rx[idx] >= 0.05)
        sl = ((prof > 0) & (rx[idx] >= 0.15))
        r = rx[idx]
        mass = tools.m_h(prof[sl], r[sl] * r500[idx])
        # mass_actual = tools.m_h(prof[prof > 0], r[prof > 0] * r500[idx])
        # print 'f_gas,500c_actual :', mass_actual / m500[idx]
        # print 'f_gas,500c_fitted :', mass / m500[idx]
        # print '-------------------'

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

        # plt.plot(r, prof)
        # plt.plot(r, prof_gas_hot(r, sl, popt[0], popt[1], # , popt[2],
        #                          mass, r500[idx]))
        # plt.title('%i'%idx)
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()

        a = np.append(a, popt[0])
        b = np.append(b, popt[1])
        m_sl = np.append(m_sl, mass)
        # c = np.append(c, popt[2])
        aerr = np.append(aerr, np.sqrt(np.diag(pcov)))[0]
        berr = np.append(berr, np.sqrt(np.diag(pcov)))[1]
        # cerr = np.append(cerr, np.sqrt(np.diag(pcov)))[2]

    return a, aerr, b, berr, m_sl, m500g # c, cerr, m500g

# ------------------------------------------------------------------------------
# End of fit_croston()
# ------------------------------------------------------------------------------

def fit_eckert():
    # with open('eckert.p', 'rb') as f:
    #     data_eckert = cPickle.load(f)

    # r500 = data_eckert['r500']
    # m500 = tools.radius_to_mass(r500, 500 * p.prms.rho_crit)
    # m500g = data_eckert['m500gas']
    # rx = data_eckert['rx']
    # rho = data_eckert['rho']
    rx, rho, m500g, r500 = bin_eckert()

    m500 = tools.radius_to_mass(r500, 500 * p.prms.rho_crit)

    pl.set_style()
    a = np.empty((0,), dtype=float)
    b = np.empty((0,), dtype=float)
    m_sl = np.empty((0,), dtype=float)
    # c = np.empty((0,), dtype=float)
    aerr = np.empty((0,), dtype=float)
    berr = np.empty((0,), dtype=float)
    # cerr = np.empty((0,), dtype=float)
    for idx, prof in enumerate(rho):
        # sl = (rx[idx] >= 0.05)
        sl = ((prof > 0) & (rx[idx] >= 0.15))
        r = rx[idx]
        mass = tools.m_h(prof[sl], r[sl] * r500[idx])
        mass_actual = tools.m_h(prof[prof > 0], r[prof > 0] * r500[idx])
        # print 'm_gas500/m_gas500_actual - 1', (mass - mass_actual) / mass_actual
        # print 'f_gas,500c_actual :', mass_actual / m500[idx]
        # print 'f_gas,500c_fitted :', mass / m500[idx]
        # print '-------------------'
        sl_fit = np.ones(sl.sum(), dtype=bool)
        popt, pcov = opt.curve_fit(lambda r, a, b: \
                                   # , c:\
                                   prof_gas_hot(r, sl_fit, a, b, # , c,
                                                  mass, r500[idx]),
                                   # r[sl], prof[sl], bounds=([0, 0, 0.5],
                                   #                          [1, 5, 10]))
                                   r[sl], prof[sl], bounds=([0, 0],
                                                            [1, 5]))

        # plt.plot(r, r**2 * prof)
        # plt.plot(r, r**2 * prof_gas_hot(r, sl, popt[0], popt[1], # popt[2],
        #                                 mass, r500[idx]))
        # plt.title('%i'%idx)
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()

        a = np.append(a, popt[0])
        b = np.append(b, popt[1])
        m_sl = np.append(m_sl, mass)
        # c = np.append(c, popt[2])
        aerr = np.append(aerr, np.sqrt(np.diag(pcov)))[0]
        berr = np.append(berr, np.sqrt(np.diag(pcov)))[1]
        # cerr = np.append(cerr, np.sqrt(np.diag(pcov)))[2]


    return a, aerr, b, berr, m_sl, m500g # c, cerr, m500g

# ------------------------------------------------------------------------------
# End of fit_eckert()
# ------------------------------------------------------------------------------

def convert_hm():
    '''
    Save fit parameters and masses for both m500c & m200m
    '''
    # rcc, rccerr, bc, bcerr, rxc, rxcerr, mgc = fit_croston()
    # rce, rceerr, be, beerr, rxe, rxeerr, mge = fit_eckert()
    rcc, rccerr, bc, bcerr, mslc, mgc = fit_croston()
    rce, rceerr, be, beerr, msle, mge = fit_eckert()

    # Get 500crit values
    m500cc = gas.mgas_to_m500c(mgc)
    m500ce = gas.mgas_to_m500c(mge)
    r500cc = tools.mass_to_radius(m500cc, 500 * p.prms.rho_crit * p.prms.h**2)
    r500ce = tools.mass_to_radius(m500ce, 500 * p.prms.rho_crit * p.prms.h**2)

    data_croston = {'rc': rcc,
                    'rcerr': rccerr,
                    'b': bc,
                    'berr': bcerr,
                    # 'rx': rxc,
                    # 'rxerr': rxcerr,
                    'm500c': m500cc}
    data_eckert = {'rc': rce,
                   'rcerr': rceerr,
                   'b': be,
                   'berr': beerr,
                   # 'rx': rxe,
                   # 'rxerr': rxeerr,
                   'm500c': m500ce}

    with open('croston_500.p', 'wb') as f:
        cPickle.dump(data_croston, f)

    with open('eckert_500.p', 'wb') as f:
        cPickle.dump(data_eckert, f)

    # Get 200mean values
    m200mc = np.array([gas.m500c_to_m200m(m) for m in m500cc])
    m200me = np.array([gas.m500c_to_m200m(m) for m in m500ce])
    r200mc = tools.mass_to_radius(m200mc, 200 * p.prms.rho_crit * p.prms.h**2 *
                                  p.prms.omegam)
    r200me = tools.mass_to_radius(m200me, 200 * p.prms.rho_crit * p.prms.h**2 *
                                  p.prms.omegam)

    rcc *= r500cc/r200mc
    rccerr *= r500cc/r200mc
    rce *= r500ce/r200me
    rceerr *= r500ce/r200me
    # rxc *= r500cc/r200mc
    # rxcerr *= r500cc/r200mc
    # rxe *= r500ce/r200me
    # rxeerr *= r500ce/r200me

    data_croston = {'rc': rcc,
                    'rcerr': rccerr,
                    'b': bc,
                    'berr': bcerr,
                    # 'rx': rxc,
                    # 'rxerr': rxcerr,
                    'm200m': m200mc}
    data_eckert = {'rc': rce,
                   'rcerr': rceerr,
                   'b': be,
                   'berr': beerr,
                   # 'rx': rxe,
                   # 'rxerr': rxeerr,
                   'm200m': m200me}

    with open('croston_200.p', 'wb') as f:
        cPickle.dump(data_croston, f)

    with open('eckert_200.p', 'wb') as f:
        cPickle.dump(data_eckert, f)

# ------------------------------------------------------------------------------
# End of convert_hm()
# ------------------------------------------------------------------------------

def plot_parameters(mean=False):
    if mean:
        with open('croston_200.p', 'rb') as f:
            data_c = cPickle.load(f)
        with open('eckert_200.p', 'rb') as f:
            data_e = cPickle.load(f)
    else:
        with open('croston_500.p', 'rb') as f:
            data_c = cPickle.load(f)
        with open('eckert_500.p', 'rb') as f:
            data_e = cPickle.load(f)

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

def f_gas(m, mc, a, f0):
    return (p.prms.omegab/p.prms.omegam - f0) * (m/mc)**a/ (1 + (m/mc)**a) + f0

def f_gas_prms():
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

    fmopt, fmcov = opt.curve_fit(lambda m, mc, a: f_gas(m, mc, a, 0.),
                                 m[m>1e14], f_med[m>1e14],
                                 bounds=([1e5, 0],
                                         [1e15, 10]))
    f1opt, f1cov = opt.curve_fit(lambda m, mc, a: f_gas(m, mc, a, 0.),
                                 m[m>1e14], f_q16[m>1e14],
                                 bounds=([1e5, 0],
                                         [1e15, 10]))
    f2opt, f2cov = opt.curve_fit(lambda m, mc, a: f_gas(m, mc, a, 0.),
                                 m[m>1e14], f_q84[m>1e14],
                                 bounds=([1e5, 0],
                                         [1e15, 10]))

    fm_prms = {"mc": fmopt[0],
               "a": fmopt[1],
               "f0": 0.}
    f1_prms = {"mc": f1opt[0],
               "a": f1opt[1],
               "f0": 0.}
    f2_prms = {"mc": f2opt[0],
               "a": f2opt[1],
               "f0": 0.}

    return fm_prms, f1_prms, f2_prms

# ------------------------------------------------------------------------------
# End of f_gas_prms()
# ------------------------------------------------------------------------------

def prof_prms(mean=False):
    if mean:
        with open('croston_200.p', 'rb') as f:
            data_c = cPickle.load(f)
        with open('eckert_200.p', 'rb') as f:
            data_e = cPickle.load(f)
    else:
        with open('croston_500.p', 'rb') as f:
            data_c = cPickle.load(f)
        with open('eckert_500.p', 'rb') as f:
            data_e = cPickle.load(f)

    m500c = np.append(data_c['m500c'], data_e['m500c'])
    rc = np.append(data_c['rc'], data_e['rc'])
    rcerr = np.append(data_c['rcerr'], data_e['rcerr'])
    b = np.append(data_c['b'], data_e['b'])
    berr = np.append(data_c['berr'], data_e['berr'])

    sl = (rc > 1e-2)

    copt, ccov = opt.curve_fit(corr_fit, np.log10(rc)[sl], np.log10(b)[sl])
    corr_prms = {"a": copt[0],
                 "b": copt[1]}

    rc_min = rc[sl].min()
    rc_max = rc[sl].max()

    rc_med = np.median(rc)
    beta_med = np.median(b)

    # pl.set_style('mark')
    # plt.plot(rc, b)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()

    # rc_med = np.median(rc)
    # rc_q16 = np.percentile(rc, 16)
    # rc_q84 = np.percentile(rc, 84)
    # b_med = np.median(b)
    # b_q16 = np.percentile(b, 16)
    # b_q84 = np.percentile(b, 84)

    # return rc_med, rc_q16, rc_q84, b_med, b_q16, b_q84
    return corr_prms, rc_min, rc_max, rc_med, beta_med

# ------------------------------------------------------------------------------
# End of prof_prms()
# ------------------------------------------------------------------------------
def prof_beta(r_range, rc, beta):
    return 1. / (1 + (r_range/rc)**2)**(beta/2)

def minimize(prms, sl500, m500, m200, r_range):
    '''
    Force beta profile to contain m200 in r_range, and minimize difference
    between m500 and m500 in profile through beta & rc
    '''
    prof = prof_beta(r_range, prms[0], prms[1])
    norm = tools.m_h(prof, r_range)
    prof *= m200 / norm
    # add integration up to r500 -> force to be equal to m500
    m500_new = tools.m_h(prof[sl500], r_range[sl500])
    return np.abs(m500_new - m500)

# ------------------------------------------------------------------------------
# End of minimize()
# ------------------------------------------------------------------------------

def beta_mass(f500c, f200m, prms=p.prms):
    '''
    Compute the fit parameters for the beta profile going through
    f500c and f200m
    '''
    r = prms.r_range_lin#[::5]
    m200m = prms.m_range_lin#[::5]
    r200m = tools.mass_to_radius(m200m, 200 * prms.omegam * prms.rho_crit *
                                 prms.h**2)
    m500c = np.array([gas.m200m_to_m500c(m) for m in m200m])
    r500c = tools.mass_to_radius(m500c, 500 * prms.rho_crit * prms.h**2)

    # # Gas fractions
    # gas_prms = f_gas_prms()
    # f200m = 1 - prms.f_dm
    # f500c = f_gas(m500c, **gas_prms[0])

    beta = np.empty((0,), dtype=float)
    rc = np.empty((0,), dtype=float)
    for idx, f in enumerate(f500c):
        sl500 = (r[idx] <= r500c[idx])
        res = opt.minimize(minimize, [1.2, 3./2],
                           args=(sl500, f * m500c[idx], f200m[idx] * m200m[idx],
                                 r[idx]),
                           bounds=((0,10), (0,3)))
        # print res.x
        rc = np.append(rc, res.x[0])
        beta = np.append(beta, res.x[1])

    m500_h = np.empty((0,), dtype=float)
    prof_h = np.zeros_like(r)
    for idx, prof in enumerate(prof_h):
        sl500 = (r[idx] <= r500c[idx])

        # parameters should ensure match at f500c for f200m
        p = prof_beta(r[idx], rc[idx], beta[idx])
        norm = tools.m_h(p, r[idx])
        p *= f200m[idx] * m200m[idx] / norm

        prof_h[idx] = p
        m500_h = np.append(m500_h, tools.m_h(p[sl500], r[idx][sl500]))

    m200_h = tools.m_h(prof_h, r)

    # print 'prms: '
    # print beta
    # print rc/r500c
    # print 'mass fractions: '
    # print m200_h / m200m
    # print m500_h / m500c
    # print f500c

    return beta, rc/r500c, m500c, r, prof_h

# ------------------------------------------------------------------------------
# End of beta_mass()
# ------------------------------------------------------------------------------

# def beta_slope(f200m, prms=p.prms):
#     '''
#     Add power law to beta profile for r>r500c such that the total mass at r200m
#     equals f200m x m200m
#     '''
#     r = prms.r_range_lin
#     m200m = prms.m_range_lin
#     r200m = tools.mass_to_radius(m200m, 200 * prms.omegam * prms.rho_crit *
#                                  prms.h**2)
#     m500c = np.array([gas.m200m_to_m500c(m) for m in m200m])
#     r500c = tools.mass_to_radius(m500c, 500 * prms.rho_crit * prms.h**2)

#     # # Gas fractions
#     # gas_prms = f_gas_prms()
#     # f200m = 1 - prms.f_dm
#     # f500c = f_gas(m500c, **gas_prms[0])

#     beta = np.empty((0,), dtype=float)
#     rc = np.empty((0,), dtype=float)
#     for idx, f in enumerate(f500c):
#         sl500 = (r[idx] <= r500c[idx])
#         res = opt.minimize(minimize, [1.2, 3./2],
#                            args=(sl500, f * m500c[idx], f200m[idx] * m200m[idx],
#                                  r[idx]),
#                            bounds=((0,10), (0,3)))
#         # print res.x
#         rc = np.append(rc, res.x[0])
#         beta = np.append(beta, res.x[1])

#     m500_h = np.empty((0,), dtype=float)
#     prof_h = np.zeros_like(r)
#     for idx, prof in enumerate(prof_h):
#         sl500 = (r[idx] <= r500c[idx])

#         # parameters should ensure match at f500c for f200m
#         p = prof_beta(r[idx], rc[idx], beta[idx])
#         norm = tools.m_h(p, r[idx])
#         p *= f200m[idx] * m200m[idx] / norm

#         prof_h[idx] = p
#         m500_h = np.append(m500_h, tools.m_h(p[sl500], r[idx][sl500]))

#     m200_h = tools.m_h(prof_h, r)

#     # print 'prms: '
#     # print beta
#     # print rc/r500c
#     # print 'mass fractions: '
#     # print m200_h / m200m
#     # print m500_h / m500c
#     # print f500c

#     return beta, rc/r500c, m500c, r, prof_h

# # ------------------------------------------------------------------------------
# # End of beta_slope()
# # ------------------------------------------------------------------------------

def fit_prms():
    '''
    Return observational beta profile fit parameters
    '''
    with open('croston_500.p', 'rb') as f:
        data_c = cPickle.load(f)
    with open('eckert_500.p', 'rb') as f:
        data_e = cPickle.load(f)

    rc_e = data_e['rc']
    rc_c = data_c['rc']
    rc = np.append(rc_e, rc_c)

    b_e = data_e['b'] # equal to literature standard
    b_c = data_c['b']
    beta = np.append(b_e, b_c)

    m500c_e = data_e['m500c']
    m500c_c = data_c['m500c']
    m500c = np.append(m500c_e, m500c_c)

    return np.median(rc), np.median(beta)

# ------------------------------------------------------------------------------
# End of fit_prms()
# ------------------------------------------------------------------------------

def mass_diff(alpha, x_range, mass_ratio):
    '''
    Integrate power law such as to match mass ratio
    '''
    return tools.Integrate(x_range ** (alpha + 2.), x_range) - mass_ratio

# ------------------------------------------------------------------------------
# End of mass_diff()
# ------------------------------------------------------------------------------

def plot_profiles():
    with open('croston.p', 'rb') as f:
        data_c = cPickle.load(f)
    with open('eckert.p', 'rb') as f:
        data_e = cPickle.load(f)

    pl.set_style()
    fig = plt.figure(figsize=(18,8))
    ax1 = fig.add_axes([0.1,0.1,0.4,0.8])
    ax2 = fig.add_axes([0.5,0.1,0.4,0.8])
    # ax1 = fig.add_axes([0.05,0.1,0.3,0.8])
    # ax2 = fig.add_axes([0.35,0.1,0.3,0.8])
    # ax3 = fig.add_axes([0.65,0.1,0.3,0.8])

    rx_c = data_c['rx']
    rx_e = data_e['rx']

    rho_c = data_c['rho']
    rho_e = data_e['rho']

    ax1.set_prop_cycle(cycler(c='k', lw=[0.5]))
    for r, prof in zip(rx_c, rho_c):
        ax1.plot(r, prof, ls='-', lw=0.5, c='k')

    ax1.set_ylim([1e11, 1e16])
    ticks = ax1.get_xticklabels()
    ticks[-5].set_visible(False)

    ax1.xaxis.set_tick_params(pad=8)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$r/r_\mathrm{500c}$')
    ax1.set_ylabel(r'$\rho(r) \, [\mathrm{M_\odot/Mpc^3}]$')
    ax1.set_title(r'Croston+08')

    # get median binned profiles
    r_med, rho_med, m500_med, r500_med = bin_eckert()

    ax2.set_prop_cycle(cycler(c='k', lw=[0.5]))
    for r, prof in zip(rx_e, rho_e):
        ax2.plot(r, prof, ls='-', lw=0.5, c='k')

    # for r, prof in zip(r_med, rho_med):
    #     ax2.plot(r, prof, ls='-', lw=2, c='r')

    ax2.set_ylim([1e11, 1e16])
    ticks = ax2.get_xticklabels()
    ticks[-3].set_visible(False)

    ax2.xaxis.set_tick_params(pad=8)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_yticklabels([])
    ax2.set_xlabel(r'$r/r_\mathrm{500c}$')
    # ax2.set_ylabel(r'$\rho(r) \, [\mathrm{M_\odot/Mpc^3}]$')
    text = ax2.set_title(r'Eckert+16')
    title_props = text.get_fontproperties()

    plt.show()

# ------------------------------------------------------------------------------
# End of plot_profiles()
# ------------------------------------------------------------------------------

def plot_gas_fractions():
    pl.set_style()
    fig = plt.figure(figsize=(10,9))
    ax3 = fig.add_subplot(111)

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

    fm_prms, f1_prms, f2_prms = f_gas_prms()

    ##########################################################################
    # Get shadow twiny instance for ratio plot, to also have m200m
    # need it in this order to get correct yticks with log scale, since
    # twin instance seems to mess things up...
    axs = ax3.twiny()
    m200 = np.array([gas.m500c_to_m200m(mass) for mass in m])
    axs.plot(m200, m200)
    axs.cla()

    axs.tick_params(axis='x', pad=5)


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
    ax3.set_prop_cycle(c='k')
    ax3.plot(m, f_med, ls='-', c='k', lw=2, label='median')
    # ax3.plot(m, f_q16, ls='--', c='k', lw=2)
    # ax3.plot(m, f_q84, ls='--', c='k', lw=2)
    ax3.set_prop_cycle(c='r')
    ax3.plot(m, f_gas(m, **fm_prms), ls='-', c='r', lw=2, label='fit')
    # ax3.plot(m, f_gas(m, **f1_prms), ls='--', c='r', lw=1)
    # ax3.plot(m, f_gas(m, **f2_prms), ls='--', c='r', lw=1)

    ax3.xaxis.set_tick_params(pad=8)
    ax3.set_xlim([1e13, 10**(15.5)])
    ax3.set_ylim([0.01, 0.17])
    ax3.set_xscale('log')
    text = ax3.set_xlabel(r'$m_{500\mathrm{c}} \, [\mathrm{M_\odot}]$')
    ax3.set_ylabel(r'$f_{\mathrm{gas}}$')
    title_props = text.get_fontproperties()
    leg = ax3.legend(loc='best', numpoints=1, frameon=True, framealpha=0.8)
    leg.get_frame().set_linewidth(0.0)

    axs.set_xlim(gas.m500c_to_m200m(ax3.get_xlim()[0]),
                 gas.m500c_to_m200m(ax3.get_xlim()[-1]))
    axs.set_xscale('log')

    axs.set_xlabel(r'$m_{\mathrm{200m}} \, [\mathrm{M_\odot}]$', labelpad=10)

    f_bar = p.prms.omegab/p.prms.omegam
    ax3.axhline(y=f_bar, c='k', ls='--')
    # add annotation to f_bar
    ax3.annotate(r'$f_{\mathrm{b}}$',
                 # xy=(1e14, 0.16), xycoords='data',
                 # xytext=(1e14, 0.15), textcoords='data',
                 xy=(10**(13), f_bar), xycoords='data',
                 xytext=(1.2 * 10**(13),
                         f_bar * 0.95), textcoords='data',
                 fontproperties=title_props)

    plt.savefig('obs_gas_fractions.pdf')
    plt.show()

# ------------------------------------------------------------------------------
# End of plot_gas_fractions()
# ------------------------------------------------------------------------------

def plot_fit_profiles():
    with open('croston.p', 'rb') as f:
        data_c = cPickle.load(f)
    with open('eckert.p', 'rb') as f:
        data_e = cPickle.load(f)

    a_e, aerr_e, b_e, berr_e, msl_e, m500g_e = fit_eckert()
    a_c, aerr_c, b_c, berr_c, msl_c, m500g_c = fit_croston()

    # pl.set_style()
    fig = plt.figure(figsize=(18,8))
    ax1 = fig.add_axes([0.1,0.1,0.4,0.4])
    ax2 = fig.add_axes([0.5,0.1,0.4,0.4])
    ax3 = fig.add_axes([0.1,0.5,0.4,0.4])
    ax4 = fig.add_axes([0.5,0.5,0.4,0.4])

    rx_c = data_c['rx']
    rx_e = data_e['rx']

    r500_c = data_c['r500']
    r500_e = data_e['r500']

    rho_c = data_c['rho']
    rho_e = data_e['rho']

    ax1.set_prop_cycle(cycler(c='k', lw=[0.5]))
    m500_fit_c = np.empty((0), dtype=float)
    m500_prof_c = np.empty((0), dtype=float)
    for idx, r, prof in zip(np.arange(len(r500_c)), rx_c, rho_c):
        sl = ((prof > 0) & (rx_c[idx] >= 0.15))
        fit = prof_gas_hot(r, sl, a_c[idx], b_c[idx], msl_c[idx], r500_c[idx])
        ax3.plot(r[1:], (fit[1:] - prof[1:]) / prof[1:], ls='-', lw=0.5, c='k')

        cum_mass_fit = np.array([tools.m_h(fit[:i], r[:i] * r500_c[idx])
                                   for i in np.arange(1, r.shape[0])])
        cum_mass_prof = np.array([tools.m_h(prof[:i], r[:i] * r500_c[idx])
                                   for i in np.arange(1, r.shape[0])])
        m500_fit_c = np.append(m500_fit_c, tools.m_h(fit, r * r500_c[idx]))
        m500_prof_c = np.append(m500_prof_c, tools.m_h(prof, r * r500_c[idx]))

        ax1.plot(r[1:], (cum_mass_fit - cum_mass_prof) / m500_prof_c[idx],
                 ls='-', lw=0.5, c='k')

        # ax1.plot(r, fit, ls='--', lw=0.5, c='r')


    ax3.set_ylim([-0.6, 0.4])
    ticks = ax3.get_yticklabels()
    ticks[-6].set_visible(False)

    ax3.set_xscale('log')
    ax3.set_ylabel(r'$\rho_\mathrm{fit}(r)/\rho_\mathrm{obs}(r) - 1$')
    ax3.set_xticklabels([])
    ax3.set_title(r'Croston+08')

    ax1.set_ylim([-0.2, 0.2])
    ticks = ax1.get_xticklabels()
    ticks[-5].set_visible(False)

    ax1.xaxis.set_tick_params(pad=8)
    ax1.set_xscale('log')
    ax1.set_xlabel(r'$r/r_\mathrm{500c}$')
    ax1.set_ylabel(r'$\frac{m_\mathrm{fit}(<r) - m_\mathrm{obs}(<r)}{m_\mathrm{gas,500c}}$')

    # get median binned profiles
    r_med, rho_med, m500_med, r500_med = bin_eckert()

    ax2.set_prop_cycle(cycler(c='k', lw=[0.5]))
    m500_fit_e = np.empty((0), dtype=float)
    m500_prof_e = np.empty((0), dtype=float)
    for idx, r, prof in zip(np.arange(len(r500_med)), r_med, rho_med):
        sl = ((prof > 0) & (r >= 0.15))
        fit = prof_gas_hot(r, sl, a_e[idx], b_e[idx], msl_e[idx], r500_med[idx])
        ax4.plot(r[1:], (fit[1:] - prof[1:]) / prof[1:], ls='-', lw=0.5, c='k')

        cum_mass_fit = np.array([tools.m_h(fit[:i], r[:i] * r500_e[idx])
                                   for i in np.arange(1, r.shape[0])])
        cum_mass_prof = np.array([tools.m_h(prof[:i], r[:i] * r500_e[idx])
                                   for i in np.arange(1, r.shape[0])])
        m500_fit_e = np.append(m500_fit_e, tools.m_h(fit, r * r500_e[idx]))
        m500_prof_e = np.append(m500_prof_e, tools.m_h(prof, r * r500_e[idx]))

        ax2.plot(r[1:], (cum_mass_fit - cum_mass_prof) / m500_prof_e[idx],
                 ls='-', lw=0.5, c='k')

        # ax2.plot(r, fit, ls='--', lw=0.5, c='k')

    # for r, prof in zip(r_med, rho_med):
    #     ax2.plot(r, prof, ls='-', lw=2, c='r')

    # print np.mean(np.append(m500_fit_e / m500_prof_e,
    #                         m500_fit_c / m500_prof_c))
    # print np.median(np.append(m500_fit_e / m500_prof_e,
    #                         m500_fit_c / m500_prof_c))

    ax4.set_ylim([-0.6, 0.4])
    ax4.set_xscale('log')
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])
    ax4.set_title(r'Eckert+16')


    ax2.set_ylim([-0.2, 0.2])
    ticks = ax2.get_xticklabels()
    ticks[-3].set_visible(False)

    ax2.xaxis.set_tick_params(pad=8)
    ax2.set_xscale('log')
    ax2.set_yticklabels([])
    ax2.set_xlabel(r'$r/r_\mathrm{500c}$')
    plt.show()

# ------------------------------------------------------------------------------
# End of plot_fit_profiles()
# ------------------------------------------------------------------------------

def corr_fit(x, a, b):
    return a + b * x

def plot_correlation():
    with open('croston_500.p', 'rb') as f:
        data_c = cPickle.load(f)
    with open('eckert_500.p', 'rb') as f:
        data_e = cPickle.load(f)
    beta, rc, m500c, r, prof_h = beta_mass()

    m500c_d = np.append(data_c['m500c'], data_e['m500c'])
    rc_d = np.append(data_c['rc'], data_e['rc'])
    beta_d = np.append(data_c['b'], data_e['b'])

    sl = (rc_d > 1e-2)

    copt, ccov = opt.curve_fit(corr_fit, np.log10(rc_d[sl]), np.log10(beta_d[sl]))
    corr_prms = {"a": copt[0],
                 "b": copt[1]}

    fig = plt.figure(figsize=(20,6))
    fig.subplots_adjust(left=0.05, right=0.95)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    pl.set_style()
    # Plot rc mass dependence
    ax1.set_prop_cycle(pl.cycle_mark())
    ax1.plot(data_e['m500c'], data_e['rc'], label=r'Eckert+16')
    ax1.plot(data_c['m500c'], data_c['rc'], label=r'Croston+08')
    ax1.plot(m500c, rc, label=r'$m_\mathrm{200m}$ matched')
    ax1.set_xlabel(r'$m_\mathrm{500c}/\mathrm{M_\odot}$')
    ax1.set_ylabel(r'$r_c/r_\mathrm{500c}$')
    ax1.set_ylim(ymin=1e-2)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(loc='best')

    # Plot rc mass dependence
    ax2.set_prop_cycle(pl.cycle_mark())
    ax2.plot(data_e['m500c'], data_e['b'], label=r'Eckert+16')
    ax2.plot(data_c['m500c'], data_c['b'], label=r'Croston+08')
    ax2.plot(m500c, beta, label=r'$m_\mathrm{200m}$ matched')
    ax2.set_xlabel(r'$m_\mathrm{500c}/\mathrm{M_\odot}$')
    ax2.set_ylabel(r'$\beta$')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend(loc='best')

    ############################################################################
    # Plot correlation
    ax3.scatter(np.median(rc_d), np.median(beta_d), c='r', marker='x', s=40)
    s1 = ax3.scatter(data_e['rc'], data_e['b'], marker='o',
                    c=np.log10(data_e['m500c']),
                    label=r'Eckert+16', cmap='magma', lw=0)
    s2 = ax3.scatter(data_c['rc'], data_c['b'], marker='D',
                    c=np.log10(data_c['m500c']),
                    label=r'Croston+08', cmap='magma', lw=0)
    s3 = ax3.scatter(rc, beta, marker='^',
                    c=np.log10(m500c), label=r'$m_\mathrm{200m}$ matched',
                    cmap='magma', lw=0)

    # Plot fit to correlation
    r_bins = np.logspace(np.log10(rc_d[sl].min()), np.log10(rc_d[sl].max()), 10)
    r = tools.bins2center(r_bins)
    r_bin_idx = np.digitize(rc_d[sl], r_bins)
    med = np.array([np.median(beta_d[sl][r_bin_idx == r_bin]) for r_bin in
                    np.arange(1, len(r_bins))])
    # q16 = np.array([np.percentile(beta_d[sl][r_bin_idx == r_bin], 16) for r_bin in
    #                 np.arange(1, len(r_bins))])
    # q84 = np.array([np.percentile(beta_d[sl][r_bin_idx == r_bin], 84) for r_bin in
    #                 np.arange(1, len(r_bins))])
    ax3.set_prop_cycle(pl.cycle_line())
    ax3.plot(r, np.power(10, corr_fit(np.log10(r), **corr_prms)),
            ls='-', c='k', lw=1)
    ax3.plot(r, med, ls='--', c='k', lw=0.5)
    # ax3.plot(r, q16, ls='-.', c='k', lw=0.5)
    # ax3.plot(r, q84, ls='-.', c='k', lw=0.5)
    # ax3.set_xlim(xmin=1e-3)
    ax3.set_xlabel(r'$r_c/r_\mathrm{500c}$')
    ax3.set_ylabel(r'$\beta$')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    cb = fig.colorbar(s1)
    cb.set_label(r'$\log_{10} m_\mathrm{500c}/\mathrm{M_\odot}$',
                 rotation=270, labelpad=25)

    # Set label color to black
    s1.set_color('k')
    s2.set_color('k')
    s3.set_color('k')
    handles = [s1, s2, s3]
    labs = [s1.get_label(), s2.get_label(), s3.get_label()]
    ax3.legend(handles, labs, loc=1)

    plt.show()

# ------------------------------------------------------------------------------
# End of plot_correlation()
# ------------------------------------------------------------------------------

def plot_profiles_croston_presentation():
    with open('croston.p', 'rb') as f:
        data_c = cPickle.load(f)

    fig1 = plt.figure(1,figsize=(10,9))
    ax1 = fig1.add_subplot(111)

    rx_c = data_c['rx']
    rho_c = data_c['rho']

    pc = cycler(c=['k'], ls=['-'])
    ax1.set_prop_cycle(pc)
    for r, prof in zip(rx_c, rho_c):
        ax1.plot(r, prof, lw=0.5)

    ax1.set_ylim([1e11, 1e16])

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$r/r_\mathrm{500c}$')
    ax1.set_ylabel(r'$\rho(r) \, [\mathrm{M_\odot/Mpc^3}]$')
    ax1.set_title(r'Croston+08')

    plt.savefig('obs_croston.pdf', transparent=True)
    plt.show()

# ------------------------------------------------------------------------------
# End of plot_profiles_croston_presentation()
# ------------------------------------------------------------------------------

def plot_profiles_eckert_presentation():
    with open('eckert.p', 'rb') as f:
        data_e = cPickle.load(f)

    pl.set_style('line')
    fig1 = plt.figure(1,figsize=(10,9))
    ax1 = fig1.add_subplot(111)

    rx_e = data_e['rx']
    rho_e = data_e['rho']

    pc = cycler(c=['k'], ls=['-'])
    ax1.set_prop_cycle(pc)
    for r, prof in zip(rx_e, rho_e):
        ax1.plot(r, prof, lw=0.5)

    ax1.set_ylim([1e11, 1e16])

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$r/r_\mathrm{500c}$')
    ax1.set_ylabel(r'$\rho(r) \, [\mathrm{M_\odot/Mpc^3}]$')
    ax1.set_title(r'Eckert+16')

    plt.savefig('obs_eckert.pdf', transparent=True)
    plt.show()

# ------------------------------------------------------------------------------
# End of plot_profiles_croston_presentation()
# ------------------------------------------------------------------------------

def plot_profiles_fgas_presentation():
    fig1 = plt.figure(1,figsize=(10,9))
    ax3 = fig1.add_subplot(111)
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

    fm_prms, f1_prms, f2_prms = f_gas_prms()

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


    ax3.plot(m, f_med, ls='-', c='k', lw=2, marker='None')
    ax3.plot(m, f_q16, ls='--', c='k', lw=2, marker='None')
    ax3.plot(m, f_q84, ls='--', c='k', lw=2, marker='None')
    ax3.plot(m, f_gas(m, **fm_prms), ls='-', c='r', lw=1, marker='None')
    ax3.plot(m, f_gas(m, **f1_prms), ls='--', c='r', lw=1, marker='None')
    ax3.plot(m, f_gas(m, **f2_prms), ls='--', c='r', lw=1, marker='None')
    text = ax3.set_title(r'Gas fractions')
    title_props = text.get_fontproperties()

    f_bar = p.prms.omegab/p.prms.omegam
    ax3.axhline(y=f_bar, c='k', ls='--')
    # add annotation to f_bar
    ax3.annotate(r'$f_{\mathrm{b}}$',
                 # xy=(1e14, 0.16), xycoords='data',
                 # xytext=(1e14, 0.15), textcoords='data',
                 xy=(10**(13), f_bar), xycoords='data',
                 xytext=(1.2 * 10**(13),
                         f_bar * 0.95), textcoords='data',
                 fontproperties=title_props)

    ax3.set_xlim([1e13, 10**(15.5)])
    ax3.set_ylim([0.01, 0.17])
    ax3.set_xscale('log')
    ax3.set_xlabel(r'$m_{500\mathrm{c}} \, [\mathrm{M_\odot}]$')
    ax3.set_ylabel(r'$f_{\mathrm{gas,500c}}$')
    leg = ax3.legend(loc='best', numpoints=1)

    plt.savefig('obs_fgas.pdf', transparent=True)
    plt.show()

# ------------------------------------------------------------------------------
# End of plot_profiles_presentation()
# ------------------------------------------------------------------------------

def plot_fit_prms_presentation():
    with open('croston_200.p', 'rb') as f:
        data_c = cPickle.load(f)
    with open('eckert_200.p', 'rb') as f:
        data_e = cPickle.load(f)

    rc_e = data_e['rc']
    rc_c = data_c['rc']
    rc = np.append(rc_e, rc_c)

    b_e = data_e['b'] / 3. # equal to literature standard
    b_c = data_c['b'] / 3.
    beta = np.append(b_e, b_c)

    m200m_e = data_e['m200m']
    m200m_c = data_c['m200m']
    m200m = np.append(m200m_e, m200m_c)

    pl.set_style('mark')

    fig = plt.figure(figsize=(10,9))
    ax = fig.add_subplot(111)

    ax.plot(m200m_e, rc_e, label='Eckert+16')
    ax.plot(m200m_c, rc_c, label='Croston+08')
    ax.axhline(y=np.median(rc), ls='-', c='k')
    ax.annotate('median', xy=(1.05 * m200m.min(), 1.05 * np.median(rc)))
    ax.set_xscale('log')
    ax.set_xlabel('$m_{\mathrm{200m}}\, [\mathrm{M_\odot}]$')
    ax.set_ylabel('$r_\mathrm{c}/r_{\mathrm{200m}}$')
    ax.legend(loc='best')
    plt.savefig('obs_rc_fit.pdf', transparent=True)

    plt.clf()
    fig = plt.figure(figsize=(10,9))
    ax = fig.add_subplot(111)

    ax.plot(m200m_e, b_e, label='Eckert+16')
    ax.plot(m200m_c, b_c, label='Croston+08')
    ax.axhline(y=np.median(beta), ls='-', c='k')
    ax.annotate('median', xy=(1.05 * m200m.min(), 1.05 * np.median(beta)))
    ax.set_xscale('log')
    ax.set_xlabel('$m_{\mathrm{200m}}\, [\mathrm{M_\odot}]$')
    ax.set_ylabel(r'$\beta$')
    ax.legend(loc='best')
    plt.savefig('obs_beta_fit.pdf', transparent=True)

# ------------------------------------------------------------------------------
# End of plot_fit_prms_presentation()
# ------------------------------------------------------------------------------

def plot_correlation_presentation():
    with open('croston_500.p', 'rb') as f:
        data_c = cPickle.load(f)
    with open('eckert_500.p', 'rb') as f:
        data_e = cPickle.load(f)
    beta, rc, m500c, r, prof_h = beta_mass()

    m500c_d = np.append(data_c['m500c'], data_e['m500c'])
    rc_d = np.append(data_c['rc'], data_e['rc'])
    beta_d = np.append(data_c['b'], data_e['b'])

    sl = (rc_d > 1e-2)

    copt, ccov = opt.curve_fit(corr_fit, np.log10(rc_d[sl]), np.log10(beta_d[sl]))
    corr_prms = {"a": copt[0],
                 "b": copt[1]}

    fig = plt.figure(figsize=(10,9))
    ax3 = fig.add_subplot(111)

    pl.set_style('mark')
    ############################################################################
    # Plot correlation
    ax3.scatter(np.median(rc_d), np.median(beta_d), c='r', marker='x', s=80)
    s1 = ax3.scatter(data_e['rc'], data_e['b'], marker='o',
                     c=np.log10(data_e['m500c']),
                     label=r'Eckert+16', cmap='magma', lw=0, s=80)
    s2 = ax3.scatter(data_c['rc'], data_c['b'], marker='D',
                    c=np.log10(data_c['m500c']),
                    label=r'Croston+08', cmap='magma', lw=0, s=80)
    s3 = ax3.scatter(rc, beta, marker='^',
                    c=np.log10(m500c), label=r'$m_\mathrm{200m}$ matched',
                    cmap='magma', lw=0, s=80)

    # Plot fit to correlation
    r_bins = np.logspace(np.log10(rc_d[sl].min()), np.log10(rc_d[sl].max()), 10)
    r = tools.bins2center(r_bins)
    r_bin_idx = np.digitize(rc_d[sl], r_bins)
    med = np.array([np.median(beta_d[sl][r_bin_idx == r_bin]) for r_bin in
                    np.arange(1, len(r_bins))])
    # q16 = np.array([np.percentile(beta_d[sl][r_bin_idx == r_bin], 16) for r_bin in
    #                 np.arange(1, len(r_bins))])
    # q84 = np.array([np.percentile(beta_d[sl][r_bin_idx == r_bin], 84) for r_bin in
    #                 np.arange(1, len(r_bins))])
    ax3.plot(r, np.power(10, corr_fit(np.log10(r), **corr_prms)),
             ls='-', c='k', lw=1, marker='None')
    ax3.plot(r, med, ls='--', c='k', lw=0.5, marker='None')
    # ax3.plot(r, q16, ls='-.', c='k', lw=0.5)
    # ax3.plot(r, q84, ls='-.', c='k', lw=0.5)
    # ax3.set_xlim(xmin=1e-3)
    ax3.set_xlabel(r'$r_c/r_\mathrm{500c}$')
    ax3.set_ylabel(r'$\beta$')
    ax3.set_xscale('log')
    ax3.set_ylim([1,4])
    ax3.set_yscale('log')
    ax3.set_yticks([1,2,3,4])
    # ax3.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax3.set_yticklabels(["1","2","3","4"])
    cb = fig.colorbar(s1)
    cb.set_label(r'$\log_{10} m_\mathrm{500c}/\mathrm{M_\odot}$',
                 rotation=270, labelpad=30)

    # Set label color to black
    s1.set_color('k')
    s2.set_color('k')
    s3.set_color('k')
    handles = [s1, s2, s3]
    labs = [s1.get_label(), s2.get_label(), s3.get_label()]
    # handles = [s1, s2]
    # labs = [s1.get_label(), s2.get_label()]
    ax3.legend(handles, labs, loc=2)
    plt.savefig('obs_corr_matched.pdf', transparent=True)

    plt.show()

# ------------------------------------------------------------------------------
# End of plot_correlation_presentation()
# ------------------------------------------------------------------------------

def plot_beta_presentation():
    with open('croston.p', 'rb') as f:
        data_croston = cPickle.load(f)

    r500 = data_croston['r500']
    m500g = data_croston['m500gas']
    rx = data_croston['rx']
    rho = data_croston['rho']

    idx = 25

    r = rx[idx]
    prof = rho[idx]

    rc, rc_err, beta, beta_err, m_sl, m500g = fit_croston()

    fig = plt.figure(figsize=(10,9))
    ax = fig.add_subplot(111)

    sl = ((prof > 0) & (r >= 0.05))
    mass = tools.m_h(prof[sl], r[sl] * r500[idx])
    print '%e'%tools.radius_to_mass(r500[idx], 500 * p.prms.rho_crit * p.prms.h**2)

    pl.set_style('mark')
    ax.plot(r, prof, lw=0, marker='o')
    ax.set_prop_cycle(pl.cycle_line())
    ax.plot(r, prof_gas_hot(r, sl, rc[idx], beta[idx], mass, r500[idx]))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$r/r_\mathrm{500c}$')
    ax.set_ylabel(r'$\rho(r) \, [\mathrm{M_\odot/Mpc^3}]$')

    plt.savefig('obs_beta.pdf', transparent=True)
    plt.show()


# ------------------------------------------------------------------------------
# End of plot_beta_presentation()
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

    m500c = np.array([gas.m200m_to_m500c(mass) for mass in m_range[::5]])
    f500c = d.f_gas(m500c, **d.f_gas_prms()[0])

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
