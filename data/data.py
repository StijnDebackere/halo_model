import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.interpolate as interp
import astropy.constants as const
import astropy.units as u
import astropy.io.fits as fits
import glob
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

ddir = '/Volumes/Data/stijn/Documents/Universiteit/MR/code/halo/data/'
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
    # rho = mu * m_p * n_gas -> fully ionised: mu=0.59 (X=0.75, Y=0.25)
    # n_gas = n_e + n_H + n_He = 2n_H + 3n_He (fully ionized)
    #       = (2 + 3Y/(4X))n_H
    # n_He = Y/(4X) n_H
    # n_e = (2 + 3Y/(4X)) / (1 + Y/(4X)) n
    # => n_gas = 2.07 n_e
    # calculate correct mu factor for Z metallicity gas
    n2rho = 1.92 * 0.59 * const.m_p.cgs * 1/u.cm**3 # in cgs
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

        # plt.plot(rx[idx], prof)
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.title('$m=10^{%.2f}\mathrm{M_\odot}$'%np.log10(m500[sysnum[idx] - 1]))
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
    # rho = mu * m_p * n_gas -> fully ionised: mu=0.59 (X=0.75, Y=0.25)
    # n_gas = n_e + n_H + n_He = 2n_H + 3n_He (fully ionized)
    # n_He = Y/(4X) n_H
    # => n_gas = 2.25 n_H
    n2rho = 2.25 * 0.59 * const.m_p.cgs * 1/u.cm**3 # in cgs
    cgs2cos = (1e6 * const.pc.cgs)**3 / const.M_sun.cgs

    r500 = np.empty((0,), dtype=float)
    z = np.empty((0,), dtype=float)
    rx = []
    rho = []
    rho_err = []
    for f in files:
        # actual data
        data = fits.open(f)
        # !!!! check whether this still needs correction factor from weak lensing
        r500 = np.append(r500, data[1].header['R500'] * 1e-3) # [Mpc]
        z = np.append(z, data[1].header['REDSHIFT'])
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

    m500gas = mgas500[mdata2data]

    print m500mt[mdata2data]
    print 4./3 * np.pi * 500 * p.prms.rho_crit * r500**3

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

def bin_eckert():
    with open('eckert.p', 'rb') as f:
        data_eckert = cPickle.load(f)

    r500 = data_eckert['r500']
    m500g = data_eckert['m500gas']
    rx = data_eckert['rx']
    rho = data_eckert['rho']

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
        rho_new = np.empty((1, n_r))
        for idx in idx_in_bin.nonzero()[0]:
            f_rho = interp.interp1d(rx[idx], rho[idx])
            rho_new = np.concatenate([rho_new, f_rho(r_range).reshape(1,-1)],
                                     axis=0)

        r_ranges[idx_m] = r_range
        rho_med[idx_m] = np.median(rho_new, axis=0)

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

def fit_croston():
    '''
    Fit profiles to the observations
    '''
    with open('croston.p', 'rb') as f:
        data_croston = cPickle.load(f)

    r500 = data_croston['r500']
    m500g = data_croston['m500gas']
    rx = data_croston['rx']
    rho = data_croston['rho']

    pl.set_style()
    a = np.empty((0,), dtype=float)
    b = np.empty((0,), dtype=float)
    # c = np.empty((0,), dtype=float)
    aerr = np.empty((0,), dtype=float)
    berr = np.empty((0,), dtype=float)
    # cerr = np.empty((0,), dtype=float)
    for idx, prof in enumerate(rho):
        sl = (rx[idx] >= 0.05)
        r = rx[idx]
        mass = tools.m_h(prof[sl], r[sl] * r500[idx])
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

        # plt.plot(r, r**2 * prof)
        # plt.plot(r, r**2 * prof_gas_hot(r, sl, popt[0], popt[1], # , popt[2],
        #                                   mass, r500[idx]))
        # plt.title('%i'%idx)
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()

        a = np.append(a, popt[0])
        b = np.append(b, popt[1])
        # c = np.append(c, popt[2])
        aerr = np.append(aerr, np.sqrt(np.diag(pcov)))[0]
        berr = np.append(berr, np.sqrt(np.diag(pcov)))[1]
        # cerr = np.append(cerr, np.sqrt(np.diag(pcov)))[2]

    return a, aerr, b, berr, m500g # c, cerr, m500g

# ------------------------------------------------------------------------------
# End of fit_croston()
# ------------------------------------------------------------------------------

def fit_eckert():
    rx, rho, m500g, r500 = bin_eckert()

    pl.set_style()
    a = np.empty((0,), dtype=float)
    b = np.empty((0,), dtype=float)
    # c = np.empty((0,), dtype=float)
    aerr = np.empty((0,), dtype=float)
    berr = np.empty((0,), dtype=float)
    # cerr = np.empty((0,), dtype=float)
    for idx, prof in enumerate(rho):
        sl = (rx[idx] >= 0.05)
        r = rx[idx]
        mass = tools.m_h(prof[sl], r[sl] * r500[idx])
        sl_fit = np.ones(sl.sum(), dtype=bool)
        popt, pcov = opt.curve_fit(lambda r, a, b: \
                                   # , c:\
                                   prof_gas_hot(r, sl_fit, a, b, # , c,
                                                  mass, r500[idx]),
                                   # r[sl], prof[sl], bounds=([0, 0, 0.5],
                                   #                          [1, 5, 10]))
                                   r[sl], prof[sl], bounds=([0, 0],
                                                            [1, 5]))

        # plt.plot(r, prof)
        # plt.plot(r, prof_gas_hot(r, sl, popt[0], popt[1], # popt[2],
        #                            mass, r500[idx]))
        # plt.title('%i'%idx)
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()

        a = np.append(a, popt[0])
        b = np.append(b, popt[1])
        # c = np.append(c, popt[2])
        aerr = np.append(aerr, np.sqrt(np.diag(pcov)))[0]
        berr = np.append(berr, np.sqrt(np.diag(pcov)))[1]
        # cerr = np.append(cerr, np.sqrt(np.diag(pcov)))[2]


    return a, aerr, b, berr, m500g # c, cerr, m500g

# ------------------------------------------------------------------------------
# End of fit_eckert()
# ------------------------------------------------------------------------------

def convert_hm():
    '''
    Save fit parameters and masses for both m500c & m200m
    '''
    # rcc, rccerr, bc, bcerr, rxc, rxcerr, mgc = fit_croston()
    # rce, rceerr, be, beerr, rxe, rxeerr, mge = fit_eckert()
    rcc, rccerr, bc, bcerr, mgc = fit_croston()
    rce, rceerr, be, beerr, mge = fit_eckert()

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
                     marker='x', label=r'Eckert+15')
        ax1.set_xlabel(r'$m_{200\mathrm{m}} \, [\mathrm{M}_\odot]$')
        ax1.set_title(r'$r_c/r_{200\mathrm{m}}$')
    else:
        ax1.errorbar(data_c['m500c'], data_c['rc'], yerr=data_c['rcerr'],
                     marker='o', label=r'Croston+08')
        ax1.errorbar(data_e['m500c'], data_e['rc'], yerr=data_e['rcerr'],
                     marker='x', label=r'Eckert+15')
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
                     label=r'Eckert+15')
        ax2.set_xlabel(r'$m_{200\mathrm{m}} \, [\mathrm{M}_\odot]$')
    else:
        ax2.errorbar(data_c['m500c'], data_c['b'], yerr=data_c['berr'], marker='o',
                     label=r'Croston+08')
        ax2.errorbar(data_e['m500c'], data_e['b'], yerr=data_e['berr'], marker='x',
                     label=r'Eckert+15')
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
    #                  marker='x', label=r'Eckert+15')
    #     ax3.set_xlabel(r'$m_{200\mathrm{m}} \, [\mathrm{M}_\odot]$')
    #     ax3.set_title(r'$r_x/r_{200\mathrm{m}}$')
    # else:
    #     ax3.errorbar(data_c['m500c'], data_c['rx'], yerr=data_c['rxerr'],
    #                  marker='o', label=r'Croston+08')
    #     ax3.errorbar(data_e['m500c'], data_e['rx'], yerr=data_e['rxerr'],
    #                  marker='x', label=r'Eckert+15')
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
    return (p.prms.omegab/p.prms.omegam - f0) / (1 + (mc/m)**a) + f0

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

    rc_med = np.median(rc)
    rc_q16 = np.percentile(rc, 16)
    rc_q84 = np.percentile(rc, 84)
    b_med = np.median(b)
    b_q16 = np.percentile(b, 16)
    b_q84 = np.percentile(b, 84)

    return rc_med, rc_q16, rc_q84, b_med, b_q16, b_q84

# ------------------------------------------------------------------------------
# End of prof_prms()
# ------------------------------------------------------------------------------