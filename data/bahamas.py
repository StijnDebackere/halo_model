import numpy as np
import scipy.optimize as opt
import h5py
import sys
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import palettable
from cycler import cycler

# allow import of plot
sys.path.append('~/Documents/Universiteit/MR/code')
import plot as pl

import halo.parameters as p
import halo.tools as tools
import halo.gas as gas
import halo.density_profiles as profs

import pdb

cm2mpc = 3.2407788498994389e-25

def load_data(ptypes=[0,1,4]):
    ddir = '/Volumes/Data/stijn/Documents/Universiteit/MR/code/halo/data/'
    # prof = ddir + 'BAHAMAS/eagle_subfind_particles_032_profiles.hdf5'
    bins = ddir + 'BAHAMAS/eagle_subfind_particles_032_profiles_binned.hdf5'
    # profs = h5py.File(ddir + prof, 'r')
    binned = h5py.File(bins, 'r')

    rho_med = []
    rho_mean = []
    std = []
    q16 = []
    q84 = []
    mass_med = []
    mass_mean = []
    m200 = []
    m500 = []
    nbin = []
    for ptype in ptypes:
        rho_med.append(binned['PartType%i/MedianDensity'%ptype][:])
        rho_mean.append(binned['PartType%i/MeanDensity'%ptype][:])
        std.append(binned['PartType%i/StD'%ptype][:])
        q16.append(binned['PartType%i/Q16'%ptype][:])
        q84.append(binned['PartType%i/Q84'%ptype][:])
        mass_med.append(binned['PartType%i/MedianM200'%ptype][:])
        mass_mean.append(binned['PartType%i/MeanM200'%ptype][:])
        m200.append(binned['PartType%i/M200'%ptype][:])
        m500.append(binned['PartType%i/M500'%ptype][:])
        nbin.append(binned['PartType%i/NumBin'%ptype][:])

    m_bins = binned['MBins_M_Mean200'][:]
    r_bins = binned['RBins_R_Mean200'][:]
    binned.close()

    data = {'rho_med' : np.array(rho_med),
            'rho_mean' : np.array(rho_mean),
            'std' : np.array(std),
            'q16' : np.array(q16),
            'q84' : np.array(q84),
            'm_med' : np.array(mass_med),
            'm_mean' : np.array(mass_mean),
            'm200' : np.array(m200),
            'm500' : np.array(m500),
            'nbin' : np.array(nbin),
            'm_bins' : m_bins,
            'r_bins' : r_bins}

    return data

# ------------------------------------------------------------------------------
# End of load_data()
# ------------------------------------------------------------------------------

def compare_method_eckert():
    '''
    Compare different gas mass derivations consistency with m500
    '''
    r_eckert, rho_eckert, s, mwl, mgas = gas.rhogas_eckert()
    # only integrate up to 1 r500
    m_x = tools.m_h(rho_eckert[:,:-1], r_eckert[:-1].reshape(1,-1), axis=-1)
    mgas_f = mwl * gas.f_gas(mwl, **gas.f_gas_fit())

    # get r500 from ratio between r & x=r/r500 integration
    r500_f_eckert = (mgas_f / m_x)**(1./3)
    r500_eckert = (mgas/m_x)**(1./3)

    m500_f_eckert = tools.m_delta(r500_f_eckert, 500, p.prms.rho_crit * 0.7**2)
    m500_eckert = tools.m_delta(r500_eckert, 500, p.prms.rho_crit * 0.7**2)

    print 'm500 from f_gas : ', m500_f_eckert
    print 'ratio wrt mwl   : ', m500_f_eckert / mwl
    print 'f_gas determined: ', mgas_f / m500_f_eckert
    print 'f_gas actual    : ', mgas_f / mwl
    print 'm500 from T_gas : ', m500_eckert
    print 'ratio wrt mwl   : ', m500_eckert / mwl
    print 'f_gas determined: ', mgas / m500_eckert
    print 'f_gas actual    : ', mgas / mwl

# ------------------------------------------------------------------------------
# End of compare_eckert()
# ------------------------------------------------------------------------------

def m_bins_eckert():
    '''
    Return m_bins to be used in profiles to slice BAHAMAS data as to reproduce
    the Eckert masses
    '''
    # bahamas data
    profiles = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_500_crit.hdf5', 'r')

    m500 = np.sort(profiles['PartType0/M500'][:])
    profiles.close()

    # eckert data
    data = np.loadtxt('halo/data/data_mccarthy/gas/xxl_table.csv',
                      delimiter=',', unpack=True)
    z, T300, r500, Mgas_500, Yx_500, fgas_500, used = data
    # r500 is in kpc, rho_crit in Mpc
    M500 = 4./3 * np.pi * 500 * p.prms.rho_crit * 0.7**2 * (0.001 * r500)**3
    M500 /= 1.3 # faulty weak lensing mass determination
    m_bins = []
    m_bins.append(M500[T300 < 2].mean())
    m_bins.append(M500[(2 < T300) & (T300 < 3)].mean())
    m_bins.append(M500[(3 < T300) & (T300 < 4)].mean())
    m_bins.append(M500[4 < T300].mean())
    m_bins = np.array(m_bins)

    # medians coincide approximately with means (double checked this)
    slices = tools.median_slices(m500, m_bins,
                                 m_bins.reshape(-1,1) *
                                 np.array([0.75, 1.25]).reshape(-1,2))

    m_bins_eckert = m500[slices]

    return m_bins_eckert, m_bins

# ------------------------------------------------------------------------------
# End of m_bins_eckert()
# ------------------------------------------------------------------------------

def m_bins_sun():
    '''
    Return m_bins to be used in profiles to slice BAHAMAS data as to reproduce
    the Sun mass
    '''
    # bahamas data
    profiles = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_500_crit.hdf5', 'r')

    m500 = np.sort(profiles['PartType0/M500'][:])
    profiles.close()

    # sun data
    r500, rho, err = gas.rhogas_sun()
    M_sun = gas.rhogas_sun.M

    slices = tools.median_slices(m500, np.array([M_sun]),
                                 np.array([0.75, 1.25]) * M_sun)
    m_bins_sun = np.array([m500[s] for s in slices])
    return m_bins_sun

# ------------------------------------------------------------------------------
# End of m_bins_sun()
# ------------------------------------------------------------------------------

def m_bins_croston():
    '''
    Return m_bins to be used in profiles to slice BAHAMAS data as to reproduce
    the Croston mass
    '''
    # bahamas data
    profiles = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_500_crit.hdf5', 'r')

    m500 = np.sort(profiles['PartType0/M500'][:])
    profiles.close()

    # croston data
    r500, rho, err = gas.rhogas_croston()
    M_croston = gas.rhogas_croston.M

    slices = tools.median_slices(m500, np.array([M_croston]),
                                 np.array([0.75, 1.25]) * M_croston)
    m_bins_croston = np.array([m500[s] for s in slices])
    return m_bins_croston

# ------------------------------------------------------------------------------
# End of m_bins_croston()
# ------------------------------------------------------------------------------

def compare_med_mean(ptype, sliced=slice(None,None)):
    '''
    Compare the median and mean profiles in the BAHAMAS simulations.
    '''
    data = load_data(ptypes=[ptype])
    r = tools.bins2center(data['r_bins'])
    # m = tools.bins2center(data['m_bins'])

    pl.set_style()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    meds = data['rho_med'][0][sliced]
    q16 = data['q16'][0][sliced]
    q84 = data['q84'][0][sliced]
    m_med = data['m_med'][0][sliced]
    mean = data['rho_mean'][0][sliced]
    std = data['std'][0][sliced]
    m_mean = data['m_mean'][0][sliced]
    lines = []
    fills = []
    for idx, med in enumerate(meds):
        l_med, = ax.plot(r, med)
        f_med = ax.fill_between(r, q16[idx], q84[idx],
                                facecolor=l_med.get_color(),
                                alpha=0.2,
                                linewidth=0)
        l_mean, = ax.plot(r, mean[idx])
        f_mean = ax.fill_between(r, mean[idx] - std[idx], mean[idx] + std[idx],
                                 facecolor=l_mean.get_color(),
                                 alpha=0.2,
                                 linewidth=0)
        lines.extend([l_med, l_mean])
        fills.extend([f_med, f_mean])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$r/r_{200}$')
    ax.set_ylabel(r'$\rho(r)$ in $M_\odot/$Mpc$^3$')

    masses = np.empty((m_med.size + m_mean.size))
    masses[0::2] = m_med
    masses[1::2] = m_mean
    ax.legend([(line, fill) for line,fill in zip(lines, fills)],
              [r'$M_{\mathrm{med}}=10^{%.2f}M_\odot$'%np.log10(m)
               if idx%2 == 0 else
               r'$M_{\mathrm{mean}}=10^{%.2f}M_\odot$'%np.log10(m)
               for idx, m in enumerate(masses)])

    plt.show()

# ------------------------------------------------------------------------------
# End of compare_med_mean()
# ------------------------------------------------------------------------------

def plot_fgas_bahamas():
    '''
    Plot gas fraction in BAHAMAS vs our observed relation
    '''
    # bahamas data
    profiles = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_500_crit.hdf5', 'r')
    m500 = profiles['PartType0/M500'][:]
    r500 = profiles['PartType0/R500'][:]
    rho = profiles['PartType0/Densities'][:]
    numpart = profiles['PartType0/NumPartGroup'][:]
    grnr = profiles['PartType0/GroupNumber'][:]

    r_bins = profiles['RBins_R_Crit500'][:]
    r = tools.bins2center(r_bins)
    r500 *= cm2mpc

    # m_bins = binned['MBins_M_Crit500'][:]
    # m = tools.bins2center(m_bins)
    binned = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_binned_500_crit.hdf5', 'r')
    m_bins = np.logspace(13, 15, 10)
    m_bins = np.concatenate([m_bins[:-1].reshape(-1,1),
                             m_bins[1:].reshape(-1,1)],
                            axis=1)
    m = tools.bins2center(m_bins)


    # cut sample
    idx_cut = ((numpart[grnr] > 10000) & (m500 >= 1e13))
    # idx_cut = (m500 >= 1e13)

    mgas = tools.m_h(rho, r.reshape(1,-1) * r500.reshape(-1,1), axis=-1)

    # bin sample
    f_binned = np.ones_like(m)
    q16 = np.ones_like(m)
    q84 = np.ones_like(m)
    for idx, m_bin in enumerate(m_bins):
        sl = ((m500[idx_cut] >= m_bin[0]) & (m500[idx_cut] < m_bin[1]))
        f = mgas[idx_cut][sl]/m500[idx_cut][sl]
        f_binned[idx] = np.median(f)
        q16[idx] = np.percentile(f, 16)
        q84[idx] = np.percentile(f, 84)

    pl.set_style()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # img = ax.hexbin(np.log10(m500[idx_cut]), (mgas/m500)[idx_cut], cmap='magma',
    #                 bins='log')
    ax.plot(np.log10(m), f_binned, c='r', label=r'BAHAMAS')
    ax.plot(np.log10(m), q16, c='r', ls='--')
    ax.plot(np.log10(m), q84, c='r', ls='--')

    # plot observed fractions
    m500_obs, f_obs = np.loadtxt('halo/data/data_mccarthy/gas/M500_fgas_BAHAMAS_data.dat',
                         unpack=True)
    ax.set_prop_cycle(pl.cycle_mark())
    ax.plot(m500_obs[23:33], f_obs[23:33],
            c='k', marker='o', markerfacecolor='k',
            label=r'Vikhlinin+2006')
    ax.plot(m500_obs[128:165], f_obs[128:165],
            c='k', marker='^', markerfacecolor='k',
            label=r'Maughan+2008')
    ax.plot(m500_obs[:23], f_obs[:23],
            c='k', marker='v', markerfacecolor='k',
            label=r'Sun+2009')
    ax.plot(m500_obs[33:64], f_obs[33:64],
            c='k', marker='<', markerfacecolor='k',
            label=r'Pratt+2009')
    ax.plot(m500_obs[64:128], f_obs[64:128],
            c='k', marker='>', markerfacecolor='k',
            label=r'Lin+2012')
    ax.plot(m500_obs[165:], f_obs[165:],
            c='k', marker='D', markerfacecolor='k',
            label=r'Lovisari+2015')

    ax.set_prop_cycle(pl.cycle_line())
    # plot fit to observed fractions
    m_range = np.logspace(np.log10(np.nanmin(m500[idx_cut])),
                          np.log10(np.nanmax(m500[idx_cut])), 100)
    ax.plot(np.log10(m_range), gas.f_gas(m_range, **gas.f_gas_fit()),
            c='b', ls='-.',
            label=r'fit')

    ax.set_xlim([13, 15.25])
    ax.set_ylim([0.02, 0.145])
    ax.set_xlabel(r'$M_{500} \, [\log_{10}M_\odot]$')
    ax.set_ylabel(r'$f_{\mathrm{gas}}$')

    # for line in ax.xaxis.get_ticklines():
    #     line.set_color('w')
    # for line in ax.yaxis.get_ticklines():
    #     line.set_color('w')

    # cb = fig.colorbar(img)
    # cb.set_label(r'$\log_{10} N_{\mathrm{bin}}$', rotation=270, labelpad=25)

    leg = ax.legend(loc='best', numpoints=1)
    # for text in leg.get_texts():
    #     plt.setp(text, color='w')

    plt.show()

    profiles.close()
    binned.close()

# ------------------------------------------------------------------------------
# End of plot_fgas_bahamas()
# ------------------------------------------------------------------------------

def plot_cM_bahamas():
    '''
    Plot concentration mass relation in BAHAMAS
    '''
    pl.set_style()
    profiles = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles.hdf5', 'r')

    m500 = profiles['PartType1/M500'][:]
    m200 = profiles['PartType1/M200'][:]
    numpart = profiles['PartType1/NumPartGroup'][:]
    grnr = profiles['PartType1/GroupNumber'][:]
    relaxed = profiles['PartType1/Relaxed'][:]
    # r500 = profiles['PartType1/R500'][::100]
    idx = ((numpart[grnr] >= 1e4) & relaxed)

    m500 = m500[idx]
    m200 = m200[idx]

    c_x = tools.M_to_c200(m200, m500, 200, 500, p.prms.rho_m * 0.7**2)
    np.save('halo/data/BAHAMAS/c_x.npy', c_x)
    m_range = np.logspace(np.log10(m200).min(), np.log10(m200).max(), 100)
    c_cor = profs.c_correa(m_range, 0).reshape(-1)

    m_bins = np.logspace(np.log10(m200).min(), np.log10(m200).max(), 20)
    m = tools.bins2center(m_bins)
    m_bin_idx = np.digitize(m200, m_bins)
    c_med = np.array([np.median(c_x[m_bin_idx == m_bin])
                                for m_bin in np.arange(1, len(m_bins))])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    img = ax.hexbin(np.log10(m200), np.log10(c_x), cmap='magma',
                    bins='log')
    ax.plot(np.log10(m_range), np.log10(c_cor), c='w',
            label=r'$c_{\mathrm{correa}}$')
    ax.plot(np.log10(m), np.log10(c_med), c='w', label=r'$c_{\mathrm{med}}$')
    ax.set_xlabel(r'$M_{200} \, [\log_{10}M_\odot]$')
    ax.set_ylabel(r'$\log_{10} c_{200}$')
    ax.set_xlim([np.log10(m).min(), np.log10(m).max()])

    for line in ax.xaxis.get_ticklines():
        line.set_color('w')
    for line in ax.yaxis.get_ticklines():
        line.set_color('w')

    leg = ax.legend(loc='best')
    for text in leg.get_texts():
        plt.setp(text, color='w')

    cb = fig.colorbar(img)
    cb.set_label(r'$\log_{10} N_{\mathrm{bin}}$', rotation=270, labelpad=25)

    plt.show()

    profiles.close()

# ------------------------------------------------------------------------------
# End of plot_cM_bahamas()
# ------------------------------------------------------------------------------

def compare_individual_binned():
    '''
    Compare the median mass obtained via the median of the individual profile
    masses with the median for the median binned profile mass.
    '''
    profiles = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_500.hdf5', 'r')
    binned = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_binned_500.hdf5', 'r')

    r_bins = profiles['RBins_R_Mean500'][:]
    r = tools.bins2center(r_bins)
    m_bins = binned['MBins_M_Mean500'][:]
    m = tools.bins2center(m_bins).reshape(-1)

    rho_i = profiles['PartType0/Densities'][:]
    r500_i = profiles['PartType0/R500'][:] * cm2mpc
    m500_i = profiles['PartType0/M500'][:]
    mgas_i = tools.m_h(rho_i, r.reshape(1,-1) * r500_i.reshape(-1,1), axis=-1)
    m_med_i = np.array([np.median(mgas_i[(m_bin[0] <= m500_i) &
                                         (m500_i <= m_bin[1])])
                        for m_bin in m_bins])

    rho = binned['PartType0/MedianDensity'][:]
    r500_inbin = binned['PartType0/R500'][:]
    numbin = binned['PartType0/NumBin'][:]
    to_slice = np.concatenate([[0], numbin])
    bin_slice = np.concatenate([np.cumsum(to_slice[:-1]).reshape(-1,1),
                                np.cumsum(to_slice[1:]).reshape(-1,1)], axis=-1)
    r500 = np.array([np.median(r500_inbin[sl[0]:sl[1]]) for sl in bin_slice])
    m_med = tools.m_h(rho, r.reshape(1,-1) * r500.reshape(-1,1), axis=-1)

    # compare actual mass in simulation to mass extrapolated from f_gas
    f_gas = gas.f_gas(m, **gas.f_gas_fit())
    m_x = tools.m_h(rho, r.reshape(1,-1), axis=-1)
    r500_f = (((f_gas * m)/m_x)**(1./3))
    mgas_f = tools.m_h(rho, r.reshape(1,-1) * r500_f.reshape(-1,1), axis=-1)

    pl.set_style('mark')
    plt.plot(m, m_med_i, label=r'Individual')
    plt.plot(m, m_med, label=r'Binned')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$M_{500} \, [M_\odot]$')
    plt.ylabel(r'$M_{\mathrm{gas},500} \, [M_\odot]$')
    plt.legend(loc='best')
    plt.show()

    pl.set_style('mark')
    plt.plot(m, m_med_i/m, label=r'Individual')
    plt.plot(m, m_med/m, label=r'Binned')
    plt.plot(m, f_gas, label=r'Observations')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$M_{500} \, [M_\odot]$')
    plt.ylabel(r'$f_{\mathrm{gas},500} \, [M_\odot]$')
    plt.legend(loc='best')
    plt.show()

    pl.set_style('mark')
    plt.plot(m, mgas_f, label=r'$M_{\mathrm{gas,obs}}$')
    plt.plot(m, m_med, label=r'$M_{\mathrm{gas,sim}}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$M_{500}\,[M_\odot]$')
    plt.ylabel(r'$M_{\mathrm{gas}}\,[M_\odot]$')
    plt.legend(loc='best')
    plt.show()

    profiles.close()
    binned.close()

# ------------------------------------------------------------------------------
# End of compare_individual_binned()
# ------------------------------------------------------------------------------

def fit_beta_bahamas():
    '''
    Fit beta profiles to the bahamas bins
    '''
    # bahamas data
    binned = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_binned_500_crit_Tgt1e65_5r500c.hdf5', 'r')

    rho = binned['PartType0/MedianDensity'][:]
    q16 = binned['PartType0/Q16'][:]
    q84 = binned['PartType0/Q84'][:]
    r500_inbin = binned['PartType0/R500'][:]
    numbin = binned['PartType0/NumBin'][:]

    to_slice = np.concatenate([[0], numbin])
    bin_slice = np.concatenate([np.cumsum(to_slice[:-1]).reshape(-1,1),
                                np.cumsum(to_slice[1:]).reshape(-1,1)], axis=-1)

    r500 = np.array([np.median(r500_inbin[sl[0]:sl[1]]) for sl in bin_slice])
    err = np.maximum(rho - q16, q84 - rho)

    r_bins = binned['RBins_R_Crit500'][:]
    r = tools.bins2center(r_bins)
    # idx_500 = np.argmin(np.abs(r - 1))
    idx_500 = r.shape[0] - 1
    m_bins = binned['MBins_M_Crit500'][:]

    norm = tools.m_h(rho[:,:idx_500 + 1], r[:idx_500 + 1].reshape(1,-1), axis=-1)
    rho_norm = rho / norm.reshape(-1,1)
    q16_norm = q16 / norm.reshape(-1,1)
    q84_norm = q84 / norm.reshape(-1,1)
    err_norm = err / norm.reshape(-1,1)

    binned.close()

    fit_prms_p = []
    covs_p     = []
    profiles_p = []
    fit_prms_b = []
    covs_b     = []
    profiles_b = []
    for idx, profile in enumerate(rho_norm):
        sl = ((r >= 0.05) & (r <= r[idx_500]) & (profile > 0))
        fit_p, cov_p, prof_p = profs.fit_profile_beta_plaw(r[sl], 1,
                                                           r[idx_500],
                                                           profile[sl],
                                                           err=err_norm[idx][sl])
        fit_b, cov_b, prof_b = profs.fit_profile_beta(r[sl], 1,
                                                      r[idx_500],
                                                      profile[sl],
                                                      err=err_norm[idx][sl])
        fit_prms_p.append(fit_p)
        covs_p.append(cov_p)
        profiles_p.append(prof_p)
        fit_prms_b.append(fit_b)
        covs_b.append(cov_b)
        profiles_b.append(prof_b)

        # print fit_b['r_c']
        # plt.plot(r, profile)
        # plt.plot(r[sl], prof_b)
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()

    return fit_prms_p, fit_prms_b, covs_p, covs_b, profiles_p, profiles_b

# ------------------------------------------------------------------------------
# End of fit_beta_bahamas()
# ------------------------------------------------------------------------------

def plot_beta_bahamas():
    '''
    Plot beta profile fits and profiles for mass bins
    '''
    fit_p, fit_b, covs_p, covs_b, profiles_p, profiles_b = fit_beta_bahamas()
    binned = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_binned_500_crit_Tgt1e65_5r500c.hdf5', 'r')

    rho = binned['PartType0/MedianDensity'][:]
    q16 = binned['PartType0/Q16'][:]
    q84 = binned['PartType0/Q84'][:]
    r500_inbin = binned['PartType0/R500'][:]
    numbin = binned['PartType0/NumBin'][:]
    to_slice = np.concatenate([[0], numbin])
    bin_slice = np.concatenate([np.cumsum(to_slice[:-1]).reshape(-1,1),
                                np.cumsum(to_slice[1:]).reshape(-1,1)], axis=-1)
    r500 = np.array([np.median(r500_inbin[sl[0]:sl[1]]) for sl in bin_slice])
    err = np.maximum(rho - q16, q84 - rho)

    r_bins = binned['RBins_R_Crit500'][:]
    r = tools.bins2center(r_bins)
    idx_500 = np.argmin(np.abs(r - 1))
    m_bins = binned['MBins_M_Crit500'][:]
    m = tools.bins2center(m_bins)

    norm = tools.m_h(rho[:,:idx_500 + 1], r[:idx_500 + 1].reshape(1,-1), axis=-1)
    rho_norm = rho / norm.reshape(-1,1)
    q16_norm = q16 / norm.reshape(-1,1)
    q84_norm = q84 / norm.reshape(-1,1)
    err_norm = err / norm.reshape(-1,1)

    binned.close()
    pl.set_style()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_prop_cycle(pl.cycle_line())
    lines = []
    # for idx, prms in enumerate(fit_p):
    #     prof = profs.profile_b_plaw(r[r>=0.05], 1., r[idx_500], **prms)
    #     prof = profs.profile_beta_extra(r[r>=0.05], prof, r[idx_500], 3.)
    #     line, = ax.plot(r[r>=0.05], prof * m[idx])
    #     lines.append(line)

    for idx, prms in enumerate(fit_b):
        prof = profs.profile_b(r[r>=0.05], 1., r[idx_500], **prms)
        # prof = profs.profile_beta_extra(r[r>=0.05], prof, r[idx_500], 3.)
        line, = ax.plot(r[r>=0.05], prof * m[idx])
        lines.append(line)

    ax.set_prop_cycle(pl.cycle_mark())
    marks = []
    for idx, prof in enumerate(rho_norm):
        mark, = ax.plot(r[r>=0.05], prof[r>=0.05] * m[idx])
        marks.append(mark)

    ax.set_xlim([0.9*r[r>=0.05].min(), 1.1*r[r>=0.05].max()])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$r/r_{500}$')
    ax.set_ylabel(r'$M u(r)$')

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend([(line, mark) for line, mark in zip(lines, marks)],
              [r'$M=10^{%.1f}$'%np.log10(c) for c in m],
              loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()

# ------------------------------------------------------------------------------
# End of plot_beta_bahamas()
# ------------------------------------------------------------------------------

def plot_gas_bahamas():
    '''
    Plot profiles for mass bins
    '''
    # binned = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_binned_500_5r500c.hdf5', 'r')

    # rho = binned['PartType0/MedianDensity'][:]
    # q16 = binned['PartType0/Q16'][:]
    # q84 = binned['PartType0/Q84'][:]
    # r500_inbin = binned['PartType0/R500'][:]
    # numbin = binned['PartType0/NumBin'][:]
    # to_slice = np.concatenate([[0], numbin])
    # bin_slice = np.concatenate([np.cumsum(to_slice[:-1]).reshape(-1,1),
    #                             np.cumsum(to_slice[1:]).reshape(-1,1)], axis=-1)
    # r500 = np.array([np.median(r500_inbin[sl[0]:sl[1]]) for sl in bin_slice])
    # err = np.maximum(rho - q16, q84 - rho)

    # r_bins = binned['RBins_R_Crit500'][:]
    # r = tools.bins2center(r_bins)
    # idx_500 = np.argmin(np.abs(r - 1))
    # m_bins = binned['MBins_M_Crit500'][:]
    # m = tools.bins2center(m_bins)

    # norm = tools.m_h(rho, r.reshape(1,-1), axis=-1)
    # rho_norm = rho / norm.reshape(-1,1)
    # q16_norm = q16 / norm.reshape(-1,1)
    # q84_norm = q84 / norm.reshape(-1,1)
    # err_norm = err / norm.reshape(-1,1)

    # binned.close()

    # pl.set_style()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    # ax.set_prop_cycle(pl.cycle_line())

    # lines = []
    # for idx, prof in enumerate(rho_1[::2]):
    #     line, = ax.plot(r[r>=0.05], prof[r>=0.05] * m[::2][idx])
    #     fill = ax.fill_between(r[r>=0.05],
    #                            q16_norm[::2][idx][r>=0.05] * m[::2][idx],
    #                            q84_norm[::2][idx][r>=0.05] * m[::2][idx],
    #                            facecolor=line.get_color(), alpha=0.3,
    #                            edgecolor='none')
    #     lines.append((line, fill))

    # ax.set_xlim([0.9*r[r>=0.05].min(), 1.1*r[r>=0.05].max()])
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_xlabel(r'$r/r_{500}$')
    # ax.set_ylabel(r'$M u(r)$')

    # # Shrink current axis by 20%
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # # Put a legend to the right of the current axis
    # ax.legend([(line) for line in lines],
    #           [r'$M=10^{%.1f}$'%np.log10(c) for c in m[::2]],
    #           loc='center left', bbox_to_anchor=(1, 0.5))

    # plt.show()

    T1 = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_binned_500_crit_Tlt1e45_5r500c_M11_13.hdf5', 'r')
    T2 = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_binned_500_crit_T1e45_1e65_5r500c_M11_13.hdf5', 'r')
    T3 = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_binned_500_crit_Tgt1e65_5r500c_M11_13.hdf5', 'r')

    # get density profiles
    rho_1 = T1['PartType0/MedianDensity'][:]
    rho_2 = T2['PartType0/MedianDensity'][:]
    rho_3 = T3['PartType0/MedianDensity'][:]

    # m200 = T1['PartType0/M200'][:]
    # r200 = T1['PartType0/R200'][:] * cm2mpc

    r_bins = T1['RBins_R_Crit500'][:]
    r = tools.bins2center(r_bins)
    m_bins = T1['MBins_M_Crit500'][:]
    m = tools.bins2center(m_bins)
    print m_bins

    T1.close()
    T2.close()
    T3.close()

    # set color cycle
    cycle_color = palettable.colorbrewer.qualitative.Set1_3.mpl_colors
    cycle_ls = ['--', '-.', ':']
    cycle_lw = [1,2,3]
    cycle_tot = (cycler('ls', cycle_ls) *
                 (cycler('color', cycle_color) + cycler('lw', cycle_lw)))

    pl.set_style('line')
    # fig = plt.figure(figsize=(18,6))
    # ax_m1 = fig.add_subplot(131)
    # ax_m2 = fig.add_subplot(132)
    # ax_m3 = fig.add_subplot(133)
    # ax_m1.set_prop_cycle(cycle_tot)
    # ax_m2.set_prop_cycle(cycle_tot)
    # ax_m3.set_prop_cycle(cycle_tot)

    masses = np.array([r'$10^{%.2f}<\mathrm{m_{200}/M_\odot}<10^{%.2f}$'%(np.log10(i), np.log10(j)) for i, j in m_bins])

    # axes = [ax_m1, ax_m2, ax_m3]
    for idx, mass in enumerate(masses):
        prof_1 = rho_1[idx]
        prof_2 = rho_2[idx]
        prof_3 = rho_3[idx]
        # prof_t = rho_t[idx]

        tot = prof_1 + prof_2 + prof_3
        tot[tot == 0] = np.nan
        prof_1[prof_1 == 0] = np.nan
        prof_2[prof_2 == 0] = np.nan
        prof_3[prof_3 == 0] = np.nan
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # prof_t[prof_t == 0] = np.nan
        # l0, = ax.plot(r, tot * r**2)
        # l1, = ax.plot(r, prof_1 * r**2)
        # l2, = ax.plot(r, prof_2 * r**2)
        # l3, = ax.plot(r, prof_3 * r**2)
        l0, = ax.plot(r, tot, c='k', lw=4, ls='-', label='Total')
        l1, = ax.plot(r, prof_1)
        l2, = ax.plot(r, prof_2)
        l3, = ax.plot(r, prof_3)
        ax.set_xlim([0.9 * r.min(), 1.1 * r.max()])
        # ax.set_ylim([1e9, 1e13])
        ax.set_ylim([1e9, 1e16])

        lines = [l0, l3, l2, l1]

        ax.set_title(masses[idx])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$r/r_{500}$')
        # ax.set_ylabel(r'$\rho(r)(r/r_{500})^2$ in $M_\odot/$Mpc$^{3}$')
        ax.set_ylabel(r'$\rho(r)$ in $M_\odot/$Mpc$^{3}$')

        temps = np.array([r'Total', r'$T/\mathrm{K}>10^{6.5}$',
                          r'$10^{4.5} < T/\mathrm{K}<10^{6.5}$',
                          r'$T/\mathrm{K}<10^{4.5}$'])


        ax.legend(lines, temps, loc='best')

        plt.show()

# ------------------------------------------------------------------------------
# End of plot_gas_bahamas()
# ------------------------------------------------------------------------------

def plot_dm_bahamas():
    '''
    Plot profiles for mass bins
    '''
    binned = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_binned.hdf5', 'r')
    dmonly = h5py.File('halo/data/BAHAMAS/DMONLY/eagle_subfind_particles_032_profiles_binned.hdf5', 'r')

    rho = binned['PartType1/MedianDensity'][:]
    q16 = binned['PartType1/Q16'][:]
    q84 = binned['PartType1/Q84'][:]
    r200 = binned['PartType1/MedianR200'][:]
    # m200 = binned['PartType1/MedianM200'][:]

    dmo_rho = dmonly['PartType1/MedianDensity'][:]
    dmo_q16 = dmonly['PartType1/Q16'][:]
    dmo_q84 = dmonly['PartType1/Q84'][:]
    dmo_r200 = dmonly['PartType1/MedianR200'][:]
    # dmo_m200 = dmonly['PartType1/MedianM200'][:]

    # m_agn > m_dmo

    r_bins = binned['RBins_R_Mean200'][:]
    r = tools.bins2center(r_bins)
    m_bins = binned['MBins_M_Mean200'][:]
    m = tools.bins2center(m_bins)

    norm = tools.m_h(rho, r200.reshape(-1,1) * r.reshape(1,-1), axis=-1)
    rho_norm = rho / norm.reshape(-1,1)
    q16_norm = q16 / norm.reshape(-1,1)
    q84_norm = q84 / norm.reshape(-1,1)

    dmo_norm = tools.m_h(dmo_rho, dmo_r200.reshape(-1,1) * r.reshape(1,-1), axis=-1)
    dmo_rho_norm = dmo_rho / dmo_norm.reshape(-1,1)
    dmo_q16_norm = dmo_q16 / dmo_norm.reshape(-1,1)
    dmo_q84_norm = dmo_q84 / dmo_norm.reshape(-1,1)

    binned.close()
    dmonly.close()

    pl.set_style()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_prop_cycle(pl.cycle_line())

    lines = []
    lines_dmo = []
    for idx, prof in enumerate(rho_norm[::4]):
        line1, = ax.plot(r[r>=0.05], dmo_rho_norm[::4][idx][r>=0.05] * m[::4][idx])
        fill1 = ax.fill_between(r[r>=0.05],
                               dmo_q16_norm[::4][idx][r>=0.05] * m[::4][idx],
                               dmo_q84_norm[::4][idx][r>=0.05] * m[::4][idx],
                               facecolor=line1.get_color(), alpha=0.3,
                               edgecolor='none')
        line, = ax.plot(r[r>=0.05], prof[r>=0.05] * m[::4][idx])
        fill = ax.fill_between(r[r>=0.05],
                               q16_norm[::4][idx][r>=0.05] * m[::4][idx],
                               q84_norm[::4][idx][r>=0.05] * m[::4][idx],
                               facecolor=line1.get_color(), alpha=0.3,
                               edgecolor='none')
        lines.append((line, fill))
        lines_dmo.append((line1, fill1))

    ax.set_xlim([0.9*r[r>=0.05].min(), 1.1*r[r>=0.05].max()])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$r/r_{200}$')
    ax.set_ylabel(r'$M u(r)$')

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    handles = np.array(zip(lines_dmo, lines), dtype=object).reshape(-1,2)
    # handles needs to be list of tuples
    handles = map(tuple, handles)
    labels = np.array([[r'$M_{\mathrm{DMO}}=10^{%.1f}$'%np.log10(c),
                        r'$M_{\mathrm{AGN}}=10^{%.1f}$'%np.log10(c)]
                       for c in m[::4]]).reshape(-1)
    ax.legend([handle for handle in handles],
              [lab for lab in labels],
              loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()

# ------------------------------------------------------------------------------
# End of plot_dm_bahamas()
# ------------------------------------------------------------------------------

def plot_dm_ratio_bahamas():
    '''
    Plot profiles for mass bins
    '''
    binned = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_binned.hdf5', 'r')
    dmonly = h5py.File('halo/data/BAHAMAS/DMONLY/eagle_subfind_particles_032_profiles_binned.hdf5', 'r')

    rho = binned['PartType1/MedianDensity'][:]
    q16 = binned['PartType1/Q16'][:]
    q84 = binned['PartType1/Q84'][:]
    r200 = binned['PartType1/MedianR200'][:]

    dmo_rho = dmonly['PartType1/MedianDensity'][:]
    dmo_q16 = dmonly['PartType1/Q16'][:]
    dmo_q84 = dmonly['PartType1/Q84'][:]
    dmo_r200 = dmonly['PartType1/MedianR200'][:]

    r_bins = binned['RBins_R_Mean200'][:]
    r = tools.bins2center(r_bins)
    m_bins = binned['MBins_M_Mean200'][:]
    m = tools.bins2center(m_bins)

    norm = tools.m_h(rho, r200.reshape(-1,1) * r.reshape(1,-1), axis=-1)
    rho_norm = rho / norm.reshape(-1,1)
    q16_norm = q16 / norm.reshape(-1,1)
    q84_norm = q84 / norm.reshape(-1,1)

    dmo_norm = tools.m_h(dmo_rho, dmo_r200.reshape(-1,1) * r.reshape(1,-1), axis=-1)
    dmo_rho_norm = dmo_rho / dmo_norm.reshape(-1,1)
    dmo_q16_norm = dmo_q16 / dmo_norm.reshape(-1,1)
    dmo_q84_norm = dmo_q84 / dmo_norm.reshape(-1,1)

    binned.close()
    dmonly.close()

    pl.set_style()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_prop_cycle(pl.cycle_line())

    lines = []
    for idx, prof in enumerate(rho_norm[::4]):
        dmo_prof = dmo_rho_norm[::4][idx]
        line, = ax.plot(r * r200[::4][idx], prof/dmo_prof)
        lines.append((line,))

    ax.axhline(y=1, c='k', ls='--')
    ax.set_xlim([(0.9*r*r200[0]).min(), (1.1*r*r200[::4][idx]).max()])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$r \, [\mathrm{Mpc}/h]$')
    ax.set_ylabel(r'$u_{\mathrm{AGN}}(r) / u_{\mathrm{DMO}}(r)$')

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    labels = np.array([r'$M=10^{%.1f}$'%np.log10(c)
                       for c in m[::4]]).reshape(-1)
    ax.legend([handle for handle in lines],
              [lab for lab in labels],
              loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()

# ------------------------------------------------------------------------------
# End of plot_dm_ratio_bahamas()
# ------------------------------------------------------------------------------

def plot_stars_bahamas():
    '''
    Plot profiles for mass bins
    '''
    binned = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_binned.hdf5', 'r')

    rho = binned['PartType4/MedianDensity'][:]
    q16 = binned['PartType4/Q16'][:]
    q84 = binned['PartType4/Q84'][:]

    r_bins = binned['RBins_R_Mean200'][:]
    r = tools.bins2center(r_bins)
    m_bins = binned['MBins_M_Mean200'][:]
    m = tools.bins2center(m_bins)

    norm = tools.m_h(rho, r.reshape(1,-1), axis=-1)
    rho_norm = rho / norm.reshape(-1,1)
    q16_norm = q16 / norm.reshape(-1,1)
    q84_norm = q84 / norm.reshape(-1,1)

    binned.close()

    pl.set_style()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_prop_cycle(pl.cycle_line())

    lines = []
    for idx, prof in enumerate(rho_norm[::2]):
        # print prof[r>=0.05] * m[::2][idx]
        line, = ax.plot(r, prof * m[::2][idx])
        fill = ax.fill_between(r,
                               q16_norm[::2][idx] * m[::2][idx],
                               q84_norm[::2][idx] * m[::2][idx],
                               facecolor=line.get_color(), alpha=0.1,
                               edgecolor='none')
        lines.append((line,fill))

    ax.set_xlim([0.9*r.min(), 1.1*r.max()])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$r/r_{200}$')
    ax.set_ylabel(r'$M u(r)$')

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    labels = np.array([r'$M=10^{%.1f}$'%np.log10(c)
                       for c in m[::2]]).reshape(-1)
    ax.legend([handle for handle in lines],
              [lab for lab in labels],
              loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()

# ------------------------------------------------------------------------------
# End of plot_stars_bahamas()
# ------------------------------------------------------------------------------

def compare_bahamas_eckert(r_bins=np.logspace(-2.5, 0.1, 20),
                           m_bins=np.logspace(13, 15, 20)):
    '''
    Plot median profiles for BAHAMAS and mean profiles from Eckert
    '''
    # get gas profiles
    r_eckert, rho_eckert, s, mwl, mgas = gas.rhogas_eckert()
    m_eckert = mwl

    m_bins_e, m_bins = m_bins_eckert()

    # bahamas data
    binned = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_binned_500_crit_eckert.hdf5', 'r')

    r_bins = binned['RBins_R_Crit500'][:]
    r = tools.bins2center(r_bins)

    m500 = binned['PartType0/M500'][:]
    numbin = binned['PartType0/NumBin'][:]

    numbin = np.concatenate([[0], numbin])
    bin_slice = np.concatenate([np.cumsum(numbin[:-1]).reshape(-1,1),
                                np.cumsum(numbin[1:]).reshape(-1,1)], axis=-1)

    m = np.array([np.mean(m500[sl[0]:sl[1]]) for sl in bin_slice])

    rho_bah = binned['PartType0/MedianDensity'][:]

    binned.close()
    pl.set_style()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    marks = []
    lines = []
    ax.set_prop_cycle(pl.cycle_mark())
    for idx, prof in enumerate(rho_eckert):
        mark = ax.errorbar(r_eckert, prof, yerr=s[idx], fmt='o')
        marks.append(mark)

    ax.set_prop_cycle(pl.cycle_line())
    for prof in rho_bah:
        line, = ax.plot(r, prof)
        lines.append(line)

    ax.set_xlim([0.9*r_eckert.min(), 1.1*r_eckert.max()])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$r/r_{500}$')
    ax.set_ylabel(r'$\rho(r)$ in $M_\odot/$Mpc$^3$')
    ax.legend([(line, mark) for line, mark in zip(lines, marks)],
              [r'$M_{\mathrm{mean},b}=10^{%.2f}M_\odot, \, M_{\mathrm{mean},e}=10^{%.2f}M_\odot$'%(np.log10(mass), np.log10(m_bins[idx]))
               for idx, mass in enumerate(m)])
    plt.show()

# ------------------------------------------------------------------------------
# End of compare_bahamas_eckert()
# ------------------------------------------------------------------------------

def compare_bahamas_sun_croston():
    '''
    Compare BAHAMAS profiles binned to reproduce median mass in Sun & Croston
    samples.
    '''
    r500_sun, rho_sun, err_sun = gas.rhogas_sun()
    r500_sun = r500_sun.reshape(-1,4).mean(axis=1)
    rho_sun = rho_sun.reshape(-1,4).mean(axis=1)
    err_sun = 1./2 * np.sqrt(np.sum(err_sun.reshape(2,-1,4)**2, axis=-1))

    r500_croston, rho_croston, err_croston = gas.rhogas_croston()

    binned = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_binned_500_crit_T1e6_papercut.hdf5', 'r')

    r_bins = binned['RBins_R_Crit500'][:]
    r = tools.bins2center(r_bins)

    m500 = binned['PartType0/M500'][:]
    numbin = binned['PartType0/NumBin'][:]
    m = np.array([np.median(m500[:numbin[0]]),
                  np.median(m500[numbin[0]:])])

    rho_bah = binned['PartType0/MedianDensity'][:]

    binned.close()
    pl.set_style()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_prop_cycle(pl.cycle_mark())
    marks = []
    # mark = ax.errorbar(r500_sun, rho_sun * r500_sun**2,
    #                    yerr=[err_sun[0] * r500_sun**2,
    #                          err_sun[1] * r500_sun**2],
    #                    fmt='o')
    mark = ax.errorbar(r500_sun, rho_sun,
                       yerr=[err_sun[0],
                             err_sun[1]],
                       fmt='o')
    marks.append(mark)
    # mark = ax.errorbar(r500_croston, rho_croston * r500_croston**2,
    #                    yerr=[err_croston[0] * r500_croston**2,
    #                          err_croston[1] * r500_croston**2],
    #                    fmt='o')
    mark = ax.errorbar(r500_croston, rho_croston,
                       yerr=[err_croston[0],
                             err_croston[1]],
                       fmt='o')
    marks.append(mark)

    ax.set_prop_cycle(pl.cycle_line())
    lines = []
    for idx, prof in enumerate(rho_bah):
        # line, = ax.plot(r, r**2 * prof/(p.prms.rho_crit * 0.7**2))#*1.3)
        line, = ax.plot(r, prof/(p.prms.rho_crit * 0.7**2))
        lines.append(line)

    ax.set_xlim([0.9*r500_croston.min(), 1.1*r500_croston.max()])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$r/r_{500}$')
    # ax.set_ylabel(r'$\rho(r)/\rho_{\mathrm{crit}} (r/r_{500})^2$')
    ax.set_ylabel(r'$\rho(r)/\rho_{\mathrm{crit}}$')
    ax.legend([(line, mark) for line, mark in zip(lines, marks)],
              [r'$M_{\mathrm{med}}=10^{%.1f}M_\odot$'%np.log10(c) for c in m])
    plt.show()

# ------------------------------------------------------------------------------
# End of compare_bahamas_sun_croston()
# ------------------------------------------------------------------------------

def compare_temperature():
    '''
    Compare gas profiles for all gas and only hot
    '''
    Tlt1e45 = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_binned_500_crit_Tlt1e45_5r500c.hdf5', 'r')
    T1e65   = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_binned_500_crit_T1e45_1e65_5r500c.hdf5', 'r')
    Tgt1e65   = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_binned_500_crit_Tgt1e65_5r500c.hdf5', 'r')
    # Tot = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_censat_binned_500_crit_r500c_M131415.hdf5', 'r')

    # get density profiles
    rho_1 = Tlt1e45['PartType0/MedianDensity'][:]
    rho_2 = T1e65['PartType0/MedianDensity'][:]
    rho_3 = Tgt1e65['PartType0/MedianDensity'][:]
    # rho_t = Tot['PartType0/MedianDensity'][:]

    # get radial range
    r_bins_1 = Tlt1e45['RBins_R_Crit500'][:]
    r_bins_2 = T1e65['RBins_R_Crit500'][:]
    r_bins_3 = Tgt1e65['RBins_R_Crit500'][:]
    # r_bins_t = Tot['RBins_R_Crit500'][:]
    r_1 = tools.bins2center(r_bins_1)
    r_2 = tools.bins2center(r_bins_2)
    r_3 = tools.bins2center(r_bins_3)
    # r_t = tools.bins2center(r_bins_t)

    # get mass range
    m_bins_1 = Tlt1e45['MBins_M_Crit500'][:]
    m_bins_2 = T1e65['MBins_M_Crit500'][:]
    m_bins_3 = Tgt1e65['MBins_M_Crit500'][:]
    # m_bins_t = Tot['MBins_M_Crit500'][:]
    m_1 = tools.bins2center(m_bins_1)
    m_2 = tools.bins2center(m_bins_2)
    m_3 = tools.bins2center(m_bins_3)
    # m_t = tools.bins2center(m_bins_t)

    Tlt1e45.close()
    T1e65.close()
    Tgt1e65.close()
    # Tot.close()

    # set color cycle
    cycle_color = palettable.colorbrewer.qualitative.Set1_3.mpl_colors
    cycle_ls = ['--', '-.', ':']
    cycle_lw = [1,2,3]
    cycle_tot = (cycler('ls', cycle_ls) *
                 (cycler('color', cycle_color) + cycler('lw', cycle_lw)))

    pl.set_style('line')
    fig = plt.figure(figsize=(18,6))
    ax_m1 = fig.add_subplot(131)
    ax_m2 = fig.add_subplot(132)
    ax_m3 = fig.add_subplot(133)
    ax_m1.set_prop_cycle(cycle_tot)
    ax_m2.set_prop_cycle(cycle_tot)
    ax_m3.set_prop_cycle(cycle_tot)

    masses = np.array([r'$10^{%.2f}<\mathrm{m_{200}/M_\odot}<10^{%.2f}$'%(np.log10(i), np.log10(j)) for i, j in m_bins_1])

    axes = [ax_m1, ax_m2, ax_m3]
    for idx, ax in enumerate(axes):
        prof_1 = rho_1[idx]
        prof_2 = rho_2[idx]
        prof_3 = rho_3[idx]
        # prof_t = rho_t[idx]

        tot = prof_1 + prof_2 + prof_3
        tot[tot == 0] = np.nan
        prof_1[prof_1 == 0] = np.nan
        prof_2[prof_2 == 0] = np.nan
        prof_3[prof_3 == 0] = np.nan
        # prof_t[prof_t == 0] = np.nan
        # l0, = ax.plot(r_1, tot * r_1**2)
        # l1, = ax.plot(r_1, prof_1 * r_1**2)
        # l2, = ax.plot(r_2, prof_2 * r_2**2)
        # l3, = ax.plot(r_3, prof_3 * r_3**2)
        # l4, = ax.plot(r_t, prof_t * r_t**2)
        l0, = ax.plot(r_1, tot, c='k', lw=4, ls='-', label='Total')
        l1, = ax.plot(r_1, prof_1)
        l2, = ax.plot(r_2, prof_2)
        l3, = ax.plot(r_3, prof_3)
        ax.set_xlim([0.9 * r_1.min(), 1.1 * r_1.max()])
        # ax.set_ylim([1e9, 1e13])
        ax.set_ylim([1e9, 1e16])

        lines = [l0, l3, l2, l1]

        ax.set_title(masses[idx])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$r/r_{500}$')
        if idx == 0:
            # ax.set_ylabel(r'$\rho(r)(r/r_{500})^2$ in $M_\odot/$Mpc$^{3}$')
            ax.set_ylabel(r'$\rho(r)$ in $M_\odot/$Mpc$^{3}$')

    temps = np.array([r'Total', r'$T/\mathrm{K}>10^{6.5}$',
                      r'$10^{4.5} < T/\mathrm{K}<10^{6.5}$',
                      r'$T/\mathrm{K}<10^{4.5}$'])


    ax_m1.legend(lines, temps, loc='best')

    plt.show()

# ------------------------------------------------------------------------------
# End of compare_temperature()
# ------------------------------------------------------------------------------

def compare_temperature_censat(r2=True):
    '''
    Compare gas profiles for all gas and only hot
    '''
    Tlt1e45 = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_censat_binned_500_crit_Tlt1e45_5r500c.hdf5', 'r')
    T1e65   = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_censat_binned_500_crit_T1e45_1e65_5r500c.hdf5', 'r')
    Tgt1e65   = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_censat_binned_500_crit_Tgt1e65_5r500c.hdf5', 'r')
    binned = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_censat_binned_500_crit_5r500c.hdf5', 'r')

    # get density profiles
    rho_1_c = Tlt1e45['PartType0/CenMedianDensity'][:]
    rho_2_c = T1e65['PartType0/CenMedianDensity'][:]
    rho_3_c = Tgt1e65['PartType0/CenMedianDensity'][:]
    rho_1_s = Tlt1e45['PartType0/SatMedianDensity'][:]
    rho_2_s = T1e65['PartType0/SatMedianDensity'][:]
    rho_3_s = Tgt1e65['PartType0/SatMedianDensity'][:]
    # rho_t = Tot['PartType0/MedianDensity'][:]
    # rho_c = binned['PartType4/CenMedianDensity'][:]
    # rho_s = binned['PartType4/SatMedianDensity'][:]

    # want to know hot gas mass wrt halo mass
    r500 = Tgt1e65['PartType0/R500'][:]
    numbin = Tgt1e65['PartType0/NumBin'][:]
    cum_index = np.cumsum(np.concatenate([[0], numbin]))
    r500_med = np.array([np.median([r500[cum_index[idx]:cum_index[idx+1]]])
                         for idx in range(cum_index.shape[0] - 1)])

    # get radial range
    # r_bins = Tlt1e45['RBins_R_Crit500'][:]
    r_bins = Tlt1e45['RBins_R_Mean200'][:]
    r = tools.bins2center(r_bins)

    # get mass range
    m_bins = Tlt1e45['MBins_M_Crit500'][:]
    m = tools.bins2center(m_bins)

    Tlt1e45.close()
    T1e65.close()
    Tgt1e65.close()
    # binned.close()

    # set color cycle
    c_stars = palettable.colorbrewer.qualitative.Set1_4.mpl_colors[-1]
    cycle_color = palettable.colorbrewer.qualitative.Set1_3.mpl_colors
    cycle_ls = ['-', '--']
    cycle_lw = [1,2,3]
    cycle_tot = (cycler('ls', cycle_ls) *
                 (cycler('color', cycle_color) + cycler('lw', cycle_lw)))

    pl.set_style('line')
    fig = plt.figure(figsize=(18,8))
    ax_m1 = fig.add_axes([0.05, 0.5, 0.3, 0.4])
    ax_m2 = fig.add_axes([0.35, 0.5, 0.3, 0.4])
    ax_m3 = fig.add_axes([0.65, 0.5, 0.3, 0.4])
    ax_m4 = fig.add_axes([0.05, 0.1, 0.3, 0.4])
    ax_m5 = fig.add_axes([0.35, 0.1, 0.3, 0.4])
    ax_m6 = fig.add_axes([0.65, 0.1, 0.3, 0.4])
    ax_m1.set_prop_cycle(cycle_tot)
    ax_m2.set_prop_cycle(cycle_tot)
    ax_m3.set_prop_cycle(cycle_tot)
    ax_m4.set_prop_cycle(cycle_tot)
    ax_m5.set_prop_cycle(cycle_tot)
    ax_m6.set_prop_cycle(cycle_tot)

    masses = np.array([r'$10^{%.2f}<\mathrm{m_{200}/M_\odot}<10^{%.2f}$'%(np.log10(i), np.log10(j)) for i, j in m_bins])


    axes = [ax_m1, ax_m2, ax_m3, ax_m4, ax_m5, ax_m6]
    for idx, ax in enumerate(axes):
        prof_1_c = rho_1_c[idx]
        prof_2_c = rho_2_c[idx]
        prof_3_c = rho_3_c[idx]
        prof_1_s = rho_1_s[idx]
        prof_2_s = rho_2_s[idx]
        prof_3_s = rho_3_s[idx]
        # prof_c  = rho_c[idx]
        # prof_s  = rho_s[idx]

        tot_c = prof_1_c + prof_2_c + prof_3_c
        tot_s = prof_1_s + prof_2_s + prof_3_s
        tot = tot_c + tot_s
        # m_comp = tools.m_h(tot, r * r500_med[idx])
        # print 'Halo mass: %.6e'%(m_bins[idx].mean())
        # print 'Comp mass: %.6e'%(m_comp)
        # print '-----------'
        tot[tot == 0] = np.nan
        prof_1_c[prof_1_c == 0] = np.nan
        prof_2_c[prof_2_c == 0] = np.nan
        prof_3_c[prof_3_c == 0] = np.nan
        prof_1_s[prof_1_s == 0] = np.nan
        prof_2_s[prof_2_s == 0] = np.nan
        prof_3_s[prof_3_s == 0] = np.nan
        prof_c[prof_c == 0] = np.nan
        prof_s[prof_s == 0] = np.nan
        if r2:
            l0, = ax.plot(r, tot * r**2, c='k', lw=4, ls='-', label='Total')
            l1_c, = ax.plot(r, prof_1_c * r**2)
            l2_c, = ax.plot(r, prof_2_c * r**2)
            l3_c, = ax.plot(r, prof_3_c * r**2)
            l1_s, = ax.plot(r, prof_1_s * r**2)
            l2_s, = ax.plot(r, prof_2_s * r**2)
            l3_s, = ax.plot(r, prof_3_s * r**2)
            if idx in [0,1,2]:
                ratio = prof_1_c[~np.isnan(prof_1_c)] / prof_c[~np.isnan(prof_1_c)]
                print idx
                print ratio
                print np.mean(ratio)
                print np.median(ratio)
                print '-----------------------'
                l_sc, = ax.plot(r, prof_c * r**2, lw=1, marker='*', c=c_stars,
                                label='Stars')
                # l_ss, = ax.plot(r, prof_s * r**2, lw=1, ls='--', c=c_stars)
            ax.set_ylim([1e9, 1e13])
        else:
            l0, = ax.plot(r, tot, c='k', lw=4, ls='-', label='Total')
            l1_c, = ax.plot(r, prof_1_c)
            l2_c, = ax.plot(r, prof_2_c)
            l3_c, = ax.plot(r, prof_3_c)
            l1_s, = ax.plot(r, prof_1_s)
            l2_s, = ax.plot(r, prof_2_s)
            l3_s, = ax.plot(r, prof_3_s)
            if idx in [0,1,2]:
                l_sc, = ax.plot(r, prof_c, lw=1, marker='*', c=c_stars,
                                label='Stars')
                # l_ss, = ax.plot(r, prof_s, lw=1, ls='--', c=c_stars)
            ax.set_ylim([1e9, 1e16])

        ax.set_xlim([0.9 * r.min(), 1.1 * r.max()])

        # Want custom legend:
        # show linestyle for total
        # show linestyles for cen vs sat
        # show linestyles for temps
        line_cen = mlines.Line2D([], [], ls='-', color='k', label='Central')
        line_sat = mlines.Line2D([], [], ls='--', color='k', label='Satellite')
        line_1 = mlines.Line2D([], [])
        line_1.update_from(l1_c)
        line_1.set_linestyle('-')
        line_1.set_label(r'$T/\mathrm{K}<10^{4.5}$')
        line_2 = mlines.Line2D([], [])
        line_2.update_from(l2_c)
        line_2.set_linestyle('-')
        line_2.set_label(r'$10^{4.5} < T/\mathrm{K}<10^{6.5}$')
        line_3 = mlines.Line2D([], [])
        line_3.update_from(l3_c)
        line_3.set_linestyle('-')
        line_3.set_label(r'$T/\mathrm{K}>10^{6.5}$')

        lines = [l0, line_cen, line_sat, line_1, line_2, line_3, l_sc]

        # need to set visibility to False BEFORE log scaling
        if idx == 0:
            ticks = ax.get_yticklabels()
            # strange way of getting correct label
            ticks[-4].set_visible(False)

        if idx < 3:
            titletxt = ax.set_title(masses[idx])
            title_props = titletxt.get_fontproperties()
        else:
            ax.text(0.5, 0.89, masses[idx], ha='center', transform=ax.transAxes,
                    fontproperties=title_props)

        ax.set_xscale('log')
        ax.set_yscale('log')
        # remove unnecessary labels
        if not idx in [0, 3]:
            ax.yaxis.set_ticklabels([])
        if idx < 3:
            ax.xaxis.set_ticklabels([])
        if idx >= 3:
            ax.xaxis.set_ticks_position('bottom')
            text = ax.set_xlabel(r'$r/r_{500}$')
            font_properties = text.get_fontproperties()

    if r2:
        fig.text(0.005, 0.5,
                 r'$\rho(r)(r/r_{500})^2 \, [\mathrm{M_\odot}/\mathrm{Mpc}^{3}]$',
                 va='center', rotation='vertical', fontproperties=font_properties)
    else:
        fig.text(0.005, 0.5,
                 r'$\rho(r) \, [\mathrm{M_\odot}/\mathrm{Mpc}^{3}]$',
                 va='center', rotation='vertical', fontproperties=font_properties)

    ax_m1.legend(handles=lines, loc='best')

    plt.show()

# ------------------------------------------------------------------------------
# End of compare_temperature_censat()
# ------------------------------------------------------------------------------

def compare_censat(ptype, r2=True):
    '''
    Compare central vs satellite profiles
    '''
    binned = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_censat_binned_500_crit_5r500c.hdf5', 'r')

    # get density profiles
    rho_c = binned['PartType%i/CenMedianDensity'%ptype][:]
    rho_s = binned['PartType%i/SatMedianDensity'%ptype][:]

    # get radial range
    r_bins = binned['RBins_R_Crit500'][:]
    r = tools.bins2center(r_bins)

    # get mass range
    m_bins = binned['MBins_M_Crit500'][:]
    m = tools.bins2center(m_bins)

    binned.close()

    # set color cycle
    cycle_color = palettable.colorbrewer.qualitative.Set1_3.mpl_colors
    cycle_ls = ['-', '-', '--']
    cycle_tot = (cycler('color', cycle_color) + cycler('ls', cycle_ls))

    pl.set_style('line')
    fig = plt.figure(figsize=(18,6))
    ax_m1 = fig.add_axes([0.05, 0.5, 0.3, 0.4])
    ax_m2 = fig.add_axes([0.35, 0.5, 0.3, 0.4])
    ax_m3 = fig.add_axes([0.65, 0.5, 0.3, 0.4])
    ax_m4 = fig.add_axes([0.05, 0.1, 0.3, 0.4])
    ax_m5 = fig.add_axes([0.35, 0.1, 0.3, 0.4])
    ax_m6 = fig.add_axes([0.65, 0.1, 0.3, 0.4])
    ax_m1.set_prop_cycle(cycle_tot)
    ax_m2.set_prop_cycle(cycle_tot)
    ax_m3.set_prop_cycle(cycle_tot)
    ax_m4.set_prop_cycle(cycle_tot)
    ax_m5.set_prop_cycle(cycle_tot)
    ax_m6.set_prop_cycle(cycle_tot)

    masses = np.array([r'$10^{%.2f}<\mathrm{m_{200}/M_\odot}<10^{%.2f}$'%(np.log10(i), np.log10(j)) for i, j in m_bins])

    axes = [ax_m1, ax_m2, ax_m3, ax_m4, ax_m5, ax_m6]
    for idx, ax in enumerate(axes):
        prof_1 = rho_c[idx]
        prof_2 = rho_s[idx]
        tot = prof_1 + prof_2
        tot[tot == 0] = np.nan
        prof_1[prof_1 == 0] = np.nan
        prof_2[prof_2 == 0] = np.nan
        if r2:
            l0, = ax.plot(r, tot * r**2, lw=3, c='k')
            l1, = ax.plot(r, prof_1 * r**2, lw=2)
            l2, = ax.plot(r, prof_2 * r**2, lw=2)
            if idx < 3:
                ax.set_ylim([1e9, 1e14])
            else:
                ax.set_ylim([1e9, 1e13])
        else:
            l0, = ax.plot(r, tot, lw=3, c='k')
            l1, = ax.plot(r, prof_1, lw=2)
            l2, = ax.plot(r, prof_2, lw=2)
            ax.set_ylim([1e9, 1e17])

        ax.set_xlim([0.9 * r.min(), 1.1 * r.max()])

        lines = [l0, l1, l2]

        # need to set visibility to False BEFORE log scaling
        if idx == 0:
            ticks = ax.get_yticklabels()
            # strange way of getting correct label
            ticks[-4].set_visible(False)

        if idx < 3:
            titletxt = ax.set_title(masses[idx])
            title_props = titletxt.get_fontproperties()
        else:
            ax.text(0.5, 0.89, masses[idx], ha='center', transform=ax.transAxes,
                    fontproperties=title_props)

        ax.set_xscale('log')
        ax.set_yscale('log')

        # remove unnecessary labels
        if not idx in [0, 3]:
            ax.yaxis.set_ticklabels([])
        if idx < 3:
            ax.xaxis.set_ticklabels([])
        if idx >= 3:
            ax.xaxis.set_ticks_position('bottom')
            text = ax.set_xlabel(r'$r/r_{500}$')
            font_properties = text.get_fontproperties()

    if r2:
        fig.text(0.005, 0.5,
                 r'$\rho(r)(r/r_{500})^2 \, [\mathrm{M_\odot}/\mathrm{Mpc}^{3}]$',
                 va='center', rotation='vertical', fontproperties=font_properties)
    else:
        fig.text(0.005, 0.5,
                 r'$\rho(r)\, [\mathrm{M_\odot}/\mathrm{Mpc}^{3}]$',
                 va='center', rotation='vertical', fontproperties=font_properties)

    labs = np.array(['Total', 'Central', 'Satellites'])

    ax_m1.legend(lines, labs, loc='best')

    plt.show()

# ------------------------------------------------------------------------------
# End of compare_censat()
# ------------------------------------------------------------------------------

# def fit_beta_parameters():
#     '''
#     Fit the mass dependence of the different fit parameters
#     '''
#     fit_p, fit_b, covs_p, covs_b, profs_p, profs_b = fit_beta_bahamas()
#     fit_p_sun, fit_b_sun = gas.fit_sun_profile()
#     fit_p_croston, fit_b_croston = gas.fit_croston_profile()
#     M_sun = gas.rhogas_sun.M
#     M_croston = gas.rhogas_croston.M

#     m_bins = np.logspace(13, 15, 20)
#     m = tools.bins2center(m_bins)

#     beta = np.array([i['beta'] for i in fit_b])
#     r_c = np.array([i['r_c'] for i in fit_b])

#     beta_sun = fit_b_sun[0]['beta']
#     r_c_sun = fit_b_sun[0]['r_c']
#     beta_croston = fit_b_croston[0]['beta']
#     r_c_croston = fit_b_croston[0]['r_c']

#     idx_sun = np.searchsorted(m, M_sun)
#     idx_croston = np.searchsorted(m, M_croston)
#     m_new = np.insert(m, [idx_sun, idx_croston], [M_sun, M_croston])
#     beta_new = np.insert(m, [idx_sun, idx_croston], [beta_sun, beta_croston])
#     r_c_new = np.insert(m, [idx_sun, idx_croston], [r_c_sun, r_c_croston])

#     # force fit through measurements
#     sigma = np.ones_like(m)
#     sigma = np.insert(sigma, [idx_sun, idx_croston], [1e-100, 1e-100])

#     # define function for beta fit
#     def beta_fit(m_range, m_c, alpha):
#         return 1 + 2. / (1 + (m_c/m_range)**alpha)

#     beta_prms, cov = opt.curve_fit(beta_fit, m_new, beta_new,
#                                    bounds=([1e13, 0], [1e15, 5]),
#                                    sigma=sigma)
#     beta_prms = {'m_c': beta_prms[0],
#                  'alpha': beta_prms[1]}

#     # define function for r_c fit
#     def rc_fit(m_range, m_c, alpha):
#         return 0.4 / (1 + (m_c/m_range)**alpha)

#     rc_prms, cov = opt.curve_fit(rc_fit, m_new, r_c_new,
#                                  bounds=([1e13, 0], [1e15, 5]),
#                                  sigma=sigma)
#     rc_prms = {'m_c': rc_prms[0],
#                'alpha': rc_prms[1]}

#     pl.set_style('mark')
#     fig = plt.figure()
#     ax = fig.add_subplot(111)

#     ax.plot(m, beta)
#     ax.plot(M_sun, fit_b_sun[0]['beta'])
#     ax.plot(M_croston, fit_b_croston[0]['beta'])
#     ax.set_prop_cycle(pl.cycle_line())
#     ax.plot(m_new, beta_fit(m_new, **beta_prms))
#     ax.set_xscale('log')
#     ax.set_xlabel(r'$m_{500c} \, [M_\odot]$')
#     ax.set_ylabel(r'$\beta$')
#     plt.show()

#     pl.set_style('mark')
#     fig = plt.figure()
#     ax = fig.add_subplot(111)

#     ax.plot(m, r_c)
#     ax.plot(M_sun, fit_b_sun[0]['r_c'])
#     ax.plot(M_croston, fit_b_croston[0]['r_c'])
#     ax.set_prop_cycle(pl.cycle_line())
#     ax.plot(m_new, rc_fit(m_new, **rc_prms))
#     ax.set_xscale('log')
#     ax.set_xlabel(r'$m_{500c} \, [M_\odot]$')
#     ax.set_ylabel(r'$r_c$')
#     plt.show()

#     return beta_prms, rc_prms

# # ------------------------------------------------------------------------------
# # End of fit_beta_parameters()
# # ------------------------------------------------------------------------------

# def fit_plaw_parameters():
#     '''
#     Fit the mass dependence of the different fit parameters
#     '''
#     fit_p, fit_b, covs_p, covs_b, profs_p, profs_b = fit_beta_bahamas()
#     fit_p_sun, fit_b_sun = gas.fit_sun_profile()
#     fit_p_croston, fit_b_croston = gas.fit_croston_profile()
#     M_sun = gas.rhogas_sun.M
#     M_croston = gas.rhogas_croston.M

#     m_bins = np.logspace(13, 15, 20)
#     m = tools.bins2center(m_bins)

#     beta = np.array([i['beta'] for i in fit_p])
#     gamma = np.array([i['gamma'] for i in fit_p])
#     r_c = np.array([i['r_c'] for i in fit_p])

#     beta_sun = fit_p_sun[0]['beta']
#     gamma_sun = fit_p_sun[0]['gamma']
#     r_c_sun = fit_p_sun[0]['r_c']
#     beta_croston = fit_p_croston[0]['beta']
#     gamma_croston = fit_p_croston[0]['gamma']
#     r_c_croston = fit_p_croston[0]['r_c']

#     idx_sun = np.searchsorted(m, M_sun)
#     idx_croston = np.searchsorted(m, M_croston)
#     m_new = np.insert(m, [idx_sun, idx_croston], [M_sun, M_croston])
#     beta_new = np.insert(m, [idx_sun, idx_croston], [beta_sun, beta_croston])
#     gamma_new = np.insert(m, [idx_sun, idx_croston], [gamma_sun, gamma_croston])
#     r_c_new = np.insert(m, [idx_sun, idx_croston], [r_c_sun, r_c_croston])

#     # force fit through measurements
#     sigma = np.ones_like(m)
#     sigma = np.insert(sigma, [idx_sun, idx_croston], [1e-100, 1e-100])

#     # define function for beta fit
#     def beta_fit(m_range, m_c, alpha):
#         return 1 + 2. / (1 + (m_c/m_range)**alpha)

#     beta_prms, cov = opt.curve_fit(beta_fit, m_new, beta_new,
#                                    bounds=([1e13, 0], [1e15, 5]),
#                                    sigma=sigma)
#     beta_prms = {'m_c': beta_prms[0],
#                  'alpha': beta_prms[1]}

#     # define function for gamma fit
#     def gamma_fit(m_range, m_c, alpha):
#         return 1 + 2. / (1 + (m_c/m_range)**alpha)

#     gamma_prms, cov = opt.curve_fit(gamma_fit, m_new, gamma_new,
#                                    bounds=([1e13, 0], [1e15, 5]),
#                                    sigma=sigma)
#     gamma_prms = {'m_c': gamma_prms[0],
#                  'alpha': gamma_prms[1]}

#     # define function for r_c fit
#     def rc_fit(m_range, m_c, alpha):
#         return 0.4 / (1 + (m_c/m_range)**alpha)

#     rc_prms, cov = opt.curve_fit(rc_fit, m_new, r_c_new,
#                                  bounds=([1e13, 0], [1e15, 5]),
#                                  sigma=sigma)
#     rc_prms = {'m_c': rc_prms[0],
#                'alpha': rc_prms[1]}

#     pl.set_style('mark')
#     fig = plt.figure()
#     ax = fig.add_subplot(111)

#     ax.plot(m, beta)
#     ax.plot(M_sun, fit_p_sun[0]['beta'])
#     ax.plot(M_croston, fit_p_croston[0]['beta'])
#     ax.set_prop_cycle(pl.cycle_line())
#     ax.plot(m_new, beta_fit(m_new, **beta_prms))
#     ax.set_xscale('log')
#     ax.set_xlabel(r'$m_{500c} \, [M_\odot]$')
#     ax.set_ylabel(r'$\beta$')
#     plt.show()

#     pl.set_style('mark')
#     fig = plt.figure()
#     ax = fig.add_subplot(111)

#     ax.plot(m, gamma)
#     ax.plot(M_sun, fit_p_sun[0]['gamma'])
#     ax.plot(M_croston, fit_p_croston[0]['gamma'])
#     ax.set_prop_cycle(pl.cycle_line())
#     ax.plot(m_new, gamma_fit(m_new, **gamma_prms))
#     ax.set_xscale('log')
#     ax.set_xlabel(r'$m_{500c} \, [M_\odot]$')
#     ax.set_ylabel(r'$\gamma$')
#     plt.show()

#     pl.set_style('mark')
#     fig = plt.figure()
#     ax = fig.add_subplot(111)

#     ax.plot(m, r_c)
#     ax.plot(M_sun, fit_p_sun[0]['r_c'])
#     ax.plot(M_croston, fit_p_croston[0]['r_c'])
#     ax.set_prop_cycle(pl.cycle_line())
#     ax.plot(m_new, rc_fit(m_new, **rc_prms))
#     ax.set_xscale('log')
#     ax.set_xlabel(r'$m_{500c} \, [M_\odot]$')
#     ax.set_ylabel(r'$r_c$')
#     plt.show()

#     return beta_prms, gamma_prms, rc_prms

# # ------------------------------------------------------------------------------
# # End of fit_plaw_parameters()
# # ------------------------------------------------------------------------------

def prof_stars(x, a, b, m200, r200):
    profile = (b*x)**(a)
    mass = tools.m_h(profile, x * r200)
    profile *= m200/mass

    return profile

def fit_stars_bahamas():
    '''
    Fit gNFW profiles to the bahamas bins
    '''
    # bahamas data
    profiles = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles.hdf5', 'r')
    rho = profiles['PartType4/Densities'][:]
    m200 = profiles['PartType4/M200'][:]
    r200 = profiles['PartType4/R200'][:]
    r_range = tools.bins2center(profiles['RBins_R_Mean200'][:])
    numpart = profiles['PartType4/NumPartGroup'][:]
    grnr = profiles['PartType4/GroupNumber'][:].astype(int)
    relaxed = profiles['PartType4/Relaxed'][:]
    profiles.close()

    sl = ((numpart[grnr] > 1e3) & relaxed)

    rho = rho[sl]
    m200 = m200[sl]
    r200 = r200[sl] * cm2mpc

    a = np.empty_like(m200)
    b = np.empty_like(m200)

    pl.set_style('line')

    for idx, prof in enumerate(rho):
        sl = (prof > 0)
        if sl.sum() > 0:
            popt, pcov = opt.curve_fit(lambda r_range, a, b: \
                                       prof_stars(r_range, a, b,
                                                  m200[idx], r200[idx]),
                                       r_range[sl],
                                       prof[sl])
            a[idx] = popt[0]
            b[idx] = popt[1]

            plt.plot(r_range[sl], prof[sl])
            plt.plot(r_range, prof_stars(r_range, popt[0], popt[1],
                                         m200[idx], r200[idx]))
            plt.xscale('log')
            plt.yscale('log')
            plt.show()
        else:
            a[idx] = np.nan
            b[idx] = np.nan


    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # img = ax.hexbin(np.log10(m200), np.log10(c), cmap='magma',
    #                 bins='log')
    # # ax.plot(np.log10(m_range), np.log10(c_cor), c='w',
    # #         label=r'$c_{\mathrm{correa}}$')
    # # ax.plot(np.log10(m), np.log10(c_med), c='w', label=r'$c_{\mathrm{med}}$')
    # ax.set_xlabel(r'$M_{200} \, [\log_{10}M_\odot]$')
    # ax.set_ylabel(r'$\log_{10} c_{200}$')
    # ax.set_xlim([np.log10(m200).min(), np.log10(m200).max()])

    # # for line in ax.xaxis.get_ticklines():
    # #     line.set_color('w')
    # # for line in ax.yaxis.get_ticklines():
    # #     line.set_color('w')

    # # leg = ax.legend(loc='best')
    # # for text in leg.get_texts():
    # #     plt.setp(text, color='w')

    # cb = fig.colorbar(img)
    # cb.set_label(r'$\log_{10} N_{\mathrm{bin}}$', rotation=270, labelpad=25)

    # plt.show()

    return a, b

# ------------------------------------------------------------------------------
# End of fit_stars_bahamas()
# ------------------------------------------------------------------------------

def prof_gas_hot_c(x, a, b, c, m, r200):
    '''beta profile'''
    profile = (1 + (x/a)**2)**(-b/2) * np.exp(-(x/c)**2)
    mass = tools.m_h(profile, x * r200)
    profile *= m/mass

    return profile

def prof_gas_hot_s(x, a, b, m, r200):
    '''lognormal'''
    profile = np.exp(-(np.log10(x) - np.log10(a))**2/b)
    mass = tools.m_h(profile, x * r200)
    profile *= m/mass

    return profile

def fit_gas_hot_bahamas():
    '''
    Fit profiles to the hot gas for M>1e13 in bahamas
    '''
    # T3 = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_censat_200_mean_Tgt1e65.hdf5', 'r')
    T3 = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_censat_binned_200_mean_Tgt1e65_M13_15p64.hdf5', 'r')

    # get density profiles
    # rho_3_c = T3['PartType0/CenDensities'][:]
    # rho_3_s = T3['PartType0/SatDensities'][:]
    rho_3_c = T3['PartType0/CenMedianDensity'][:]
    rho_3_s = T3['PartType0/SatMedianDensity'][:]

    m200 = T3['PartType0/MedianM200'][:]
    r200 = T3['PartType0/MedianR200'][:]
    # m200 = T3['PartType0/M200'][:]
    # r200 = T3['PartType0/R200'][:] * cm2mpc

    # r_bins = T3['RBins_R_Crit500'][:]
    r_bins = T3['RBins_R_Mean200'][:]
    r_range = tools.bins2center(r_bins)

    T3.close()

    pl.set_style('line')

    mass_slice = (m200 >= 1e13)
    a_c = -1 * np.ones_like(mass_slice, dtype=float)
    b_c = -1 * np.ones_like(mass_slice, dtype=float)
    a_c_err = -1 * np.ones_like(mass_slice, dtype=float)
    b_c_err = -1 * np.ones_like(mass_slice, dtype=float)
    m_c = -1 * np.ones_like(mass_slice, dtype=float)
    a_s = -1 * np.ones_like(mass_slice, dtype=float)
    b_s = -1 * np.ones_like(mass_slice, dtype=float)
    a_s_err = -1 * np.ones_like(mass_slice, dtype=float)
    b_s_err = -1 * np.ones_like(mass_slice, dtype=float)
    r_0 = -1 * np.ones_like(mass_slice, dtype=float)
    m_s = -1 * np.ones_like(mass_slice, dtype=float)
    for idx, m in zip(np.arange(m200.shape[0])[mass_slice],
                      m200[mass_slice]):
        prof_3_c = rho_3_c[idx]
        prof_3_s = rho_3_s[idx]
        slc = (prof_3_c > 0)
        sls = ((prof_3_s > 0))# & (r_range >= 1))
        # we do not want empty slices
        if slc.sum() == 0 or sls.sum() == 0:
            continue
        # check whether region is contiguous, otherwise can run into trouble
        # with mass determination
        if np.diff((slc == 0)[:-1]).nonzero()[0].size > 2:
            continue
        if np.diff((sls == 0)[:-1]).nonzero()[0].size > 2:
            continue

        mc = tools.m_h(prof_3_c[slc], r_range[slc] * r200[idx])
        ms = tools.m_h(prof_3_s[sls], r_range[sls] * r200[idx])

        if ms > 0:
            sl = np.ones(sls.sum(), dtype=bool)
            try:
                ps, cs = opt.curve_fit(lambda r_range, a, b: \
                                       prof_gas_hot_s(r_range, a, b,
                                                      ms, r200[idx]),
                                       r_range[sls],
                                       prof_3_s[sls], bounds=([0.1, 0],
                                                              [5, 10]))
            except RuntimeError:
                continue
            # want to know actual mass in profile
            m_s[idx] = ms
            if mc > 0:
                sl = np.ones(slc.sum(), dtype=bool)
                try:
                    pc, cc = opt.curve_fit(lambda r_range, a, b: \
                                           prof_gas_hot_c(r_range, a, b, ps[0],
                                                          mc, r200[idx]),
                                           r_range[slc],
                                           prof_3_c[slc], bounds=([0, 0],
                                                                  [1, 3]))
                except RuntimeError:
                    continue

                m_c[idx] = mc
                a_c[idx] = pc[0]
                b_c[idx] = pc[1]
                a_c_err[idx] = np.sqrt(np.diag(cc))[0]
                b_c_err[idx] = np.sqrt(np.diag(cc))[1]
                # plt.plot(r_range[sls], prof_3_s[sls], label='sat')
                # plt.plot(r_range, prof_gas_hot_s(r_range, sls, ps[0], ps[1],
                #                                  ms, r200[idx]),
                #          label='sat fit')
                # plt.plot(r_range[slc], prof_3_c[slc], label='cen')
                # plt.plot(r_range, prof_gas_hot_c(r_range, slc, pc[0], pc[1],
                #                                     ps[0], mc, r200[idx]),
                #          label='cen fit')
                # plt.title(r'$M=10^{%.2f}M_\odot$'%np.log10(m))
                # plt.ylim([1e9,1e16])
                # plt.xscale('log')
                # plt.yscale('log')
                # plt.legend(loc='best')
                # plt.show()
                # only add parameters if we have fit
            # only add parameters if we have fit
            a_s[idx] = ps[0]
            b_s[idx] = ps[1]
            a_s_err[idx] = np.sqrt(np.diag(cs))[0]
            b_s_err[idx] = np.sqrt(np.diag(cs))[1]
            r_0[idx] = r_range[sls][0]
        # # always add masses
        # m_c[idx] = mc
        # m_s[idx] = ms


    mass_slice = mass_slice & (m_s > 0)
    m200 = m200[mass_slice]
    m_c = m_c[mass_slice]
    a_c = a_c[mass_slice]
    b_c = b_c[mass_slice]
    a_c_err = a_c_err[mass_slice]
    b_c_err = b_c_err[mass_slice]
    m_s = m_s[mass_slice]
    a_s = a_s[mass_slice]
    b_s = b_s[mass_slice]
    a_s_err = a_s_err[mass_slice]
    b_s_err = b_s_err[mass_slice]
    r_0 = r_0[mass_slice]

    return (m200, m_c, a_c, b_c, a_c_err, b_c_err,
            m_s, a_s, b_s, a_s_err, b_s_err, r_0)

# ------------------------------------------------------------------------------
# End of fit_gas_hot_bahamas()
# ------------------------------------------------------------------------------

def gas_hot_rc_fit(m, a, b):
    return a * (m/1e14)**(b)

def gas_hot_beta_fit(m, a, b):
    return a * (m/1e14)**(b)

def gas_hot_rs_fit(m, a, b):
    return a * (m/1e14)**(b)

def gas_hot_sigma_fit(m, a, b):
    return a * (m/1e14)**(b)

def gas_hot_mc_fit(m, mc, a, b):
    m = m/1e14
    return b * (m/mc)**a / (1 + (m/mc)**a)

def gas_hot_ms_fit(m, ms, a, b):
    m = m/1e14
    return b * (m/ms)**a / (1 + (m/ms)**a)

def gas_hot_r0_fit(m, a, b):
    return a * (m/1e14)**(b)

def plot_gas_hot_fit_bahamas_median(m200, m_c, a_c, b_c, m_s, a_s, b_s, r_0):
    pl.set_style()
    sl_ac = (a_c[m_c>0] >= 1e-5)
    # bin relation for a_c > 1e-5 & a_c < 1e-5
    m_bins = np.logspace(np.log10(m200[m_c>0]).min(),
                         np.log10(m200[m_c>0]).max(), 20)
    m = tools.bins2center(m_bins)
    m_bin_idx = np.digitize(m200[m_c>0], m_bins)

    a_c_1 = np.array([np.median(a_c[m_c>0][(m_bin_idx == m_bin) & sl_ac]) for m_bin in np.arange(1, len(m_bins))])
    a_c_2 = np.array([np.median(a_c[m_c>0][(m_bin_idx == m_bin) & ~sl_ac]) for m_bin in np.arange(1, len(m_bins))])

    ##########################################################################
    fig = plt.figure(figsize=(15,10))
    ax1 = fig.add_subplot(221)
    img1 = ax1.hexbin(np.log10(m200[m_c > 0]),
                     np.clip(np.log10(a_c[m_c > 0]), -16, 0),
                     cmap='magma',
                      bins='log',
                      gridsize=40)
    ax1.plot(np.log10(m),
            np.log10(a_c_1),
            c='w', label=r'$r_{\mathrm{med},1}$')
    ax1.plot(np.log10(m),
            np.log10(a_c_2),
            c='w', label=r'$r_{\mathrm{med},2}$')
    ax1.set_xlabel(r'$m_{500c} \, [\log_{10}M_\odot]$')
    ax1.set_ylabel(r'$\log_{10} r_c/r_{500c}$')
    ax1.set_xlim([np.log10(m200[m_c>0]).min(),
                 np.log10(m200[m_c>0]).max()])

    for line in ax1.xaxis.get_ticklines():
        line.set_color('w')
    for line in ax1.yaxis.get_ticklines():
        line.set_color('w')

    leg = ax1.legend(loc='best')
    for text in leg.get_texts():
        plt.setp(text, color='w')

    cb1 = fig.colorbar(img1)
    cb1.set_label(r'$\log_{10} N_{\mathrm{bin}}$', rotation=270, labelpad=25)

    ##########################################################################
    ax2 = fig.add_subplot(222)
    img2 = ax2.hexbin(np.log10(m200[m_c > 0]),
                      np.clip(b_c[m_c > 0], 0, 3),
                      cmap='magma',
                      bins='log',
                      gridsize=40)
    b_c_1 = np.array([np.median(b_c[m_c>0][(m_bin_idx == m_bin) & sl_ac]) for m_bin in np.arange(1, len(m_bins))])
    b_c_2 = np.array([np.median(b_c[m_c>0][(m_bin_idx == m_bin) & ~sl_ac]) for m_bin in np.arange(1, len(m_bins))])

    ax2.plot(np.log10(m),
            b_c_1,
            c='w', label=r'$\beta_{\mathrm{med},1}$')
    ax2.plot(np.log10(m),
            b_c_2,
            c='w', label=r'$\beta_{\mathrm{med},2}$')
    ax2.set_xlabel(r'$m_{500c} \, [\log_{10}M_\odot]$')
    ax2.set_ylabel(r'$\beta$')
    ax2.set_xlim([np.log10(m200[m_c>0]).min(),
                 np.log10(m200[m_c>0]).max()])

    for line in ax2.xaxis.get_ticklines():
        line.set_color('w')
    for line in ax2.yaxis.get_ticklines():
        line.set_color('w')

    leg = ax2.legend(loc='best')
    for text in leg.get_texts():
        plt.setp(text, color='w')

    cb2 = fig.colorbar(img2)
    cb2.set_label(r'$\log_{10} N_{\mathrm{bin}}$', rotation=270, labelpad=25)

    ##########################################################################
    ax3 = fig.add_subplot(223)
    img3 = ax3.hexbin(np.log10(m200[m_c > 0]),
                      np.clip(a_s[m_c > 0], 0.1, 5),
                      cmap='magma',
                      bins='log',
                      gridsize=40)

    a_s_1 = np.array([np.median(a_s[m_c>0][(m_bin_idx == m_bin) & sl_ac]) for m_bin in np.arange(1, len(m_bins))])
    a_s_2 = np.array([np.median(a_s[m_c>0][(m_bin_idx == m_bin) & ~sl_ac]) for m_bin in np.arange(1, len(m_bins))])

    ax3.plot(np.log10(m),
            a_s_1,
            c='w', label=r'$r_{\mathrm{med},1}$')
    ax3.plot(np.log10(m),
            a_s_2,
            c='w', label=r'$r_{\mathrm{med},2}$')
    ax3.set_xlabel(r'$m_{500c} \, [\log_{10}M_\odot]$')
    ax3.set_ylabel(r'$r_s/r_{500c}$')
    ax3.set_xlim([np.log10(m200[m_c>0]).min(),
                 np.log10(m200[m_c>0]).max()])

    for line in ax3.xaxis.get_ticklines():
        line.set_color('w')
    for line in ax3.yaxis.get_ticklines():
        line.set_color('w')

    leg = ax3.legend(loc='best')
    for text in leg.get_texts():
        plt.setp(text, color='w')

    cb3 = fig.colorbar(img3)
    cb3.set_label(r'$\log_{10} N_{\mathrm{bin}}$', rotation=270, labelpad=25)

    ##########################################################################
    ax4 = fig.add_subplot(224)
    img4 = ax4.hexbin(np.log10(m200[m_c > 0]),
                      np.clip(b_s[m_c > 0], 0, 1),
                      cmap='magma',
                      bins='log',
                      gridsize=40)

    b_s_1 = np.array([np.median(b_s[m_c>0][(m_bin_idx == m_bin) & sl_ac]) for m_bin in np.arange(1, len(m_bins))])
    b_s_2 = np.array([np.median(b_s[m_c>0][(m_bin_idx == m_bin) & ~sl_ac]) for m_bin in np.arange(1, len(m_bins))])
    print b_s_1
    print b_s_2

    ax4.plot(np.log10(m),
            b_s_1,
            c='w', label=r'$\sigma_{\mathrm{med},1}$')
    ax4.plot(np.log10(m),
            b_s_2,
            c='w', label=r'$\sigma_{\mathrm{med},2}$')
    ax4.set_xlabel(r'$m_{500c} \, [\log_{10}M_\odot]$')
    ax4.set_ylabel(r'$\sigma_s$')
    ax4.set_xlim([np.log10(m200[m_c>0]).min(),
                 np.log10(m200[m_c>0]).max()])

    for line in ax4.xaxis.get_ticklines():
        line.set_color('w')
    for line in ax4.yaxis.get_ticklines():
        line.set_color('w')

    leg = ax4.legend(loc='best')
    for text in leg.get_texts():
        plt.setp(text, color='w')

    cb4 = fig.colorbar(img4)
    cb4.set_label(r'$\log_{10} N_{\mathrm{bin}}$', rotation=270, labelpad=25)

    plt.show()

# ------------------------------------------------------------------------------
# End of plot_gas_hot_fit_bahamas_median()
# ------------------------------------------------------------------------------

def prof_gas_warm(x, a, b, m, r200):
    '''einasto profile'''
    profile = np.exp(-(x/a)**b)
    mass = tools.m_h(profile, x * r200)
    profile *= m/mass

    return profile

def fit_gas_warm_bahamas():
    '''
    Fit profiles to the warm gas for 1e11<M<1e13 in bahamas
    '''
    T2 = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_binned_200_mean_T1e45_1e65_M12_13.hdf5', 'r')

    # get density profiles
    rho_2 = T2['PartType0/MedianDensity'][:]

    # # want to know hot gas mass wrt halo mass
    # r500 = T2['PartType0/R500'][:]
    # numbin = T2['PartType0/NumBin'][:]
    # cum_index = np.cumsum(np.concatenate([[0], numbin]))
    # r500_med = np.array([np.median([r500[cum_index[idx]:cum_index[idx+1]]])
    #                      for idx in range(cum_index.shape[0] - 1)])
    r200 = T2['PartType0/MedianR200'][:]

    # r_bins = T2['RBins_R_Crit500'][:]
    r_bins = T2['RBins_R_Mean200'][:]
    r = tools.bins2center(r_bins)
    # m_bins = T2['MBins_M_Crit500'][:]
    m_bins = T2['MBins_M_Mean200'][:]
    m = tools.bins2center(m_bins).reshape(-1)

    T2.close()

    pl.set_style('line')

    a = -1 * np.ones_like(m)
    b = -1 * np.ones_like(m)
    a_err = -1 * np.ones_like(m)
    b_err = -1 * np.ones_like(m)
    r_0 = -1 * np.ones_like(m)
    m_2 = -1 * np.ones_like(m)
    for idx, mass in enumerate(m):
        prof_2 = rho_2[idx]
        sl2 = (prof_2 > 0)
        # we do not want empty slices
        if sl2.sum() == 0:
            continue
        # check whether region is contiguous, otherwise can run into trouble
        # with mass determination
        if np.diff((sl2 == 0)[:-1]).nonzero()[0].size > 2:
            continue

        # m2 = tools.m_h(prof_2[sl2], r[sl2] * r500_med[idx])
        m2 = tools.m_h(prof_2[sl2], r[sl2] * r200[idx])
        sl = np.ones(sl2.sum(), dtype=bool)
        p2, c2 = opt.curve_fit(lambda r, a, b: \
                               prof_gas_warm(r, a, b, m2, r200[idx]),
                               r[sl2],
                               prof_2[sl2], bounds=([0.1, 0],
                                                    [5, 10]))

        # plt.plot(r[sl2], prof_2[sl2], label='warm')
        # # plt.plot(r, prof_gas_hot_s(r, sl2, p2[0], p2[1], m2, r500_med[idx]),
        # plt.plot(r[sl2], prof_gas_warm(r[sl2], p2[0], p2[1], m2, r200[idx]),
        #          label='warm fit')
        # plt.ylim([1e9, 1e16])
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.title('$M=10^{%.2f}M_\odot$'%np.log10(mass))
        # plt.legend(loc='best')
        # plt.show()
        # only add parameters if we have fit
        a[idx] = p2[0]
        b[idx] = p2[1]
        a_err[idx] = np.sqrt(np.diag(c2))[0]
        b_err[idx] = np.sqrt(np.diag(c2))[1]
        r_0[idx] = r[sl2][0]
        m_2[idx] = m2

    sl = (a > 0)

    return a[sl], b[sl], a_err[sl], b_err[sl], r_0[sl], m_2[sl], m[sl]

# ------------------------------------------------------------------------------
# End of fit_gas_warm_bahamas()
# ------------------------------------------------------------------------------

def gas_warm_rw_fit(m, a, b):
    return a * (m/1e12)**(b)

def gas_warm_sigma_fit(m, a, b):
    return a * (m/1e12)**(b) # a * np.log10((m/1e12)) + b

def gas_warm_mw_fit(m, a, b):
    return a * (m/1e12)**(b) # a * np.log10((m/1e12)) + b

def gas_warm_r0_fit(m, a, b):
    return a * (m/1e12)**(b)

def plot_gas_warm_fit_bahamas_median(rw, sigma, rw_err, sigma_err, m_2, m):
    '''
    Plot best fit relation for rw and sigma as function of m
    m = m200
    m_2 = m_warm
    '''
    fig = plt.figure(1, figsize=(18,5))
    ax_1 = fig.add_subplot(131)
    ax_2 = fig.add_subplot(132)
    ax_3 = fig.add_subplot(133)

    rw_err[0] = rw_err[1]
    sigma_err[0] = sigma_err[1]
    ropt, rcov = opt.curve_fit(gas_warm_rw_fit, m, rw, sigma=rw_err)
    rw_prms = {'a': ropt[0], 'b': ropt[1]}
    ax_1.set_prop_cycle(pl.cycle_mark())
    ax_1.errorbar(m, rw, marker='o', yerr=rw_err, label=r'simulation')
    ax_1.set_prop_cycle(pl.cycle_line())
    ax_1.plot(m, gas_warm_rw_fit(m, **rw_prms), label=r'fit')
    ax_1.set_xscale('log')
    ax_1.set_yscale('log')
    ax_1.set_xlabel(r'$m_{500c}$')
    ax_1.set_ylabel(r'$r_w/r_{500c}$')
    ax_1.legend(loc='best')

    sopt, scov = opt.curve_fit(gas_warm_sigma_fit, m, sigma, sigma=sigma_err)
    s_prms = {'a': sopt[0], 'b': sopt[1]}
    ax_2.set_prop_cycle(pl.cycle_mark())
    ax_2.errorbar(m, sigma, marker='o', yerr=sigma_err, label=r'simulation')
    ax_2.set_prop_cycle(pl.cycle_line())
    ax_2.plot(m, gas_warm_sigma_fit(m, **s_prms), label=r'fit')
    ax_2.set_xscale('log')
    ax_2.set_yscale('log')
    ax_2.set_xlabel(r'$m_{500c}$')
    ax_2.set_ylabel(r'$\sigma_w$')
    ax_2.legend(loc='best')

    ax_3.plot(m, m_2, marker='o', label=r'simulation')
    ax_3.set_prop_cycle(pl.cycle_line())
    # ax_3.plot(m, gas_warm_sigma_fit(m, **s_prms), label=r'fit')
    ax_3.set_xscale('log')
    ax_3.set_yscale('log')
    ax_3.set_xlabel(r'$m_{500c}$')
    ax_3.set_ylabel(r'$m_{\mathrm{gas,warm}}$')
    ax_3.legend(loc='best')

    plt.show()

    return rw_prms, s_prms
# ------------------------------------------------------------------------------
# End of plot_gas_warm_fit_bahamas_median()
# ------------------------------------------------------------------------------

def prof_gas_cold(x, a, b, c, m_sl, r200):
    '''beta profile'''
    profile = (1 + (x/a)**2)**(-b) * np.exp(-(x/c)**2)
    mass = tools.m_h(profile[sl], x[sl] * r200)
    profile *= m_sl/mass

    return profile

def fit_gas_cold_bahamas(rw):
    '''
    Fit profiles to the cold gas for 1e11<M<1e13 in bahamas
    '''
    T1 = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_censat_binned_500_crit_Tlt1e45_5r500c_M11_13.hdf5', 'r')

    # get density profiles
    rho_1 = T1['PartType0/CenMedianDensity'][:]

    # want to know hot gas mass wrt halo mass
    r500 = T1['PartType0/R500'][:]
    numbin = T1['PartType0/NumBin'][:]
    cum_index = np.cumsum(np.concatenate([[0], numbin]))
    r500_med = np.array([np.median([r500[cum_index[idx]:cum_index[idx+1]]])
                         for idx in range(cum_index.shape[0] - 1)])

    r_bins = T1['RBins_R_Crit500'][:]
    r = tools.bins2center(r_bins)
    m_bins = T1['MBins_M_Crit500'][:]
    m = tools.bins2center(m_bins).reshape(-1)

    T1.close()

    pl.set_style('line')

    a = -1 * np.ones_like(m)
    b = -1 * np.ones_like(m)
    a_err = -1 * np.ones_like(m)
    b_err = -1 * np.ones_like(m)
    m_1 = -1 * np.ones_like(m)
    m200 = -1 * np.ones_like(m)
    for idx, mass in enumerate(m):
        prof_1 = rho_1[idx]
        sl1 = (prof_1 > 0)
        # we do not want empty slices
        if sl1.sum() == 0:
            continue
        # check whether region is contiguous, otherwise can run into trouble
        # with mass determination
        if np.diff((sl1 == 0)[:-1]).nonzero()[0].size > 2:
            continue

        m1 = tools.m_h(prof_1[sl1], r[sl1] * r500_med[idx])
        sl = np.ones(sl1.sum(), dtype=bool)
        p1, c1 = opt.curve_fit(lambda r, a, b: \
                               prof_gas_cold(r, sl, a, b,
                                             rw[idx], m1, r500_med[idx]),
                               r[sl1],
                               prof_1[sl1], bounds=([0, 0],
                                                    [1, 5]))
        # plt.plot(r[sl1], prof_1[sl1], label='cold')
        # plt.plot(r, prof_gas_cold(r, sl1, p1[0], p1[1], rw[idx], m1,
        #                                r500_med[idx]),
        #          label='cold fit')
        # plt.ylim([1e9, 1e16])
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.title('$M=10^{%.2f}M_\odot$'%np.log10(mass))
        # plt.legend(loc='best')
        # plt.show()
        # only add parameters if we have fit
        a[idx] = p1[0]
        b[idx] = p1[1]
        a_err[idx] = np.sqrt(np.diag(c1))[0]
        b_err[idx] = np.sqrt(np.diag(c1))[1]
        m_1[idx] = tools.m_h(prof_gas_cold(r, sl1, p1[0], p1[1], rw[idx], m1,
                                           r500_med[idx]), r)
        m200[idx] = mass

    sl = (a > 0)


    return a[sl], b[sl], a_err[sl], b_err[sl], m_1[sl], m200[sl]

# ------------------------------------------------------------------------------
# End of fit_gas_cold_bahamas()
# ------------------------------------------------------------------------------

def gas_cold_rw_fit(m, a, b):
    return a * (m/1e12)**(b)

def gas_cold_sigma_fit(m, a, b):
    return a * np.log10((m/1e12)) + b


def plot_gas_cold_fit_bahamas_median(rw, sigma, rw_err, sigma_err, m_2, m):
    '''
    Plot best fit relation for rw and sigma as function of m
    '''
    fig = plt.figure(1, figsize=(15,5))
    ax_1 = fig.add_subplot(121)
    ax_2 = fig.add_subplot(122)

    rw_err[0] = rw_err[1]
    sigma_err[0] = sigma_err[1]
    ropt, rcov = opt.curve_fit(gas_cold_rw_fit, m, rw, sigma=rw_err)
    rw_prms = {'a': ropt[0], 'b': ropt[1]}
    ax_1.set_prop_cycle(pl.cycle_mark())
    ax_1.errorbar(m, rw, marker='o', yerr=rw_err, label=r'simulation')
    ax_1.set_prop_cycle(pl.cycle_line())
    ax_1.plot(m, gas_cold_rw_fit(m, **rw_prms), label=r'fit')
    ax_1.set_xscale('log')
    ax_1.set_yscale('log')
    ax_1.set_xlabel(r'$m_{500c}$')
    ax_1.set_ylabel(r'$r_w/r_{500c}$')
    ax_1.legend(loc='best')

    sopt, scov = opt.curve_fit(gas_cold_sigma_fit, m, sigma, sigma=sigma_err)
    s_prms = {'a': sopt[0], 'b': sopt[1]}
    ax_2.set_prop_cycle(pl.cycle_mark())
    ax_2.errorbar(m, sigma, marker='o', yerr=sigma_err, label=r'simulation')
    ax_2.set_prop_cycle(pl.cycle_line())
    ax_2.plot(m, gas_cold_sigma_fit(m, **s_prms), label=r'fit')
    ax_2.set_xscale('log')
    ax_2.set_yscale('log')
    ax_2.set_xlabel(r'$m_{500c}$')
    ax_2.set_ylabel(r'$\sigma$')
    ax_2.legend(loc='best')

    plt.show()

    return rw_prms, s_prms
# ------------------------------------------------------------------------------
# End of plot_gas_cold_fit_bahamas_median()
# ------------------------------------------------------------------------------

def plot_gas_fit_bahamas_median():
    '''Fit median relations to gas fitting functions'''
    rs_w, s_w, rsw_err, sw_err, r0_w, m_w, m200_w = fit_gas_warm_bahamas()
    m200_h, mc_h, rc_h, bc_h, rc_err, bc_err, ms_h, rs_h, s_h, rs_err, s_err, r0_h = fit_gas_hot_bahamas()
    bc_h[bc_h < 1e-3] = np.nan
    rc_h[rc_h < 1e-3] = np.nan

    # warm gas
    ropt, rcov = opt.curve_fit(gas_warm_rw_fit, m200_w, rs_w, sigma=rsw_err)
    rw_prms = {'a': ropt[0], 'b': ropt[1]}
    sopt, scov = opt.curve_fit(gas_warm_sigma_fit, m200_w, s_w, sigma=sw_err)
    sw_prms = {'a': sopt[0], 'b': sopt[1]}
    ropt, rcov = opt.curve_fit(gas_warm_r0_fit, m200_w, r0_w)
    r0w_prms = {'a': ropt[0], 'b': ropt[1]}


    # fig = plt.figure(figsize=(18,6))
    # ax_w = fig.add_subplot(141)
    # ax_hc = fig.add_subplot(142)
    # ax_hs = fig.add_subplot(143)
    # ax_m = fig.add_subplot(144)
    # ax_w.set_prop_cycle(pl.cycle_mark())
    # ax_w.errorbar(m200_w, rs_w, yerr=rsw_err, marker='o', label=r'$r_w/r_{200m}$')
    # ax_w.errorbar(m200_w, s_w, yerr=sw_err, marker='x', label=r'$\sigma_w$')

    # ax_w.set_prop_cycle(pl.cycle_line())
    # ax_w.plot(m200_w, gas_warm_rw_fit(m200_w, **rw_prms))
    # ax_w.plot(m200_w, gas_warm_sigma_fit(m200_w, **sw_prms))

    # ax_w.set_xlabel(r'$m_{200m} \, [\mathrm{M}_\odot]$')
    # ax_w.legend(loc='best')
    # ax_w.set_xscale('log')
    # ax_w.set_yscale('log')
    # ax_w.set_title('Warm gas')

    # hot central gas
    rc_err[rc_err == 0] = rc_err[(rc_err == 0).nonzero()[0] - 1]
    bc_err[bc_err == 0] = bc_err[(bc_err == 0).nonzero()[0] - 1]
    ropt, rcov = opt.curve_fit(gas_hot_rc_fit, m200_h[~np.isnan(rc_h)],
                               rc_h[~np.isnan(rc_h)],
                               sigma=rc_err[~np.isnan(rc_h)])
    rc_prms = {'a': ropt[0], 'b': ropt[1]}
    bopt, bcov = opt.curve_fit(gas_hot_beta_fit, m200_h[~np.isnan(bc_h)],
                               bc_h[~np.isnan(bc_h)],
                               sigma=bc_err[~np.isnan(bc_h)])
    b_prms = {'a': bopt[0], 'b': bopt[1]}


    # ax_hc.set_prop_cycle(pl.cycle_mark())
    # ax_hc.errorbar(m200_h, rc_h, yerr=rc_err, marker='o', label=r'$r_c/r_{200m}$')
    # ax_hc.errorbar(m200_h, bc_h, yerr=bc_err, marker='x', label=r'$\beta$')

    # ax_hc.set_prop_cycle(pl.cycle_line())
    # ax_hc.plot(m200_h, gas_hot_rc_fit(m200_h, **rc_prms))
    # ax_hc.plot(m200_h, gas_hot_beta_fit(m200_h, **b_prms))

    # ax_hc.set_xlabel(r'$m_{200m} \, [\mathrm{M}_\odot]$')
    # ax_hc.legend(loc='best')
    # ax_hc.set_xscale('log')
    # ax_hc.set_yscale('log')
    # ax_hc.set_title('Hot central gas')

    # hot satellite gas
    r_sl = ~np.isnan(rs_h) & (m200_h > 1e14)
    ropt, rcov = opt.curve_fit(gas_hot_rs_fit, m200_h[r_sl],
                               rs_h[r_sl], sigma=rs_err[r_sl])
    rs_prms = {'a': ropt[0], 'b': ropt[1]}
    s_sl = ~np.isnan(s_h) & (m200_h > 1e14)
    sopt, scov = opt.curve_fit(gas_hot_sigma_fit, m200_h[s_sl],
                               s_h[s_sl], sigma=s_err[s_sl])
    ss_prms = {'a': sopt[0], 'b': sopt[1]}
    ropt, rcov = opt.curve_fit(gas_hot_r0_fit, m200_h, r0_h)
    r0s_prms = {'a': ropt[0], 'b': ropt[1]}

    # ax_hs.set_prop_cycle(pl.cycle_mark())
    # ax_hs.errorbar(m200_h, rs_h, yerr=rs_err, marker='o', label=r'$r_s/r_{200m}$')
    # ax_hs.errorbar(m200_h, s_h, yerr=s_err, marker='x', label=r'$\sigma_s$')

    # ax_hs.set_prop_cycle(pl.cycle_line())
    # ax_hs.plot(m200_h, gas_hot_rs_fit(m200_h, **rs_prms))
    # ax_hs.plot(m200_h, gas_hot_sigma_fit(m200_h, **ss_prms))

    # ax_hs.set_xlabel(r'$m_{200m} \, [\mathrm{M}_\odot]$')
    # ax_hs.legend(loc='best')
    # ax_hs.set_xscale('log')
    # ax_hs.set_yscale('log')
    # ax_hs.set_title('Hot satellite gas')

    # mass contributions
    mwopt, mwcov = opt.curve_fit(gas_warm_mw_fit, m200_w, m_w/m200_w)
    mw_prms = {'a': mwopt[0], 'b': mwopt[1]}
    mcopt, mccov = opt.curve_fit(gas_hot_mc_fit, m200_h, mc_h/m200_h)
    mc_prms = {'mc': mcopt[0], 'a': mcopt[1], 'b': mcopt[2]}
    msopt, mscov = opt.curve_fit(gas_hot_ms_fit, m200_h, ms_h/m200_h)
    ms_prms = {'ms': msopt[0], 'a': msopt[1], 'b': msopt[2]}

    # ax_m.set_prop_cycle(pl.cycle_mark())
    # ax_m.plot(m200_w, m_w/m200_w, label=r'$m_{\mathrm{warm}}$')
    # ax_m.plot(m200_h, mc_h/m200_h, label=r'$m_{\mathrm{hot},c}$')
    # ax_m.plot(m200_h, ms_h/m200_h, label=r'$m_{\mathrm{hot},s}$')

    # ax_m.set_prop_cycle(pl.cycle_line())
    # ax_m.plot(m200_w, gas_warm_mw_fit(m200_w, **mw_prms))
    # ax_m.plot(m200_h, gas_hot_mc_fit(m200_h, **mc_prms))
    # ax_m.plot(m200_h, gas_hot_ms_fit(m200_h, **ms_prms))

    # ax_m.yaxis.tick_right()
    # ax_m.yaxis.set_ticks_position('both')
    # ax_m.yaxis.set_label_position("right")
    # ax_m.set_xlabel(r'$m_{200m} \, [\mathrm{M}_\odot]$')
    # ax_m.set_ylabel(r'$m/m_{200m} \, [\mathrm{M}_\odot]$', rotation=270,
    #                 labelpad=20)
    # ax_m.legend(loc='best')
    # ax_m.set_xscale('log')
    # ax_m.set_yscale('log')
    # ax_m.set_title('Mass contribution')

    # plt.show()

    return rw_prms, sw_prms, rc_prms, b_prms, rs_prms, ss_prms, mw_prms, mc_prms, ms_prms, r0w_prms, r0s_prms

def compare_fit_gas_bahamas():
    '''
    Compare how the fit performs for binned profiles
    '''
    binned = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_binned.hdf5', 'r')

    # hot gas -> need mass fraction as function of halo mass to fit
    c, c_err, masses, fit_prms = fit_dm_bahamas()

    # warm gas

    rho = binned['PartType1/MedianDensity'][:]
    q16 = binned['PartType1/Q16'][:]
    q84 = binned['PartType1/Q84'][:]
    r200 = binned['PartType1/MedianR200'][:]

    r_bins = binned['RBins_R_Mean200'][:]
    r = tools.bins2center(r_bins)
    m_bins = binned['MBins_M_Mean200'][:]
    m = tools.bins2center(m_bins)

    binned.close()

    c_rel = np.power(10, dm_c_fit(m, **fit_prms))
    for idx, prof in enumerate(rho):
        mass = tools.m_h(prof, r * r200[idx])
        profile = prof_nfw(r, c_rel[idx], mass, r200[idx])
        plt.plot(r, prof)
        plt.plot(r, profile)
        plt.title(r'$m_{200} = 10^{%.2f}M_\odot$'%np.log10(m[idx]))
        plt.xscale('log')
        plt.yscale('log')
        plt.show()

# ------------------------------------------------------------------------------
# End of compare_fit_gas_bahamas()
# ------------------------------------------------------------------------------

def fit_dm_bahamas_all():
    '''
    Fit the concentration for the BAHAMAS simulations
    '''
    # profiles = h5py.File('halo/data/BAHAMAS/DMONLY/eagle_subfind_particles_032_profiles_200_mean.hdf5', 'r')
    profiles = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles.hdf5', 'r')
    rho = profiles['PartType1/Densities'][:]
    m200 = profiles['PartType1/M200'][:]
    r200 = profiles['PartType1/R200'][:]
    r_range = tools.bins2center(profiles['RBins_R_Mean200'][:])
    numpart = profiles['PartType1/NumPartGroup'][:]
    grnr = profiles['PartType1/GroupNumber'][:].astype(int)
    relaxed = profiles['PartType1/Relaxed'][:]
    profiles.close()

    sl = ((numpart[grnr] > 1e4) & relaxed)

    rho = rho[sl]
    m200 = m200[sl]
    r200 = r200[sl] * cm2mpc

    rs = -1 * np.ones_like(m200)
    rs_err = -1 * np.ones_like(m200)
    c = -1 * np.ones_like(m200)
    c_err = -1 * np.ones_like(m200)
    dc = -1 * np.ones_like(m200)
    dc_err = -1 * np.ones_like(m200)
    masses = -1 * np.ones_like(m200)

    pl.set_style('line')
    for idx, prof in enumerate(rho):
        # sl = (prof > 0)
        sl = ((prof > 0) & (r_range > 0.05))
        mass = tools.m_h(prof[sl], r200[idx] * r_range[sl])

        popt, pcov = opt.curve_fit(lambda r_range, rs, dc: \
                                   np.log10(prof_nfw(r_range, rs, dc)),
                                   r_range[sl]*r200[idx],
                                   np.log10(prof[sl]), bounds=([0,0],[100, 1e5]))
        masses[idx] = mass
        rs[idx] = popt[0]
        dc[idx] = popt[1]
        rs_err[idx] = np.sqrt(np.diag(pcov))[0]
        dc_err[idx] = np.sqrt(np.diag(pcov))[1]

        c[idx] = r200[idx] / rs[idx]
        # if idx%1000 == 0:
        #     profile = prof_nfw(r_range[sl]*r200[idx], rs[idx], dc[idx])
        #     print mass
        #     print tools.m_h(profile, r_range[sl] * r200[idx])
        #     print m200[idx]
        #     print '---'

        #     plt.plot(r_range[sl], prof[sl])
        #     plt.plot(r_range[sl], profile)
        #     plt.xscale('log')
        #     plt.yscale('log')
        #     plt.show()

    # copt, ccov = opt.curve_fit(dm_c_fit, m200, c)
    # c_prms = {"a": copt[0],
    #           "b": copt[1],}

    # ropt, rcov = opt.curve_fit(dm_rs_fit, m200, rs)
    # r_prms = {"a": ropt[0],
    #           "b": ropt[1]}

    # dopt, dcov = opt.curve_fit(dm_dc_fit, m200, dc)
    # d_prms = {"a": dopt[0],
    #           "b": dopt[1]}

    # fopt, fcov = opt.curve_fit(dm_f_fit, m200, masses/m200)
    # f_prms = {"a": fopt[0],
    #           "b": fopt[1]}

    return rs, rs_err, dc, dc_err, c, masses, m200#, r_prms, d_prms, f_prms

# ------------------------------------------------------------------------------
# End of fit_dm_bahamas_all()
# ------------------------------------------------------------------------------

def plot_dm_fit_bahamas_median(rs, rs_err, dc, dc_err, c, masses, m200):
    # bin relation
    m_bins = np.logspace(np.log10(m200).min(), np.log10(m200).max(), 20)
    m = tools.bins2center(m_bins)
    m_bin_idx = np.digitize(m200, m_bins)
    c_med = np.array([np.median(c[m_bin_idx == m_bin])
                                for m_bin in np.arange(1, len(m_bins))])
    popt, pcov = opt.curve_fit(dm_c_fit, m[~np.isnan(c_med)],
                               c_med[~np.isnan(c_med)])
    fit_prms = {'a': popt[0], 'b': popt[1]}
    print fit_prms

    c_cor = profs.c_correa(m, 0).reshape(-1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    img = ax.hexbin(np.log10(m200), np.log10(c),
                    cmap='magma',
                    bins='log',
                    gridsize=40)
    ax.plot(np.log10(m), np.log10(c_med), c='w', label=r'$c_{\mathrm{med}}$')
    ax.plot(np.log10(m), np.log10(dm_c_fit(m, **fit_prms)), c='w',
            label=r'$c_{\mathrm{fit}}$')
    ax.plot(np.log10(m), np.log10(c_cor), c='w',
            label=r'$c_{\mathrm{correa}}$')
    ax.set_xlabel(r'$m_{200m} \, [\log_{10}\mathrm{M}_\odot]$')
    ax.set_ylabel(r'$\log_{10} c(m)$')
    ax.set_xlim([np.log10(masses).min(), np.log10(masses).max()])

    for line in ax.xaxis.get_ticklines():
        line.set_color('w')
    for line in ax.yaxis.get_ticklines():
        line.set_color('w')

    leg = ax.legend(loc='best')
    for text in leg.get_texts():
        plt.setp(text, color='w')

    cb = fig.colorbar(img)
    cb.set_label(r'$\log_{10} N_{\mathrm{bin}}$', rotation=270, labelpad=25)

    plt.show()

# ------------------------------------------------------------------------------
# End of plot_dm_fit_bahamas_median()
# ------------------------------------------------------------------------------

# def prof_nfw(x, c, m200, r200):
#     profile = (c*x)**(-1) * (1 + c*x)**(-3)
#     mass = tools.m_h(profile, x * r200)
#     profile *= m200/mass

#     return profile
def prof_nfw(r, rs, dc):
    x = r / rs
    profile = p.prms.rho_crit * p.prms.h**2 * dc * (x)**(-1) * (1 + x)**(-2)
    # mass = tools.m_h(profile, r)
    # profile *= m200/mass

    return profile

def dm_c_fit(m, a, b):
    '''
    Fit to c
    '''
    # return a + b * np.log10(m)*(1 + c * np.log10(m)**2)
    return a * (m/1e14)**b

def dm_rs_fit(m, a, b):
    '''
    Fit to c
    '''
    # return a + b * np.log10(m)*(1 + c * np.log10(m)**2)
    return a * (m/1e14)**b

def dm_dc_fit(m, a, b):
    return a * (m/1e14)**b

def dm_f_fit(m, a, b):
    return a * (m/1e14)**b

def fit_dm_bahamas():
    '''
    Fit the concentration for the BAHAMAS simulations
    '''
    # profiles = h5py.File('halo/data/BAHAMAS/DMONLY/eagle_subfind_particles_032_profiles_binned_200_mean_M13_15.hdf5', 'r')
    profiles = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_binned.hdf5', 'r')

    rho = profiles['PartType1/MedianDensity'][:]
    m200 = profiles['PartType1/MedianM200'][:]
    r200 = profiles['PartType1/MedianR200'][:]
    r_range = tools.bins2center(profiles['RBins_R_Mean200'][:])
    profiles.close()


    rs = -1 * np.ones_like(m200)
    rs_err = -1 * np.ones_like(m200)
    c = -1 * np.ones_like(m200)
    c_err = -1 * np.ones_like(m200)
    dc = -1 * np.ones_like(m200)
    dc_err = -1 * np.ones_like(m200)
    masses = -1 * np.ones_like(m200)

    pl.set_style('line')

    for idx, prof in enumerate(rho):
        # sl = (prof > 0)
        sl = ((prof > 0) & (r_range > 0.05))
        mass = tools.m_h(prof[sl], r200[idx] * r_range[sl])

        popt, pcov = opt.curve_fit(lambda r_range, rs, dc: \
                                   np.log10(prof_nfw(r_range, rs, dc)),
                                   r_range[sl]*r200[idx],
                                   np.log10(prof[sl]), bounds=([0,0],[100, 1e5]))
        masses[idx] = mass
        rs[idx] = popt[0]
        dc[idx] = popt[1]
        rs_err[idx] = np.sqrt(np.diag(pcov))[0]
        dc_err[idx] = np.sqrt(np.diag(pcov))[1]
        c[idx] = r200[idx] / rs[idx]
        # print rs[idx]
        # print dc[idx]
        # print mass
        # print tools.m_h(prof_nfw(r_range[sl]*r200[idx], rs[idx], dc[idx]),
        #                 r_range[sl] * r200[idx])
        # print m200[idx]
        # print '---'
        # plt.plot(r_range[sl], prof[sl])
        # plt.plot(r_range[sl], prof_nfw(r_range[sl]*r200[idx], rs[idx], dc[idx]))
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()

    copt, ccov = opt.curve_fit(dm_c_fit, m200, c)
    c_prms = {"a": copt[0],
              "b": copt[1],}

    ropt, rcov = opt.curve_fit(dm_rs_fit, m200, rs)
    r_prms = {"a": ropt[0],
              "b": ropt[1]}

    dopt, dcov = opt.curve_fit(dm_dc_fit, m200, dc)
    d_prms = {"a": dopt[0],
              "b": dopt[1]}

    fopt, fcov = opt.curve_fit(dm_f_fit, m200, masses/m200)
    f_prms = {"a": fopt[0],
              "b": fopt[1]}

    # print c_prms
    # plt.plot(m200, c)
    # plt.plot(m200, dm_c_fit(m200, **c_prms))
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()

    # return c, c_err, masses, m200, c_prms, m_prms
    return rs, rs_err, dc, dc_err, masses, m200, c_prms, r_prms, d_prms, f_prms

# ------------------------------------------------------------------------------
# End of fit_dm_bahamas()
# ------------------------------------------------------------------------------

def compare_fit_dm_bahamas():
    '''
    Compare how the fit performs for binned profiles
    '''
    binned = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_binned.hdf5', 'r')

    rho = binned['PartType1/MedianDensity'][:]
    q16 = binned['PartType1/Q16'][:]
    q84 = binned['PartType1/Q84'][:]
    r200 = binned['PartType1/MedianR200'][:]
    m200 = binned['PartType1/MedianM200'][:]

    r_bins = binned['RBins_R_Mean200'][:]
    r = tools.bins2center(r_bins)
    m_bins = binned['MBins_M_Mean200'][:]
    m = tools.bins2center(m_bins)

    binned.close()
    rs1 = -1 * np.ones_like(m200)
    rs_err1 = -1 * np.ones_like(m200)
    c1 = -1 * np.ones_like(m200)
    dc1 = -1 * np.ones_like(m200)
    dc_err1 = -1 * np.ones_like(m200)
    masses1 = -1 * np.ones_like(m200)

    rs2 = -1 * np.ones_like(m200)
    rs_err2 = -1 * np.ones_like(m200)
    c2 = -1 * np.ones_like(m200)
    dc2 = -1 * np.ones_like(m200)
    dc_err2 = -1 * np.ones_like(m200)
    masses2 = -1 * np.ones_like(m200)

    rs3 = -1 * np.ones_like(m200)
    rs_err3 = -1 * np.ones_like(m200)
    c3 = -1 * np.ones_like(m200)
    dc3 = -1 * np.ones_like(m200)
    dc_err3 = -1 * np.ones_like(m200)
    masses3 = -1 * np.ones_like(m200)

    # rs4 = -1 * np.ones_like(m200)
    # rs_err4 = -1 * np.ones_like(m200)
    # c4 = -1 * np.ones_like(m200)
    # dc4 = -1 * np.ones_like(m200)
    # dc_err4 = -1 * np.ones_like(m200)
    # masses4 = -1 * np.ones_like(m200)

    pl.set_style('line')
    for idx, prof in enumerate(rho):
        sl1 = (prof > 0)
        sl2 = ((prof > 0) & (r > 0.05))
        mass1 = tools.m_h(prof[sl1], r200[idx] * r[sl1])
        mass2 = tools.m_h(prof[sl2], r200[idx] * r[sl2])

        # include all particles
        popt, pcov = opt.curve_fit(lambda r, c, dc: \
                                   np.log10(prof_nfw(r, c, dc)),
                                   r[sl1]*r200[idx],
                                   np.log10(prof[sl1]), bounds=([0,0],[100, 1e5]))
        masses1[idx] = mass1
        rs1[idx] = popt[0]
        dc1[idx] = popt[1]
        rs_err1[idx] = np.sqrt(np.diag(pcov))[0]
        dc_err1[idx] = np.sqrt(np.diag(pcov))[1]
        c1[idx] = r200[idx] / rs1[idx]

        # include all particles with r > 0.05 r_vir
        popt, pcov = opt.curve_fit(lambda r, c, dc: \
                                   np.log10(prof_nfw(r, c, dc)),
                                   r[sl2]*r200[idx],
                                   np.log10(prof[sl2]), bounds=([0,0],[100, 1e5]))

        masses2[idx] = mass2
        rs2[idx] = popt[0]
        dc2[idx] = popt[1]
        rs_err2[idx] = np.sqrt(np.diag(pcov))[0]
        dc_err2[idx] = np.sqrt(np.diag(pcov))[1]
        c2[idx] = r200[idx] / rs2[idx]

        # include all particles with r > 0.05 r_vir & no logarithmic fit
        popt, pcov = opt.curve_fit(lambda r, c, dc: \
                                   prof_nfw(r, c, dc),
                                   r[sl2]*r200[idx],
                                   prof[sl2], bounds=([0,0],[100, 1e5]))

        masses3[idx] = mass2
        rs3[idx] = popt[0]
        dc3[idx] = popt[1]
        rs_err3[idx] = np.sqrt(np.diag(pcov))[0]
        dc_err3[idx] = np.sqrt(np.diag(pcov))[1]
        c3[idx] = r200[idx] / rs3[idx]

    copt, ccov = opt.curve_fit(dm_c_fit, m200, c2)
    c_prms = {"a": copt[0],
              "b": copt[1]}

    m200c = np.array([gas.m200m_to_m200c(i) for i in m200])
    r200c = tools.mass_to_radius(m200c, 200 * p.prms.rho_crit * 0.7**2)
    c_corr_c = profs.c_correa(m200, 0).reshape(-1)

    plt.plot(m200, dm_c_fit(m200, **c_prms) * r200c/r200)
    plt.plot(m200, c_corr_c)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    dopt, dcov = opt.curve_fit(dm_dc_fit, m200, dc2)
    d_prms = {"a": dopt[0],
              "b": dopt[1]}

    c_rel = np.power(10, dm_c_fit(m200, **c_prms))
    dc_rel = dm_dc_fit(m200, **d_prms)
    for idx, prof in enumerate(rho):
        # mass = tools.m_h(prof, r * r200[idx])
        prof1 = prof_nfw(r*r200[idx], rs1[idx], dc1[idx])
        prof2 = prof_nfw(r*r200[idx], rs2[idx], dc2[idx])
        prof3 = prof_nfw(r*r200[idx], rs3[idx], dc3[idx])
        plt.plot(r, (prof * r**2)/p.prms.rho_crit, marker='o', lw=0, label='sim')
        plt.plot(r, (prof1 * r**2)/p.prms.rho_crit, label='log all')
        plt.plot(r, (prof2 * r**2)/p.prms.rho_crit, label='log $r>0.05r_{200m}$')
        plt.plot(r, (prof3 * r**2)/p.prms.rho_crit, label='no log all')
        plt.title(r'$m_{200} = 10^{%.2f}M_\odot$'%np.log10(m[idx]))
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.show()

# ------------------------------------------------------------------------------
# End of compare_fit_dm_bahamas()
# ------------------------------------------------------------------------------
