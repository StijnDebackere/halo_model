import numpy as np
import scipy.optimize as opt
import h5py
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection
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
    skip = 10
    for idx, prof in enumerate(rho_norm[::skip]):
        line1, = ax.plot(r[r>=0.05], dmo_rho_norm[::skip][idx][r>=0.05] * m[::skip][idx])
        fill1 = ax.fill_between(r[r>=0.05],
                               dmo_q16_norm[::skip][idx][r>=0.05] * m[::skip][idx],
                               dmo_q84_norm[::skip][idx][r>=0.05] * m[::skip][idx],
                               facecolor=line1.get_color(), alpha=0.3,
                               edgecolor='none')
        line, = ax.plot(r[r>=0.05], prof[r>=0.05] * m[::skip][idx])
        fill = ax.fill_between(r[r>=0.05],
                               q16_norm[::skip][idx][r>=0.05] * m[::skip][idx],
                               q84_norm[::skip][idx][r>=0.05] * m[::skip][idx],
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
                       for c in m[::skip]]).reshape(-1)
    ax.legend([handle for handle in handles],
              [lab for lab in labels],
              loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()

# ------------------------------------------------------------------------------
# End of plot_dm_bahamas()
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

# def compare_bahamas_eckert(r_bins=np.logspace(-2.5, 0.1, 20),
#                            m_bins=np.logspace(13, 15, 20)):
#     '''
#     Plot median profiles for BAHAMAS and mean profiles from Eckert
#     '''
#     # get gas profiles
#     r_eckert, rho_eckert, s, mwl, mgas = gas.rhogas_eckert()
#     m_eckert = mwl

#     m_bins_e, m_bins = m_bins_eckert()

#     # bahamas data
#     binned = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_binned_500_crit_eckert.hdf5', 'r')

#     r_bins = binned['RBins_R_Crit500'][:]
#     r = tools.bins2center(r_bins)

#     m500 = binned['PartType0/M500'][:]
#     numbin = binned['PartType0/NumBin'][:]

#     numbin = np.concatenate([[0], numbin])
#     bin_slice = np.concatenate([np.cumsum(numbin[:-1]).reshape(-1,1),
#                                 np.cumsum(numbin[1:]).reshape(-1,1)], axis=-1)

#     m = np.array([np.mean(m500[sl[0]:sl[1]]) for sl in bin_slice])

#     rho_bah = binned['PartType0/MedianDensity'][:]

#     binned.close()
#     pl.set_style()
#     fig = plt.figure()
#     ax = fig.add_subplot(111)

#     marks = []
#     lines = []
#     ax.set_prop_cycle(pl.cycle_mark())
#     for idx, prof in enumerate(rho_eckert):
#         mark = ax.errorbar(r_eckert, prof, yerr=s[idx], fmt='o')
#         marks.append(mark)

#     ax.set_prop_cycle(pl.cycle_line())
#     for prof in rho_bah:
#         line, = ax.plot(r, prof)
#         lines.append(line)

#     ax.set_xlim([0.9*r_eckert.min(), 1.1*r_eckert.max()])
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.set_xlabel(r'$r/r_{500}$')
#     ax.set_ylabel(r'$\rho(r)$ in $M_\odot/$Mpc$^3$')
#     ax.legend([(line, mark) for line, mark in zip(lines, marks)],
#               [r'$M_{\mathrm{mean},b}=10^{%.2f}M_\odot, \, M_{\mathrm{mean},e}=10^{%.2f}M_\odot$'%(np.log10(mass), np.log10(m_bins[idx]))
#                for idx, mass in enumerate(m)])
#     plt.show()

# # ------------------------------------------------------------------------------
# # End of compare_bahamas_eckert()
# # ------------------------------------------------------------------------------

# def compare_bahamas_sun_croston():
#     '''
#     Compare BAHAMAS profiles binned to reproduce median mass in Sun & Croston
#     samples.
#     '''
#     r500_sun, rho_sun, err_sun = gas.rhogas_sun()
#     r500_sun = r500_sun.reshape(-1,4).mean(axis=1)
#     rho_sun = rho_sun.reshape(-1,4).mean(axis=1)
#     err_sun = 1./2 * np.sqrt(np.sum(err_sun.reshape(2,-1,4)**2, axis=-1))

#     r500_croston, rho_croston, err_croston = gas.rhogas_croston()

#     binned = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_binned_500_crit_T1e6_papercut.hdf5', 'r')

#     r_bins = binned['RBins_R_Crit500'][:]
#     r = tools.bins2center(r_bins)

#     m500 = binned['PartType0/M500'][:]
#     numbin = binned['PartType0/NumBin'][:]
#     m = np.array([np.median(m500[:numbin[0]]),
#                   np.median(m500[numbin[0]:])])

#     rho_bah = binned['PartType0/MedianDensity'][:]

#     binned.close()
#     pl.set_style()
#     fig = plt.figure()
#     ax = fig.add_subplot(111)

#     ax.set_prop_cycle(pl.cycle_mark())
#     marks = []
#     # mark = ax.errorbar(r500_sun, rho_sun * r500_sun**2,
#     #                    yerr=[err_sun[0] * r500_sun**2,
#     #                          err_sun[1] * r500_sun**2],
#     #                    fmt='o')
#     mark = ax.errorbar(r500_sun, rho_sun,
#                        yerr=[err_sun[0],
#                              err_sun[1]],
#                        fmt='o')
#     marks.append(mark)
#     # mark = ax.errorbar(r500_croston, rho_croston * r500_croston**2,
#     #                    yerr=[err_croston[0] * r500_croston**2,
#     #                          err_croston[1] * r500_croston**2],
#     #                    fmt='o')
#     mark = ax.errorbar(r500_croston, rho_croston,
#                        yerr=[err_croston[0],
#                              err_croston[1]],
#                        fmt='o')
#     marks.append(mark)

#     ax.set_prop_cycle(pl.cycle_line())
#     lines = []
#     for idx, prof in enumerate(rho_bah):
#         # line, = ax.plot(r, r**2 * prof/(p.prms.rho_crit * 0.7**2))#*1.3)
#         line, = ax.plot(r, prof/(p.prms.rho_crit * 0.7**2))
#         lines.append(line)

#     ax.set_xlim([0.9*r500_croston.min(), 1.1*r500_croston.max()])
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.set_xlabel(r'$r/r_{500}$')
#     # ax.set_ylabel(r'$\rho(r)/\rho_{\mathrm{crit}} (r/r_{500})^2$')
#     ax.set_ylabel(r'$\rho(r)/\rho_{\mathrm{crit}}$')
#     ax.legend([(line, mark) for line, mark in zip(lines, marks)],
#               [r'$M_{\mathrm{med}}=10^{%.1f}M_\odot$'%np.log10(c) for c in m])
#     plt.show()

# # ------------------------------------------------------------------------------
# # End of compare_bahamas_sun_croston()
# # ------------------------------------------------------------------------------

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
    rho_c = binned['PartType4/CenMedianDensity'][:]
    rho_s = binned['PartType4/SatMedianDensity'][:]

    # want to know hot gas mass wrt halo mass
    r500 = Tgt1e65['PartType0/R500'][:]
    numbin = Tgt1e65['PartType0/NumBin'][:]
    cum_index = np.cumsum(np.concatenate([[0], numbin]))
    r500_med = np.array([np.median([r500[cum_index[idx]:cum_index[idx+1]]])
                         for idx in range(cum_index.shape[0] - 1)])

    m500c = tools.radius_to_mass(r500_med, 500 * p.prms.rho_crit * p.prms.h**2)
    m200m = np.array([tools.m500c_to_m200m(i) for i in m500c])
    r200 = tools.mass_to_radius(m200m, 200 * p.prms.rho_crit * p.prms.h**2 *
                                p.prms.omegab)
    # get radial range
    r_bins = Tlt1e45['RBins_R_Crit500'][:]
    # r_bins = Tlt1e45['RBins_R_Mean200'][:]
    r = tools.bins2center(r_bins)

    # get mass range
    m_bins = Tlt1e45['MBins_M_Crit500'][:]
    m_bins[-1,-1] = 1e16
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

    masses = np.array([r'$10^{%.1f}<\mathrm{m_{500\mathrm{c}}/M_\odot}<10^{%.1f}$'%(np.log10(i), np.log10(j)) for i, j in m_bins])


    axes = [ax_m1, ax_m2, ax_m3, ax_m4, ax_m5, ax_m6]
    for idx, ax in enumerate(axes):
        prof_1_c = rho_1_c[idx]
        prof_2_c = rho_2_c[idx]
        prof_3_c = rho_3_c[idx]
        prof_1_s = rho_1_s[idx]
        prof_2_s = rho_2_s[idx]
        prof_3_s = rho_3_s[idx]
        prof_c  = rho_c[idx]
        prof_s  = rho_s[idx]

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
        ax.axvline(x=r200[idx]/r500_med[idx], c='k', ls='--')
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
                # print idx
                # print ratio
                # print np.mean(ratio)
                # print np.median(ratio)
                # print '-----------------------'
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
            ax.text(0.5, 0.9, masses[idx], ha='center', transform=ax.transAxes,
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
            text = ax.set_xlabel(r'$r/r_{500\mathrm{c}}$')
            font_properties = text.get_fontproperties()

        # add annotation to virial radius
        ax.annotate(r'$r_{200\mathrm{m}}$',
                    xy=(r200[idx]/r500_med[idx], ax.get_ylim()[0]),
                    xytext=(r200[idx]/r500_med[idx] * 1.2,
                            ax.get_ylim()[0] * 2),
                    fontproperties=title_props)

    if r2:
        fig.text(0.005, 0.5,
                 r'$\rho(r) \cdot (r/r_{500\mathrm{c}})^2 \, [\mathrm{M_\odot}/\mathrm{Mpc}^{3}]$',
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

def compare_censat(r2=True):
    '''
    Compare central vs satellite profiles
    '''
    binned = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_censat_binned_500_crit_5r500c_M10_16.hdf5', 'r')
    # binned = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_censat_binned_200_mean_M10_16.hdf5', 'r')

    # get density profiles
    rho_c = binned['PartType4/CenMedianDensity'][:]
    rho_s = binned['PartType4/SatMedianDensity'][:]

    # get radial range
    r_bins = binned['RBins_R_Crit500'][:]
    # r_bins = binned['RBins_R_Mean200'][:]
    r = tools.bins2center(r_bins)

    # want to know hot gas mass wrt halo mass
    r500 = binned['PartType4/R500'][:]
    numbin = binned['PartType4/NumBin'][:]
    cum_index = np.cumsum(np.concatenate([[0], numbin]))
    r500_med = np.array([np.median([r500[cum_index[idx]:cum_index[idx+1]]])
                         for idx in range(1, cum_index.shape[0] - 1)])

    m500c = tools.radius_to_mass(r500_med, 500 * p.prms.rho_crit * p.prms.h**2)
    m200m = np.array([tools.m500c_to_m200m(i) for i in m500c])
    r200 = tools.mass_to_radius(m200m, 200 * p.prms.rho_crit * p.prms.h**2 *
                                p.prms.omegab)

    # get mass range
    m_bins = binned['MBins_M_Crit500'][:]
    # m_bins = binned['MBins_M_Mean200'][:]
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

    masses = np.array([r'$10^{%.1f}<\mathrm{m_{500\mathrm{c}}/M_\odot}<10^{%.1f}$'%(np.log10(i), np.log10(j)) for i, j in m_bins])

    axes = [ax_m1, ax_m2, ax_m3, ax_m4, ax_m5, ax_m6]
    for idx, ax in enumerate(axes):
        prof_1 = rho_c[idx]
        prof_2 = rho_s[idx]
        tot = prof_1 + prof_2
        tot[tot == 0] = np.nan
        prof_1[prof_1 == 0] = np.nan
        prof_2[prof_2 == 0] = np.nan
        if idx > 0:
            ax.axvline(x=r200[idx-1]/r500_med[idx-1], c='k', ls='--')
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
            text = ax.set_xlabel(r'$r/r_{500\mathrm{c}}$')
            font_properties = text.get_fontproperties()

        # add annotation to virial radius
        if idx > 0:
            ax.annotate(r'$r_{200\mathrm{m}}$',
                        xy=(r200[idx-1]/r500_med[idx-1], ax.get_ylim()[0]),
                        xytext=(r200[idx-1]/r500_med[idx-1] * 1.2,
                                ax.get_ylim()[0] * 2),
                        fontproperties=title_props)

    if r2:
        fig.text(0.005, 0.5,
                 r'$\rho(r) \cdot (r/r_{500\mathrm{c}})^2 \, [\mathrm{M_\odot}/\mathrm{Mpc}^{3}]$',
                 va='center', rotation='vertical', fontproperties=font_properties)
    else:
        fig.text(0.005, 0.5,
                 r'$\rho(r) \, [\mathrm{M_\odot}/\mathrm{Mpc}^{3}]$',
                 va='center', rotation='vertical', fontproperties=font_properties)

    labs = np.array(['Total', 'Central', 'Satellites'])

    ax_m1.legend(lines, labs, loc='best')

    plt.show()

# ------------------------------------------------------------------------------
# End of compare_censat()
# ------------------------------------------------------------------------------

def prof_stars_s(x, a, b, m, r200):
    '''lognormal'''
    profile = np.exp(-(np.log10(x) - np.log10(a))**2/b)
    mass = tools.m_h(profile, x * r200)
    profile *= m/mass

    return profile

def prof_stars_c(x, a, b, c, m, r200):
    profile = (1 + (x/a)**b)**(-1) * np.exp(-(x/c))
    # profile = (np.exp(-(x/a)**b) + np.exp(-x/c)) * np.exp(-(x/d))
    mass = tools.m_h(profile, x * r200)
    profile *= m/mass

    return profile

def stars_mc_fit(m, a, b, mc):
    # can also try x / (1 + x^2)
    # return a / (1 + b * np.log10(m/mc)**2) + 0.005
    plaw = (m/mc)**a * (0.5 + 0.5*(m/mc)**2)**(-a/2.)
    sigm = 10**(b/(1+(mc/m)**1))
    return plaw * sigm

def stars_ms_fit(m, a, b, mc):
    return b * (m/mc)**a / (1 + (m/mc)**a)

def fit_stars_bahamas():
    '''
    Fit profiles to the bahamas stellar bins
    '''
    # bahamas data
    profiles = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_censat_binned_200_mean_M11_15p5.hdf5', 'r')

    rho_c = profiles['PartType4/CenMedianDensity'][:]
    rho_s = profiles['PartType4/SatMedianDensity'][:]
    # rho_g = profiles['PartType0/CenMedianDensity'][:]
    m200 = profiles['PartType4/MedianM200'][:]
    r200 = profiles['PartType4/MedianR200'][:]
    r_range = tools.bins2center(profiles['RBins_R_Mean200'][:])
    profiles.close()

    a_s = -1 * np.ones_like(m200)
    b_s = -1 * np.ones_like(m200)
    a_s_err = -1 * np.ones_like(m200)
    b_s_err = -1 * np.ones_like(m200)
    r0_s = -1 * np.ones_like(m200)
    a_c = -1 * np.ones_like(m200)
    b_c = -1 * np.ones_like(m200)
    c_c = -1 * np.ones_like(m200)
    f_c = -1 * np.ones_like(m200)
    a_c_err = -1 * np.ones_like(m200)
    b_c_err = -1 * np.ones_like(m200)
    c_c_err = -1 * np.ones_like(m200)
    fc_err = -1 * np.ones_like(m200)
    m_c = -1 * np.ones_like(m200)
    m_s = -1 * np.ones_like(m200)
    # for idx, profs in enumerate(zip(rho_c, rho_s, rho_g)):
    for idx, profs in enumerate(zip(rho_c, rho_s)):
        cen = profs[0]
        sat = profs[1]
        # cold = profs[2]

        slc = (cen > 0)
        sls = (sat > 0)

        if sls.sum() > 5:
            ms = tools.m_h(sat[sls], r_range[sls] * r200[idx])
            sopt, scov = opt.curve_fit(lambda r_range, a, b: \
                                       prof_stars_s(r_range, a, b,
                                                    ms, r200[idx]),
                                       r_range[sls], sat[sls],
                                       bounds=([0.02, 0],
                                               [1, 1]))
            a_s[idx] = sopt[0]
            b_s[idx] = sopt[1]
            a_s_err[idx] = np.sqrt(np.diag(scov))[0]
            b_s_err[idx] = np.sqrt(np.diag(scov))[1]
            r0_s[idx] = r_range[sls][0]
            m_s[idx] = ms

            # plt.plot(r_range, r_range**2 * sat)
            # plt.plot(r_range[sls], r_range[sls]**2 *
            #          prof_stars_s(r_range[sls], a_s[idx], b_s[idx],m_s[idx],
            #                       r200[idx]))

        if slc.sum() > 0:
            mc = tools.m_h(cen[slc], r_range[slc] * r200[idx])
            copt, ccov = opt.curve_fit(lambda r_range, a, b, c: \
                                       prof_stars_c(r_range, a, b, c,
                                                    mc, r200[idx]),
                                       r_range[slc], cen[slc],
                                       bounds=([0, 0, 0],
                                               [1, 4, 5]))
            # slg = (cold[slc] > 0)
            # fcopt, fccov = opt.curve_fit(lambda r_range, f:\
            #                              f * prof_stars_c(r_range,
            #                                               copt[0], copt[1],
            #                                               copt[2], mc, r200[idx]),
            #                              r_range[slc][slg], cold[slc][slg],
            #                              bounds=([0],[1]))

            a_c[idx] = copt[0]
            b_c[idx] = copt[1]
            c_c[idx] = copt[2]
            # f_c[idx] = fcopt[0]
            a_c_err[idx] = np.sqrt(np.diag(ccov))[0]
            b_c_err[idx] = np.sqrt(np.diag(ccov))[1]
            c_c_err[idx] = np.sqrt(np.diag(ccov))[1]
            # fc_err[idx] = np.sqrt(np.diag(fccov))[0]
            m_c[idx] = mc
            # plt.plot(r_range[slc], r_range[slc]**2 * cen[slc])
            # plt.plot(r_range[slc], r_range[slc]**2 *
            #          prof_stars_c(r_range[slc], a_c[idx], b_c[idx], c_c[idx],
            #                       m_c[idx], r200[idx]))
            # plt.plot(r_range[slc][slg], r_range[slc][slg]**2 * cold[slc][slg])
            # plt.plot(r_range[slc][slg], r_range[slc][slg]**2 * f_c[idx] *
            #          prof_stars_c(r_range[slc][slg], a_c[idx], b_c[idx], c_c[idx],
            #                       m_c[idx], r200[idx]))
            # plt.title(r'$M=10^{%.2f}\mathrm{M_\odot}$'%np.log10(m200[idx]))
            # plt.ylim(ymin=(cen[slc]).min())
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.show()

    cen_cut = (a_c >= 0)
    sat_cut = (a_s >= 0)
    sl = (m200 > 0)
    a_s = a_s[sl]
    b_s = b_s[sl]
    a_s_err = a_s_err[sl]
    b_s_err = b_s_err[sl]
    r0_s = r0_s[sl]
    a_c = a_c[sl]
    b_c = b_c[sl]
    c_c = c_c[sl]
    f_c = f_c[sl]
    a_c_err = a_c_err[sl]
    b_c_err = b_c_err[sl]
    c_c_err = c_c_err[sl]
    fc_err = fc_err[sl]
    m_s = m_s[sl]
    m_c = m_c[sl]
    m200 = m200[sl]
    # pl.set_style('mark')
    # # plt.errorbar(m200[cen_cut], b_c[cen_cut], yerr=b_c_err[cen_cut], marker='o')
    # # plt.plot(m200[cen_cut], m_c[cen_cut]/m200[cen_cut], marker='o')
    # # plt.errorbar(m200[sat_cut], b_s[sat_cut], yerr=b_s_err[sat_cut], marker='o')
    # plt.plot(m200[sat_cut], m_s[sat_cut]/m200[sat_cut], marker='o')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()

    return a_s, a_s_err, b_s, b_s_err, r0_s, m_s, a_c, a_c_err, b_c, b_c_err, c_c, c_c_err, m_c, m200

# ------------------------------------------------------------------------------
# End of fit_stars_bahamas()
# ------------------------------------------------------------------------------

def plot_stars_fit_bahamas_median(plot=False):
    '''Fit median relations to stars fitting functions'''
    a_s, a_s_err, b_s, b_s_err, r0_s, m_s, a_c, a_c_err, b_c, b_c_err, c_c, c_c_err, m_c, m200 = fit_stars_bahamas()

    # centrals
    cen_cut = ((a_c >= 0) & (m200>0))
    acopt, accov = opt.curve_fit(dm_plaw_fit, m200[cen_cut],
                                 a_c[cen_cut], sigma=a_c_err[cen_cut])
    ac_prms = {'a': acopt[0], 'b': acopt[1]}
    bcopt, bccov = opt.curve_fit(dm_plaw_fit, m200[cen_cut],
                                 b_c[cen_cut], sigma=b_c_err[cen_cut])
    bc_prms = {'a': bcopt[0], 'b': bcopt[1]}
    cc_cut = (c_c[cen_cut] < 4)
    ccopt, cccov = opt.curve_fit(dm_plaw_fit, m200[cen_cut][cc_cut],
                                 c_c[cen_cut][cc_cut],
                                 sigma=c_c_err[cen_cut][cc_cut])
    cc_prms = {'a': ccopt[0], 'b': ccopt[1]}
    mc_cut = ((m_c/m200)[cen_cut] < 0.28)
    mcopt, mccov = opt.curve_fit(stars_mc_fit, m200[cen_cut][mc_cut],
                                 (m_c/m200)[cen_cut][mc_cut],
                                 bounds=([0, -5, 1e8],
                                         [5, 0, 1e15]))
                                 # bounds=([0, -5, 0, 1e8],
                                 #         [5, 0, 3, 1e15]))
    mc_prms = {'a': mcopt[0],
               'b': mcopt[1],
               # 'c': mcopt[2],
               # 'mc': mcopt[3]}
               'mc': mcopt[2]}

    if plot:
        fig = plt.figure(figsize=(14,6))
        axc = fig.add_subplot(131)
        axs = fig.add_subplot(132)
        axm = fig.add_subplot(133)

        axc.set_prop_cycle(pl.cycle_mark())
        axc.errorbar(m200[cen_cut], a_c[cen_cut], yerr=a_c_err[cen_cut],
                     marker='o', label=r'$r_c/r_{200\mathrm{m}}$')
        axc.errorbar(m200[cen_cut], b_c[cen_cut], yerr=b_c_err[cen_cut],
                     marker='x', label=r'$\beta_c$')
        axc.errorbar(m200[cen_cut], c_c[cen_cut], yerr=c_c_err[cen_cut],
                     marker='x', label=r'$r_x/r_{200\mathrm{m}}$')
        axc.set_prop_cycle(pl.cycle_line())
        axc.plot(m200, dm_plaw_fit(m200, **ac_prms))
        axc.plot(m200, dm_plaw_fit(m200, **bc_prms))
        axc.plot(m200, dm_plaw_fit(m200, **cc_prms))
        axc.set_xlabel(r'$m_{200\mathrm{m}} \, [\mathrm{M}_\odot]$')
        axc.set_ylim(ymin=1e-4)
        axc.legend(loc=2)
        axc.set_xscale('log')
        axc.set_yscale('log')
        axc.set_title('Central galaxies')

    # satellites
    sat_cut = (a_s >= 0)
    asopt, accov = opt.curve_fit(dm_plaw_fit, m200[sat_cut],
                                 a_s[sat_cut], sigma=a_s_err[sat_cut])
    as_prms = {'a': asopt[0], 'b': asopt[1]}
    bsopt, bccov = opt.curve_fit(dm_plaw_fit, m200[sat_cut],
                                 b_s[sat_cut], sigma=b_s_err[sat_cut])
    bs_prms = {'a': bsopt[0], 'b': bsopt[1]}
    ropt, rcov = opt.curve_fit(dm_plaw_fit, m200[sat_cut], r0_s[sat_cut])
    r0_prms = {'a': ropt[0], 'b': ropt[1]}
    msopt, mccov = opt.curve_fit(stars_ms_fit, m200[sat_cut],
                                 (m_s/m200)[sat_cut],
                                 bounds=([0, 0, 1e11],
                                         [3, 3, 1e15]))
    ms_prms = {'a': msopt[0],
               'b': msopt[1],
               'mc': msopt[2]}

    if plot:
        axs.set_prop_cycle(pl.cycle_mark())
        axs.errorbar(m200[sat_cut], a_s[sat_cut], yerr=a_s_err[sat_cut],
                     marker='o', label=r'$r_s/r_{200\mathrm{m}}$')
        axs.errorbar(m200[sat_cut], b_s[sat_cut], yerr=b_s_err[sat_cut],
                     marker='x', label=r'$\beta_s$')
        axs.set_prop_cycle(pl.cycle_line())
        axs.plot(m200, dm_plaw_fit(m200, **as_prms))
        axs.plot(m200, dm_plaw_fit(m200, **bs_prms))
        axs.set_xlabel(r'$m_{200\mathrm{m}} \, [\mathrm{M}_\odot]$')
        axs.legend(loc='best')
        axs.set_xscale('log')
        axs.set_yscale('log')
        axs.set_title('Satellite galaxies')

        # mass contributions
        axm.set_prop_cycle(pl.cycle_mark())
        axm.plot(m200[cen_cut], (m_c/m200)[cen_cut],
                 label=r'$m_{\mathrm{cen}}/m_{200\mathrm{m}}$')
        axm.plot(m200[sat_cut], (m_s/m200)[sat_cut],
                 label=r'$m_{\mathrm{sat}}/m_{200\mathrm{m}}$')
        axm.set_prop_cycle(pl.cycle_line())
        axm.plot(m200, stars_mc_fit(m200, **mc_prms))
        axm.plot(m200, stars_ms_fit(m200, **ms_prms))
        axm.yaxis.tick_right()
        axm.yaxis.set_ticks_position('both')
        axm.yaxis.set_label_position("right")
        axm.set_xlabel(r'$m_{200\mathrm{m}} \, [\mathrm{M}_\odot]$')
        axm.set_ylabel(r'$m/m_{200\mathrm{m}} \, [\mathrm{M}_\odot]$',
                       rotation=270,
                       labelpad=20)
        axm.legend(loc='best')
        axm.set_xscale('log')
        axm.set_yscale('log')
        axm.set_title('Mass contribution')

        plt.show()

    return as_prms, bs_prms, r0_prms, ms_prms, ac_prms, bc_prms, cc_prms, mc_prms

# ------------------------------------------------------------------------------
# End of plot_stars_fit_bahamas_median()
# ------------------------------------------------------------------------------

def compare_fit_stars_bahamas():
    '''
    Compare how the fit performs for binned profiles
    '''
    binned = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_censat_binned_200_mean_M11_15p5.hdf5', 'r')

    rho_c = binned['PartType4/CenMedianDensity'][:]
    rho_s = binned['PartType4/SatMedianDensity'][:]
    r200_c = binned['PartType4/MedianR200'][:]
    m200_c = binned['PartType4/MedianM200'][:]

    # r_bins = binned['RBins_R_Mean200'][:]
    # r = tools.bins2center(r_bins).reshape(-1)
    # m_bins = binned['MBins_M_Mean200'][:][[1,3,4,5]]
    r_bins = binned['RBins_R_Mean200'][:]
    r = tools.bins2center(r_bins).reshape(-1)
    m_bins = binned['MBins_M_Mean200'][:]

    binned.close()

    as_prms, bs_prms, r0_prms, ms_prms, ac_prms, bc_prms, cc_prms, mc_prms = plot_stars_fit_bahamas_median()

    a_s = dm_plaw_fit(m200, **as_prms)
    b_s = dm_plaw_fit(m200, **bs_prms)
    r0_s = dm_plaw_fit(m200, **r0_prms)
    ms = m200 * stars_ms_fit(m200, **ms_prms)

    a_c = dm_plaw_fit(m200, **ac_prms)
    b_c = dm_plaw_fit(m200, **bc_prms)
    c_c = dm_plaw_fit(m200, **cc_prms)
    mc = m200 * stars_mc_fit(m200, **mc_prms)

    pl.set_style()
    fig = plt.figure(figsize=(20,6))
    ax1 = fig.add_axes([0.1, 0.1, 0.2, 0.8])
    ax2 = fig.add_axes([0.3, 0.1, 0.2, 0.8])
    ax3 = fig.add_axes([0.5, 0.1, 0.2, 0.8])
    ax4 = fig.add_axes([0.7, 0.1, 0.2, 0.8])

    masses = np.array([1e12, 1e13, 1e14, 1e15])
    axes = [ax1, ax2, ax3, ax4]

    for idx, mass in enumerate(masses):
        # find closest matching halo
        idx_match = np.argmin(np.abs(m200 - mass))
        prof = rho[idx_match]
        # m_prof = tools.m_h(prof, r * r200[idx_match])

        prof_s = np.zeros_like(r)
        # get satellite profile
        sls = (r > r0_s[idx_match])
        if sls.sum() > 1:
            prof_s[sls] = prof_stars_s(r[sls], a_s[idx_match], b_s[idx_match],
                                       ms[idx_match], r200[idx_match])

        # get central profile
        prof_c = prof_stars_c(r, a_c[idx_match], b_c[idx_match], c_c[idx_match],
                              mc[idx_match], r200[idx_match])

        prof_t = prof_s + prof_c
        # mass3 = tools.m_h(prof_t, r * r200[idx_match])
        # scale = m_prof / mass3

        prof[prof == 0] = np.nan
        prof_c[prof_c == 0] = np.nan
        prof_s[prof_s == 0] = np.nan
        prof_t[prof_t == 0] = np.nan


        axes[idx].plot(r, (prof * r**2),
                       marker='o', lw=0, label='sim')
        axes[idx].plot(r, (prof_c * r**2), label='cen')
        axes[idx].plot(r, (prof_s * r**2), label='sat')
        axes[idx].plot(r, (prof_t * r**2), label='tot')
        axes[idx].set_title(r'$m_{200\mathrm{m}} = 10^{%.2f}\mathrm{M_\odot}$'
                        %np.log10(m200[idx_match]))
        axes[idx].set_ylim([5e9, 2e13])
        if idx == 0:
            text = axes[idx].set_ylabel(r'$\rho(r) \cdot (r/r_{200\mathrm{m}})^2 \, [\mathrm{M_\odot/Mpc^3}]$')
            font_properties = text.get_fontproperties()
        # need to set visibility to False BEFORE log scaling
        if idx > 0:
            ticks = axes[idx].get_xticklabels()
            # strange way of getting correct label
            ticks[-7].set_visible(False)

        axes[idx].set_xscale('log')
        axes[idx].set_yscale('log')
        if idx > 0:
            axes[idx].yaxis.set_ticklabels([])

    fig.text(0.5, 0.03,
             r'$r/r_{200\mathrm{m}}$', ha='center',
             va='center', rotation='horizontal', fontproperties=font_properties)
    ax1.legend(loc='best')
    plt.show()

# ------------------------------------------------------------------------------
# End of compare_fit_stars_bahamas()
# ------------------------------------------------------------------------------

def prof_gas_hot_beta(x, a, b, m, r200):
    '''beta profile'''
    profile = (1 + (x/a)**2)**(-b/2)
    mass = tools.m_h(profile, x * r200)
    profile *= m/mass

    return profile

def prof_gas_hot(x, sl, a, b, m_sl, r500):
    '''beta profile'''
    profile = (1 + (x/a)**2)**(-b/2.)
    mass = tools.m_h(profile[sl], x[sl] * r500)
    profile *= m_sl/mass

    return profile

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

def fit_gas_hot_bahamas_paper():
    '''
    Fit beta profiles to the total hot gas for M>1e13 in bahamas
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
    rc = -1 * np.ones_like(mass_slice, dtype=float)
    b = -1 * np.ones_like(mass_slice, dtype=float)
    rc_err = -1 * np.ones_like(mass_slice, dtype=float)
    b_err = -1 * np.ones_like(mass_slice, dtype=float)
    m_g = -1 * np.ones_like(mass_slice, dtype=float)

    for idx, m in zip(np.arange(m200.shape[0])[mass_slice],
                      m200[mass_slice]):
        prof_3_c = rho_3_c[idx]
        prof_3_s = rho_3_s[idx]
        prof_3 = prof_3_c + prof_3_s

        sl_0 = (prof_3 == 0)
        sl = ((sl_0.sum() <= 3) & (r_range > 0.05))
        # we do not want empty slices
        if sl.sum() == 0:
            continue
        # check whether region is contiguous, otherwise can run into trouble
        # with mass determination
        if np.diff((sl == 0)[:-1]).nonzero()[0].size > 2:
            continue

        mg_tot = tools.m_h(prof_3[sl], r_range[sl] * r200[idx])

        if mg_tot > 0:
            try:
                p, c = opt.curve_fit(lambda r_range, rc, b: \
                                     prof_gas_hot_beta(r_range, rc, b,
                                                       mg_tot, r200[idx]),
                                     r_range[sl], prof_3[sl],
                                     bounds=([0., 0],
                                             [1., 3]))
            except RuntimeError:
                continue

            m_g[idx] = mg_tot
            rc[idx] = p[0]
            b[idx] = p[1]
            rc_err[idx] = np.sqrt(np.diag(c))[0]
            b_err[idx] = np.sqrt(np.diag(c))[1]

            # plt.plot(r_range[sl], prof_3[sl], label='tot')
            # plt.plot(r_range, prof_gas_hot_beta(r_range, p[0], p[1],
            #                                     m_g[idx], r200[idx]),
            #          label='tot fit')
            # plt.title(r'$M=10^{%.2f}M_\odot$'%np.log10(m))
            # plt.ylim([1e9,1e16])
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.legend(loc='best')
            # plt.show()

    mass_slice = mass_slice & (m_g > 0)
    m200 = m200[mass_slice]
    m_g = m_g[mass_slice]
    rc = rc[mass_slice]
    b = b[mass_slice]
    rc_err = rc_err[mass_slice]
    b_err = b_err[mass_slice]

    return (m200, m_g, rc, b, rc_err, b_err)

# ------------------------------------------------------------------------------
# End of fit_gas_hot_bahamas_paper()
# ------------------------------------------------------------------------------

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

def gas_warm_rw_fit(m, a, b):
    return a * (m/1e12)**(b)

def gas_warm_sigma_fit(m, a, b):
    return a * (m/1e12)**(b) # a * np.log10((m/1e12)) + b

def gas_warm_mw_fit(m, a, b):
    return a * (m/1e12)**(b) # a * np.log10((m/1e12)) + b

def gas_warm_r0_fit(m, a, b):
    return a * (m/1e12)**(b)

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
    # plt.plot(m[sl], r_0[sl])
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()

    return a[sl], b[sl], a_err[sl], b_err[sl], r_0[sl], m_2[sl], m[sl]

# ------------------------------------------------------------------------------
# End of fit_gas_warm_bahamas()
# ------------------------------------------------------------------------------

def plot_gas_fit_bahamas_median(plot=False):
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

    if plot:
        fig = plt.figure(figsize=(20,6))
        ax_w = fig.add_subplot(141)
        ax_hc = fig.add_subplot(143)
        ax_hs = fig.add_subplot(142)
        ax_m = fig.add_subplot(144)
        ax_w.set_prop_cycle(pl.cycle_mark())
        ax_w.errorbar(m200_w, rs_w, yerr=rsw_err, marker='o',
                      label=r'$r_w/r_{200\mathrm{m}}$')
        ax_w.errorbar(m200_w, s_w, yerr=sw_err, marker='x',
                      label=r'$\sigma_w$')

        ax_w.set_prop_cycle(pl.cycle_line())
        ax_w.plot(m200_w, gas_warm_rw_fit(m200_w, **rw_prms))
        ax_w.plot(m200_w, gas_warm_sigma_fit(m200_w, **sw_prms))

        ax_w.set_xlabel(r'$m_{200\mathrm{m}} \, [\mathrm{M}_\odot]$')
        ax_w.legend(loc='best')
        ax_w.set_xscale('log')
        ax_w.set_yscale('log')
        ax_w.set_title('Warm gas')

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

    if plot:
        ax_hc.set_prop_cycle(pl.cycle_mark())
        ax_hc.errorbar(m200_h, rc_h, yerr=rc_err, marker='o',
                       label=r'$r_c/r_{200\mathrm{m}}$')
        ax_hc.errorbar(m200_h, bc_h, yerr=bc_err, marker='x',
                       label=r'$\beta$')

        ax_hc.set_prop_cycle(pl.cycle_line())
        ax_hc.plot(m200_h, gas_hot_rc_fit(m200_h, **rc_prms))
        ax_hc.plot(m200_h, gas_hot_beta_fit(m200_h, **b_prms))

        ax_hc.set_xlabel(r'$m_{200\mathrm{m}} \, [\mathrm{M}_\odot]$')
        ax_hc.legend(loc='best')
        ax_hc.set_xscale('log')
        ax_hc.set_yscale('log')
        ax_hc.set_title('Hot central gas')

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

    if plot:
        ax_hs.set_prop_cycle(pl.cycle_mark())
        ax_hs.errorbar(m200_h, rs_h, yerr=rs_err, marker='o',
                       label=r'$r_s/r_{200\mathrm{m}}$')
        ax_hs.errorbar(m200_h, s_h, yerr=s_err, marker='x',
                       label=r'$\sigma_s$')

        ax_hs.set_prop_cycle(pl.cycle_line())
        ax_hs.plot(m200_h, gas_hot_rs_fit(m200_h, **rs_prms))
        ax_hs.plot(m200_h, gas_hot_sigma_fit(m200_h, **ss_prms))

        ax_hs.set_xlabel(r'$m_{200\mathrm{m}} \, [\mathrm{M}_\odot]$')
        ax_hs.legend(loc='best')
        ax_hs.set_xscale('log')
        ax_hs.set_yscale('log')
        ax_hs.set_title('Hot satellite gas')

    # mass contributions
    mwopt, mwcov = opt.curve_fit(gas_warm_mw_fit, m200_w, m_w/m200_w)
    mw_prms = {'a': mwopt[0], 'b': mwopt[1]}
    mcopt, mccov = opt.curve_fit(gas_hot_mc_fit, m200_h, mc_h/m200_h)
    mc_prms = {'mc': mcopt[0], 'a': mcopt[1], 'b': mcopt[2]}
    msopt, mscov = opt.curve_fit(gas_hot_ms_fit, m200_h, ms_h/m200_h)
    ms_prms = {'ms': msopt[0], 'a': msopt[1], 'b': msopt[2]}

    if plot:
        ax_m.set_prop_cycle(pl.cycle_mark())
        ax_m.plot(m200_w, m_w/m200_w, label=r'$m_{\mathrm{warm}}$')
        ax_m.plot(m200_h, mc_h/m200_h, label=r'$m_{\mathrm{hot,c}}$')
        ax_m.plot(m200_h, ms_h/m200_h, label=r'$m_{\mathrm{hot,s}}$')

        ax_m.set_prop_cycle(pl.cycle_line())
        ax_m.plot(m200_w, gas_warm_mw_fit(m200_w, **mw_prms))
        ax_m.plot(m200_h, gas_hot_mc_fit(m200_h, **mc_prms))
        ax_m.plot(m200_h, gas_hot_ms_fit(m200_h, **ms_prms))

        ax_m.yaxis.tick_right()
        ax_m.yaxis.set_ticks_position('both')
        ax_m.yaxis.set_label_position("right")
        ax_m.set_xlabel(r'$m_{200\mathrm{m}} \, [\mathrm{M}_\odot]$')
        ax_m.set_ylabel(r'$m/m_{200\mathrm{m}} \, [\mathrm{M}_\odot]$', rotation=270,
                        labelpad=20)
        ax_m.legend(loc='best')
        ax_m.set_xscale('log')
        # ax_m.set_yscale('log')
        ax_m.set_title('Mass contribution')

        plt.show()

    return rw_prms, sw_prms, rc_prms, b_prms, rs_prms, ss_prms, mw_prms, mc_prms, ms_prms, r0w_prms, r0s_prms

# ------------------------------------------------------------------------------
# End of plot_gas_fit_bahamas_median()
# ------------------------------------------------------------------------------

def plot_gas_fit_bahamas_median_paper(plot=False):
    '''Fit median relations to gas fitting functions'''
    m200, m_g, rc, b, rc_err, b_err = fit_gas_hot_bahamas_paper()

    if plot:
        fig = plt.figure(figsize=(10,6))
        ax_h = fig.add_subplot(121)
        ax_m = fig.add_subplot(122)

    # hot central gas
    rc_err[rc_err == 0] = rc_err[(rc_err == 0).nonzero()[0] - 1]
    b_err[b_err == 0] = b_err[(b_err == 0).nonzero()[0] - 1]

    # ropt, rcov = opt.curve_fit(gas_hot_rc_fit, m200_h[~np.isnan(rc_h)],
    #                            rc_h[~np.isnan(rc_h)],
    #                            sigma=rc_err[~np.isnan(rc_h)])
    # rc_prms = {'a': ropt[0], 'b': ropt[1]}

    # bopt, bcov = opt.curve_fit(gas_hot_beta_fit, m200_h[~np.isnan(bc_h)],
    #                            bc_h[~np.isnan(bc_h)],
    #                            sigma=bc_err[~np.isnan(bc_h)])
    # b_prms = {'a': bopt[0], 'b': bopt[1]}

    if plot:
        ax_h.set_prop_cycle(pl.cycle_mark())
        ax_h.errorbar(m200_h, rc_h, yerr=rc_err, marker='o',
                       label=r'$r_c/r_{200\mathrm{m}}$')
        ax_h.errorbar(m200_h, bc_h, yerr=bc_err, marker='x',
                       label=r'$\beta$')

        ax_h.set_prop_cycle(pl.cycle_line())
        ax_h.plot(m200_h, gas_hot_rc_fit(m200_h, **rc_prms))
        ax_h.plot(m200_h, gas_hot_beta_fit(m200_h, **b_prms))

        ax_h.set_xlabel(r'$m_{200\mathrm{m}} \, [\mathrm{M}_\odot]$')
        ax_h.legend(loc='best')
        ax_h.set_xscale('log')
        ax_h.set_yscale('log')
        ax_h.set_title('Hot central gas')

    # mass contributions
    mhopt, mhcov = opt.curve_fit(gas_hot_mc_fit, m200_h, mc_h/m200_h)
    mh_prms = {'mc': mcopt[0], 'a': mcopt[1], 'b': mcopt[2]}

    if plot:
        ax_m.set_prop_cycle(pl.cycle_mark())
        ax_m.plot(m200_w, m_w/m200_w, label=r'$m_{\mathrm{warm}}$')
        ax_m.plot(m200_h, mc_h/m200_h, label=r'$m_{\mathrm{hot,c}}$')
        ax_m.plot(m200_h, ms_h/m200_h, label=r'$m_{\mathrm{hot,s}}$')

        ax_m.set_prop_cycle(pl.cycle_line())
        ax_m.plot(m200_w, gas_warm_mw_fit(m200_w, **mw_prms))
        ax_m.plot(m200_h, gas_hot_mc_fit(m200_h, **mc_prms))
        ax_m.plot(m200_h, gas_hot_ms_fit(m200_h, **ms_prms))

        ax_m.yaxis.tick_right()
        ax_m.yaxis.set_ticks_position('both')
        ax_m.yaxis.set_label_position("right")
        ax_m.set_xlabel(r'$m_{200\mathrm{m}} \, [\mathrm{M}_\odot]$')
        ax_m.set_ylabel(r'$m/m_{200\mathrm{m}} \, [\mathrm{M}_\odot]$', rotation=270,
                        labelpad=20)
        ax_m.legend(loc='best')
        ax_m.set_xscale('log')
        # ax_m.set_yscale('log')
        ax_m.set_title('Mass contribution')

        plt.show()

    return rw_prms, sw_prms, rc_prms, b_prms, rs_prms, ss_prms, mw_prms, mc_prms, ms_prms, r0w_prms, r0s_prms

# ------------------------------------------------------------------------------
# End of plot_gas_fit_bahamas_median_paper()
# ------------------------------------------------------------------------------

def compare_fit_gas_bahamas():
    '''
    Compare how the fit performs for binned profiles
    '''
    # Load warm and hot data
    T2 = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_censat_binned_200_mean_T1e45_1e65_M11_15p5.hdf5', 'r')
    T3 = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_censat_binned_200_mean_Tgt1e65_M11_15p5.hdf5', 'r')

    # rho = binned['PartType0/MedianDensity'][1:]
    # q16 = binned['PartType0/Q16'][1:]
    # q84 = binned['PartType0/Q84'][1:]
    # r200 = binned['PartType0/MedianR200'][1:]
    # m200 = binned['PartType0/MedianM200'][1:]
    rho_w = T2['PartType0/CenMedianDensity'][:] + T2['PartType0/SatMedianDensity'][:]
    rho_h = T3['PartType0/CenMedianDensity'][:] + T3['PartType0/SatMedianDensity'][:]
    m200 = T2['PartType0/MedianM200'][:]
    r200 = T2['PartType0/MedianR200'][:]

    r_bins = T3['RBins_R_Mean200'][:]
    r = tools.bins2center(r_bins).reshape(-1)
    m_bins = T3['MBins_M_Mean200'][1:]
    m = tools.bins2center(m_bins).reshape(-1)

    T2.close()
    T3.close()

    r_x = r200
    # extract all fit parameters
    rw_prms, sw_prms, rc_prms, b_prms, rs_prms, ss_prms, mw_prms, mc_prms, ms_prms, r0w_prms, r0s_prms = plot_gas_fit_bahamas_median()

    # need stars fit parameters for cold gas
    as_prms, bs_prms, r0_prms, mscold_prms, ac_prms, bc_prms, cc_prms, mcold_prms = plot_stars_fit_bahamas_median()

    f_w = gas_warm_mw_fit(m200, **mw_prms)
    fc_h = gas_hot_mc_fit(m200, **mc_prms)
    fs_h = gas_hot_ms_fit(m200, **ms_prms)
    f_c = stars_mc_fit(m200, **mcold_prms)

    # warm gas
    rw = gas_warm_rw_fit(m200, **rw_prms)
    sw = gas_warm_sigma_fit(m200, **sw_prms)
    r0w = gas_warm_r0_fit(m200, **r0w_prms)
    prof_w = np.zeros((m200.shape[0], r.shape[0]))
    for idx, mass in enumerate(m200):
        if mass <= 1e13:
            r_sl = (r >= r0w[idx] * r_x[idx])
            if r_sl.sum() > 0:
                prof_w[idx, r_sl] = prof_gas_warm(r[r_sl], rw[idx],
                                                  sw[idx],
                                                  f_w[idx] * mass,
                                                  r_x[idx])

    # hot cen gas
    rc = gas_hot_rc_fit(m200, **rc_prms)
    bc = gas_hot_beta_fit(m200, **b_prms)
    rs = gas_hot_rs_fit(m200, **rs_prms)
    prof_c = np.zeros((m200.shape[0], r.shape[0]))
    for idx, mass in enumerate(m200):
        if (mass >= 1e13):
            prof_c[idx] = prof_gas_hot_c(r, rc[idx], bc[idx], rs[idx],
                                         fc_h[idx] * mass, r_x[idx])

    # hot sat gas
    ss = gas_hot_sigma_fit(m200, **ss_prms)
    r0s = gas_hot_r0_fit(m200, **r0s_prms)
    prof_s = np.zeros((m200.shape[0], r.shape[0]))
    for idx, mass in enumerate(m200):
        if mass >= 1e13:
            r_sl = (r >= r0s[idx] * r_x[idx])
            if r_sl.sum() > 0:
                prof_s[idx, r_sl] = prof_gas_hot_s(r[r_sl], rs[idx], ss[idx],
                                                   fs_h[idx] * mass, r_x[idx])


    # # cold gas
    # ac = dm_plaw_fit(m200, **ac_prms)
    # bc = dm_plaw_fit(m200, **bc_prms)
    # cc = dm_plaw_fit(m200, **cc_prms)
    # prof_cold = np.zeros((m.shape[0], r.shape[0]))
    # for idx, mass in enumerate(m):
    #     if mass < 1e13:
    #         prof_cold[idx] = 0.4 * prof_stars_c(r, ac[idx], bc[idx], cc[idx],
    #                                             f_c[idx] * m[idx], r_x[idx])

    # for idx, profs in enumerate(zip(rho_w, rho_h)):
    #     ch = prof_c[idx]
    #     sh = prof_s[idx]
    #     w = prof_w[idx]
    #     prof_t = (ch + sh + w)
    #     # c = prof_cold[idx]

    #     sim_w = profs[0]
    #     sim_h = profs[1]
    #     sim_t = sim_w + sim_h

    #     if (prof_t > 0).any():
    #         sim_t[sim_t == 0] = np.nan
    #         prof_t[prof_t == 0] = np.nan

    #         plt.plot(r, r**2 * sim_t, marker='o', lw=0, label='sim')
    #         plt.plot(r, r**2 * prof_t, label='tot')
    #         # c[c == 0] = np.nan
    #         ch[ch == 0] = np.nan
    #         sh[sh == 0] = np.nan
    #         w[w == 0] = np.nan
    #         # plt.plot(r, r**2 * c, label='cold')
    #         if m200[idx] > 1e13:
    #             plt.plot(r, r**2 * ch, label='cen')
    #             plt.plot(r, r**2 * sh, label='sat')
    #         else:
    #             plt.plot(r, r**2 * w, label='warm')
    #         plt.title(r'$M=10^{%.2f}\mathrm{M_\odot}$'%np.log10(m200[idx]))
    #         # plt.ylim(ymin=1e10)
    #         plt.xscale('log')
    #         plt.yscale('log')
    #         plt.legend(loc='best')
    #         plt.show()
    pl.set_style()
    fig = plt.figure(figsize=(20,6))
    ax1 = fig.add_axes([0.1, 0.1, 0.2, 0.8])
    ax2 = fig.add_axes([0.3, 0.1, 0.2, 0.8])
    ax3 = fig.add_axes([0.5, 0.1, 0.2, 0.8])
    ax4 = fig.add_axes([0.7, 0.1, 0.2, 0.8])

    masses = np.array([1e12, 1e13, 1e14, 1e15])
    axes = [ax1, ax2, ax3, ax4]

    for idx, mass in enumerate(masses):
        # find closest matching halo
        idx_match = np.argmin(np.abs(m200 - mass))
        if m200[idx_match] < 1e13:
            prof = rho_w[idx_match]
            prof_f = prof_w[idx_match]
        else:
            prof = rho_h[idx_match]
            prof_fc = prof_c[idx_match]
            prof_fs = prof_s[idx_match]
            prof_f = prof_c[idx_match] + prof_s[idx_match]

            prof_fc[prof_fc == 0] = np.nan
            prof_fs[prof_fs == 0] = np.nan

        prof[prof == 0] = np.nan
        prof_f[prof_f == 0] = np.nan

        axes[idx].plot(r, (prof * r**2),
                       marker='o', lw=0, label='sim')
        if m200[idx_match] < 1e13:
            axes[idx].plot(r, (prof_f * r**2), label='warm')
        else:
            axes[idx].plot(r, (prof_fc * r**2), label='hot cen')
            axes[idx].plot(r, (prof_fs * r**2), label='hot sat')
            axes[idx].plot(r, (prof_f * r**2), label='hot')

        axes[idx].set_title(r'$m_{200\mathrm{m}} = 10^{%.2f}\mathrm{M_\odot}$'
                        %np.log10(m200[idx_match]))
        axes[idx].set_ylim([5e9, 2e13])
        axes[idx].legend(loc='best')
        if idx == 0:
            text = axes[idx].set_ylabel(r'$\rho(r) \cdot (r/r_{200\mathrm{m}})^2 \, [\mathrm{M_\odot/Mpc^3}]$')
            font_properties = text.get_fontproperties()
        # need to set visibility to False BEFORE log scaling
        if idx > 0:
            ticks = axes[idx].get_xticklabels()
            # strange way of getting correct label
            ticks[-7].set_visible(False)

        axes[idx].set_xscale('log')
        axes[idx].set_yscale('log')
        if idx > 0:
            axes[idx].yaxis.set_ticklabels([])

    fig.text(0.5, 0.03,
             r'$r/r_{200\mathrm{m}}$', ha='center',
             va='center', rotation='horizontal', fontproperties=font_properties)
    plt.show()

# ------------------------------------------------------------------------------
# End of compare_fit_gas_bahamas()
# ------------------------------------------------------------------------------

# def prof_nfw(x, c, m200, r200):
#     profile = (c*x)**(-1) * (1 + c*x)**(-3)
#     mass = tools.m_h(profile, x * r200)
#     profile *= m200/mass

#     return profile

def prof_dm_inner(r, sl, ri, b, m_sl):
    '''
    Profile for inner dark matter bump

    r in physical coordinates, sl denotes r/r200 <= 0.05
    '''
    profile = np.exp(-(np.log10(r) - np.log10(ri))**2/b)
    mass = tools.m_h(profile[sl], r[sl])
    profile *= m_sl/mass
    # y = r / ri
    # profile  = (y)**(-1) * (1 + y**2)**(-1)
    # mass = tools.m_h(profile[sl], r[sl])
    # profile *= m_sl / mass

    return profile


def prof_nfw_nosl(r, rs, m):
    '''
    Normal NFW profile

    r in physical coordinates, sl denotes r/r200 > 0.05
    '''
    x = r / rs
    profile  = (x)**(-1) * (1 + x)**(-2)
    mass = tools.m_h(profile, r)
    profile *= m/mass

    return profile

def prof_nfw(r, sl, rs, m_sl):
    '''
    Normal NFW profile

    r in physical coordinates, sl denotes r/r200 > 0.05
    '''
    x = r / rs
    profile  = (x)**(-1) * (1 + x)**(-2)
    mass = tools.m_h(profile[sl], r[sl])
    profile *= m_sl/mass

    return profile

def dm_plaw_fit(m, a, b):
    '''
    Fit to c
    '''
    # return a + b * np.log10(m)*(1 + c * np.log10(m)**2)
    return a * (m/1e14)**b

# def dm_c_fit(m):
#     A = 8.7449969011763216
#     B = -0.093399926987858539
#     plaw =  A * (m/1e14)**B

#     return plaw

def dm_c_fit(m, a, b):
    A = 8.7449969011763216
    B = -0.093399926987858539
    mod = (1 + a * np.sin(2*np.pi/3 * np.log10((m-1e11)/b)) * np.exp(-m/1e13))
    plaw =  A * (m/1e14)**B

    return plaw * mod

def dm_c_dmo(m):
    A = 8.7449969011763216
    B = -0.093399926987858539
    plaw =  A * (m/1e14)**B
    return plaw

def dm_m1_fit(m, a, b, mc):
    # yc = b
    # gamma = c * 2 - yc
    # beta = yc - gamma
    # return beta / (1 + (1e11/m)**a) + gamma
    return a * (0.5 + 0.5 * np.tanh(-b * np.log10(m/mc)))

def dm_m2_fit(m, a, b, c):
    # yc = b
    # gamma = c * 2 - yc
    # beta = yc - gamma
    # return beta / (1 + (1e11/m)**a) + gamma
    # yc = b
    # gamma = c * 2 - yc
    # beta = yc - gamma
    # return beta / (1 + (1e11/m)**a) + gamma
    return (c - b) / (1 + (1e11/m)**a) + b

def dm_f_fit(m, a, b, c):
    return a / (1 + b * np.log10(m/c)**2) + p.prms.f_dm

def fit_dm_bahamas_all():
    '''
    Fit the concentration for the BAHAMAS simulations
    '''
    # profiles = h5py.File('halo/data/BAHAMAS/DMONLY/eagle_subfind_particles_032_profiles_200_mean.hdf5', 'r')
    profiles = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles.hdf5', 'r')
    rho = profiles['PartType1/Densities'][:] / 0.7**2
    m200 = profiles['PartType1/M200'][:] * 0.7
    r200 = profiles['PartType1/R200'][:] * 0.7
    r_range = tools.bins2center(profiles['RBins_R_Mean200'][:])
    numpart = profiles['PartType1/NumPartGroup'][:]
    grnr = profiles['PartType1/GroupNumber'][:].astype(int)
    relaxed = profiles['PartType1/Relaxed'][:]
    profiles.close()

    sl = ((numpart[grnr] > 1e4) & relaxed)
    # sl = ((numpart[grnr] > 1e2) & relaxed)
    # sl = ((numpart[grnr] > 1e4))

    rho = rho[sl]
    m200 = m200[sl]
    r200 = r200[sl] * cm2mpc

    rs = -1 * np.ones_like(m200)
    rs_err = -1 * np.ones_like(m200)
    c = -1 * np.ones_like(m200)
    c_err = -1 * np.ones_like(m200)
    # dc = -1 * np.ones_like(m200)
    # dc_err = -1 * np.ones_like(m200)
    masses = -1 * np.ones_like(m200)

    pl.set_style('line')
    for idx, prof in enumerate(rho):
        # sl = (prof > 0)
        sl = ((prof > 0) & (r_range > 0.05))
        mass = tools.m_h(prof[sl], r200[idx] * r_range[sl])

        try:
            popt, pcov = opt.curve_fit(lambda r_range, rs: \
                                       np.log10(prof_nfw_nosl(r_range, rs, mass)),
                                       r_range[sl]*r200[idx],
                                       np.log10(prof[sl]), bounds=([0],[100]))
            masses[idx] = mass
            rs[idx] = popt[0]
            # dc[idx] = popt[1]
            rs_err[idx] = np.sqrt(np.diag(pcov))[0]
            # dc_err[idx] = np.sqrt(np.diag(pcov))[1]

            c[idx] = r200[idx] / rs[idx]

            # if idx%100 == 0:
            #     profile = prof_nfw_nosl(r_range[sl]*r200[idx], rs[idx], mass)
            #     profile_cor = prof_nfw_nosl(r_range[sl]*r200[idx], rs_cor, mass)
            #     print mass
            #     print tools.m_h(profile, r_range[sl] * r200[idx])
            #     print m200[idx]
            #     print '---'

            #     plt.clf()
            #     plt.plot(r_range[sl], prof[sl], marker='o', lw=0, label='prof')
            #     plt.plot(r_range[sl], profile, label='fit')
            #     plt.title('m=%.5e'%m200[idx])
            #     plt.xscale('log')
            #     plt.yscale('log')
            #     plt.legend()
            #     plt.savefig('prof_nfw_%i_norel_ngt1e3.pdf'%idx)
        except:
            continue

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

    return rs, rs_err, c, masses, m200#, r_prms, d_prms, f_prms

# ------------------------------------------------------------------------------
# End of fit_dm_bahamas_all()
# ------------------------------------------------------------------------------

def fit_dm_bahamas_bar():
    '''
    Fit the concentration for the BAHAMAS simulations
    '''
    # profiles = h5py.File('halo/data/BAHAMAS/DMONLY/eagle_subfind_particles_032_profiles_binned_200_mean_M13_15.hdf5', 'r')
    profiles = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_binned_200_mean_M11_15p5.hdf5', 'r')

    rho = profiles['PartType1/MedianDensity'][8:]
    m200 = profiles['PartType1/MedianM200'][8:]
    r200 = profiles['PartType1/MedianR200'][8:]
    r_range = tools.bins2center(profiles['RBins_R_Mean200'][:])
    profiles.close()


    rs = -1 * np.ones_like(m200)
    rs_err = -1 * np.ones_like(m200)
    ri = -1 * np.ones_like(m200)
    ri_err = -1 * np.ones_like(m200)
    b = -1 * np.ones_like(m200)
    b_err = -1 * np.ones_like(m200)
    m1 = -1 * np.ones_like(m200)
    m2 = -1 * np.ones_like(m200)
    m = -1 * np.ones_like(m200)
    mp = -1 * np.ones_like(m200)
    pl.set_style('line')

    for idx, prof in enumerate(rho):
        mass = tools.m_h(prof, r200[idx] * r_range)
        sl2 = ((prof > 0) & (r_range > 0.05))
        mass2 = tools.m_h(prof[sl2], r200[idx] * r_range[sl2])

        # Normal NFW fit
        sl = np.ones(sl2.sum(), dtype=bool)
        popt2, pcov2 = opt.curve_fit(lambda r_range, rs: \
                                     np.log10(prof_nfw(r_range, sl, rs, mass2)),
                                     r_range[sl2]*r200[idx],
                                     np.log10(prof[sl2]), bounds=([0],[1]))

        # Inner baryon bump fit
        nfw = prof_nfw(r_range*r200[idx], sl2, popt2[0], mass2)
        diff = prof - nfw
        sl1 = ((diff > 0) & (r_range <= 0.05))

        mass1 = tools.m_h(diff[sl1], r200[idx] * r_range[sl1])
        sl = np.ones(sl1.sum(), dtype=bool)
        popt1, pcov1 = opt.curve_fit(lambda r_range, ri, b: \
                                     np.log10(prof_dm_inner(r_range, sl, ri,
                                                            b, mass1)),
                                     r_range[sl1]*r200[idx],
                                     np.log10(prof[sl1]-nfw[sl1]),
                                     bounds=([0, 0],[0.1 * r200[idx], 10]))

        ri[idx] = popt1[0]
        ri_err[idx] = np.sqrt(np.diag(pcov1))[0]
        b[idx] = popt1[1]
        b_err[idx] = np.sqrt(np.diag(pcov1))[1]
        rs[idx] = popt2[0]
        rs_err[idx] = np.sqrt(np.diag(pcov2))[0]

        nfw = prof_nfw(r_range*r200[idx], sl2, rs[idx], mass2)
        bar = prof_dm_inner(r_range*r200[idx], sl1, ri[idx], b[idx], mass1)
        # check total new mass in profile
        mass_new = tools.m_h(nfw+bar, r_range*r200[idx])
        m1[idx] = mass1
        m2[idx] = mass2
        mp[idx] = mass_new
        m[idx] = mass

        # print mass1 / mass
        # print mass2 / mass
        # print '-----------'
        # plt.plot(r_range[sl1], r_range[sl1]**2 * (prof[sl1] - nfw[sl1]))
        # plt.plot(r_range[sl1], r_range[sl1]**2 * bar[sl1])
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()

        # plt.plot(r_range, r_range**2 * prof, lw=0, marker='o', label=r'sim')
        # plt.plot(r_range, r_range**2 * (nfw + bar) * mass / mass_new,
        #          label=r'total fit')
        # plt.plot(r_range, r_range**2 * nfw)
        # plt.plot(r_range, r_range**2 * bar)
        # plt.ylim(ymin=1e11)
        # plt.xlabel(r'$r/r_{200\mathrm{m}}$')
        # plt.ylabel(r'$\rho_{\mathrm{DM}}(r) \cdot (r/r_{200\mathrm{m}})^2$')
        # plt.title(r'$M=10^{%.2f}\mathrm{M_\odot}$'%np.log10(m200[idx]))
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.legend(loc='best')
        # plt.show()

    riopt, ricov = opt.curve_fit(dm_plaw_fit, m200[ri>1e-4],
                                 (ri/r200)[ri>1e-4],
                                 sigma=ri_err[ri>1e-4])
    ri_prms = {"a": riopt[0],
               "b": riopt[1]}

    rsopt, rscov = opt.curve_fit(dm_plaw_fit, m200, rs,
                                 sigma=rs_err)
    rs_prms = {"a": rsopt[0],
               "b": rsopt[1]}

    bopt, bcov = opt.curve_fit(dm_plaw_fit, m200, b, sigma=b_err)
    b_prms = {"a": bopt[0],
              "b": bopt[1]}


    c = r200 / rs
    c_err = c**2 * rs_err

    copt, ccov = opt.curve_fit(dm_c_fit, m200, c,
                               bounds=([0, 1e10],
                                       [1, 1e15]))
    c_prms = {"a": copt[0],
              "b": copt[1]}
              # "mc": copt[3]}

    # plt.plot(m200, c)
    # m = np.logspace(11, 15, 100)
    # A = 8.7449969011763216
    # B = -0.093399926987858539
    # fit = dm_c_fit(m, **c_prms)
    # plaw = A * (m/1e14)**B
    # plt.plot(m, fit)
    # plt.plot(m, plaw)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()


    # mopt1, mcov1 = opt.curve_fit(dm_plaw_fit, m200[(1e13<m200) & (m200<1e15)],
    #                              (m1/m)[(1e13<m200) & (m200<1e15)])
    # m1_prms = {"a": mopt1[0],
    #            "b": mopt1[1]}
    mopt1, mcov1 = opt.curve_fit(dm_m1_fit, m200[m200 < 1e14],
                                 (m1/m)[m200 < 1e14],
                                 bounds=([0, 0, 1e10],
                                         [1, 10, 1e15]))
    m1_prms = {"a": mopt1[0],
               "b": mopt1[1],
               "mc": mopt1[2]}

    mopt2, mcov2 = opt.curve_fit(dm_m2_fit, m200, m2/m,
                                 bounds=([0, 0, 0],
                                         [1, 1, 1]))
    m2_prms = {"a": mopt2[0],
               "b": mopt2[1],
               "c": mopt2[2]}

    # fpopt, fpcov = opt.curve_fit(dm_f_fit, m200, m/mp)
    # fp_prms = {"a": fpopt[0],
    #            "b": fpopt[1]}

    fopt, fcov = opt.curve_fit(dm_f_fit, m200, m/m200,
                               bounds=([0, 0, 1e10],
                                       [1, 5, 1e15]))
    f_prms = {"a": fopt[0],
              "b": fopt[1],
              "c": fopt[2]}

    # # plt.errorbar(m200, rs/r200, yerr=rs_err/r200, marker='o')
    # plt.plot(m200, m1/m, marker='o', lw=0)
    # plt.plot(m200, dm_plaw_fit(m200, **m1_prms))
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()

    return rs, rs_err, c, c_err, ri, ri_err, b, b_err, m1, m2, m, m200, r200, ri_prms, b_prms, rs_prms, c_prms, m1_prms, m2_prms, f_prms

# ------------------------------------------------------------------------------
# End of fit_dm_bahamas_bar()
# ------------------------------------------------------------------------------

def fit_dm_bahamas_dmo():
    '''
    Fit the concentration for the BAHAMAS simulations
    '''
    profiles = h5py.File('halo/data/BAHAMAS/DMONLY/eagle_subfind_particles_032_profiles_binned.hdf5', 'r')

    # need to convert to hubble units
    rho = profiles['PartType1/MedianDensity'][:] / 0.7**2
    m200 = profiles['PartType1/MedianM200'][:] * 0.7
    r200 = profiles['PartType1/MedianR200'][:] * 0.7
    r_range = tools.bins2center(profiles['RBins_R_Mean200'][:])
    profiles.close()

    rs = -1 * np.ones_like(m200)
    c = -1 * np.ones_like(m200)
    rs_err = -1 * np.ones_like(m200)
    m = -1 * np.ones_like(m200)

    pl.set_style('line')

    for idx, prof in enumerate(rho):
        # sl = (prof > 0)
        sl = ((prof > 0) & (r_range > 0.05))
        mass = tools.m_h(prof[sl], r200[idx] * r_range[sl])

        popt, pcov = opt.curve_fit(lambda r_range, rs: \
                                   np.log10(prof_nfw_nosl(r_range, rs, mass)),
                                   r_range[sl]*r200[idx],
                                   np.log10(prof[sl]), bounds=([0],[100]))
        m[idx] = mass

        rs[idx] = popt[0]
        rs_err[idx] = np.sqrt(np.diag(pcov))[0]
        c[idx] = r200[idx] / rs[idx]

        # plt.plot(r_range, r_range**2 * prof)
        # plt.plot(r_range[sl], r_range[sl]**2 * prof_nfw_nosl(r_range[sl]*r200[idx],
        #                                                      rs[idx], mass))
        # plt.plot(r_range, r_range * prof)
        # plt.plot(r_range[sl], r_range[sl] * prof_nfw_nosl(r_range[sl]*r200[idx],
        #                                                   rs[idx], mass))
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()


    copt, ccov = opt.curve_fit(dm_plaw_fit, m200, c)
    c_prms = {"a": copt[0],
              "b": copt[1],}

    rsopt, rscov = opt.curve_fit(dm_plaw_fit, m200, rs)
    rs_prms = {"a": rsopt[0],
               "b": rsopt[1]}

    # dopt, dcov = opt.curve_fit(dm_dc_fit, m200, dc)
    # d_prms = {"a": dopt[0],
    #           "b": dopt[1]}


    mopt, mcov = opt.curve_fit(dm_plaw_fit, m200, m/m200)
    m_prms = {"a": mopt[0],
              "b": mopt[1]}

    # fopt, fcov = opt.curve_fit(dm_f_fit, m200, (m1 + m2)/m200)
    # f_prms = {"a": fopt[0],
    #           "b": fopt[1]}
    # print c_prms
    # plt.plot(m200, c)
    # plt.plot(m200, dm_c_fit(m200, **c_prms))
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()

    # return c, c_err, masses, m200, c_prms, m_prms
    return rs, rs_err, m, m200, rs_prms, c_prms, m_prms

# ------------------------------------------------------------------------------
# End of fit_dm_bahamas_dmo()
# ------------------------------------------------------------------------------

def fit_dm_bahamas():
    '''
    Fit the concentration for the BAHAMAS simulations
    '''
    # profiles = h5py.File('halo/data/BAHAMAS/DMONLY/eagle_subfind_particles_032_profiles_binned_200_mean_M13_15.hdf5', 'r')
    profiles = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_binned.hdf5', 'r')

    # need to convert to hubble units
    rho = profiles['PartType1/MedianDensity'][:] / 0.7**2
    m200 = profiles['PartType1/MedianM200'][:] * 0.7
    r200 = profiles['PartType1/MedianR200'][:] * 0.7
    r_range = tools.bins2center(profiles['RBins_R_Mean200'][:])
    profiles.close()

    rs = -1 * np.ones_like(m200)
    c = -1 * np.ones_like(m200)
    rs_err = -1 * np.ones_like(m200)
    m = -1 * np.ones_like(m200)

    pl.set_style('line')

    for idx, prof in enumerate(rho):
        # sl = (prof > 0)
        sl = ((prof > 0) & (r_range > 0.05))
        mass = tools.m_h(prof[sl], r200[idx] * r_range[sl])

        popt, pcov = opt.curve_fit(lambda r_range, rs: \
                                   np.log10(prof_nfw_nosl(r_range, rs, mass)),
                                   r_range[sl]*r200[idx],
                                   np.log10(prof[sl]), bounds=([0],[100]))
        m[idx] = mass

        rs[idx] = popt[0]
        rs_err[idx] = np.sqrt(np.diag(pcov))[0]
        c[idx] = r200[idx] / rs[idx]

        # plt.plot(r_range, r_range**2 * prof)
        # plt.plot(r_range[sl], r_range[sl]**2 * prof_nfw_nosl(r_range[sl]*r200[idx],
        #                                                      rs[idx], mass))
        # plt.plot(r_range, r_range * prof)
        # plt.plot(r_range[sl], r_range[sl] * prof_nfw_nosl(r_range[sl]*r200[idx],
        #                                                   rs[idx], mass))
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()


    copt, ccov = opt.curve_fit(dm_plaw_fit, m200, c)
    c_prms = {"a": copt[0],
              "b": copt[1],}

    rsopt, rscov = opt.curve_fit(dm_plaw_fit, m200, rs)
    rs_prms = {"a": rsopt[0],
               "b": rsopt[1]}

    # dopt, dcov = opt.curve_fit(dm_dc_fit, m200, dc)
    # d_prms = {"a": dopt[0],
    #           "b": dopt[1]}


    mopt, mcov = opt.curve_fit(dm_plaw_fit, m200, m/m200)
    m_prms = {"a": mopt[0],
              "b": mopt[1]}

    # fopt, fcov = opt.curve_fit(dm_f_fit, m200, (m1 + m2)/m200)
    # f_prms = {"a": fopt[0],
    #           "b": fopt[1]}
    # print c_prms
    # plt.plot(m200, c)
    # plt.plot(m200, dm_c_fit(m200, **c_prms))
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()

    # return c, c_err, masses, m200, c_prms, m_prms
    return rs, rs_err, m, m200, rs_prms, c_prms, m_prms

# ------------------------------------------------------------------------------
# End of fit_dm_bahamas()
# ------------------------------------------------------------------------------

def plot_dm_fit_bahamas_median():
    # # get concentration mass plane plot
    # rs, rs_err, c, masses, m200 = fit_dm_bahamas_all()

    # # bin relation
    # m_bins = np.logspace(np.log10(m200).min(), np.log10(m200).max(), 20)
    # m = tools.bins2center(m_bins)
    # m_bin_idx = np.digitize(m200, m_bins)
    # c_med = np.array([np.median(c[m_bin_idx == m_bin])
    #                             for m_bin in np.arange(1, len(m_bins))])
    # popt, pcov = opt.curve_fit(dm_plaw_fit, m[~np.isnan(c_med)],
    #                            c_med[~np.isnan(c_med)])
    # fit_prms = {'a': popt[0], 'b': popt[1]}

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # img = ax.hexbin(np.log10(m200), np.clip(np.log10(c), np.log10(c).min(), 1.8),
    #                 cmap='magma',
    #                 bins='log',
    #                 gridsize=40)
    # ax.plot(np.log10(m), np.log10(c_med), c='w', label=r'$c_{\mathrm{med}}$')
    # ax.plot(np.log10(m), np.log10(dm_plaw_fit(m, **fit_prms)), c='w',
    #         label=r'$c_{\mathrm{fit}}$')

    # # compare median binned
    # rs, rs_err, masses, m200, r_prms, c_prms, f_prms = fit_dm_bahamas()
    # c_binned = dm_plaw_fit(m, **c_prms)
    # ax.plot(np.log10(m), np.log10(c_binned), c='w',
    #         label=r'$c_{\mathrm{bin}}$')
    # ax.set_xlabel(r'$m_{200\mathrm{m}} \, [\log_{10}\mathrm{M}_\odot]$')
    # ax.set_ylabel(r'$\log_{10} c(m)$')
    # # ax.set_title(r'NFW concentration')

    # for line in ax.xaxis.get_ticklines():
    #     line.set_color('w')
    # for line in ax.yaxis.get_ticklines():
    #     line.set_color('w')

    # leg = ax.legend(loc='best')
    # for text in leg.get_texts():
    #     plt.setp(text, color='w')

    # cb = fig.colorbar(img)
    # cb.set_label(r'$\log_{10} N_{\mathrm{bin}}$', rotation=270, labelpad=25)

    # plt.show()

    pl.set_style()

    # get other fit parameters
    rs, rs_err, c, c_err, ri, ri_err, b, b_err, m1, m2, m, m200, r200,  ri_prms, b_prms, rs_prms, c_prms, m1_prms, m2_prms, f_prms = fit_dm_bahamas_bar()

    fig = plt.figure(figsize=(14,6))
    ax = fig.add_subplot(131)
    axb = fig.add_subplot(132)
    axm = fig.add_subplot(133)
    ax.set_prop_cycle(pl.cycle_mark())
    ax.errorbar(m200, c, yerr=c_err,
                 marker='o', label=r'$c=r_{200\mathrm{m}}/r_s$')
    ax.set_prop_cycle(pl.cycle_line())

    c_fit = dm_c_fit(m200, **c_prms)
    c_fit[m200 < 2e11] = dm_c_dmo(m200[m200<2e11])
    ax.plot(m200, c_fit)
    ax.set_xlabel(r'$m_{200\mathrm{m}} \, [\mathrm{M}_\odot]$')
    ax.legend(loc='best')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('NFW profile')

    axb.set_prop_cycle(pl.cycle_mark())
    axb.errorbar(m200, ri/r200, yerr=ri_err/r200,
                 marker='o', label=r'$r_i/r_{200\mathrm{m}}$') #\, [\mathrm{Mpc}/h]$')
    axb.plot(m200, m1/m, marker='x', label=r'$m_{i}/m_{\mathrm{DM}}$')
    axb.errorbar(m200, b, yerr=b_err, marker='+', label=r'$\beta$')
    axb.set_prop_cycle(pl.cycle_line())
    axb.plot(m200, dm_plaw_fit(m200, **ri_prms))
    axb.plot(m200, dm_m1_fit(m200, **m1_prms))
    axb.plot(m200, dm_plaw_fit(m200, **b_prms))
    axb.set_ylim([1e-4,10])
    axb.set_xlabel(r'$m_{200\mathrm{m}} \, [\mathrm{M}_\odot]$')
    axb.legend(loc='best')
    axb.set_xscale('log')
    axb.set_yscale('log')
    axb.set_title('Baryon modification')

    axm.set_prop_cycle(pl.cycle_mark())
    axm.plot(m200, m/m200, marker='o', label=r'$m_{\mathrm{DM}}/m_{200\mathrm{m}}$')
    axm.plot(m200, m2/m, marker='+',
             label=r'$m_{\mathrm{NFW}}/m_{\mathrm{DM}}$')
    axm.set_prop_cycle(pl.cycle_line())
    axm.plot(m200, dm_f_fit(m200, **f_prms))
    axm.plot(m200, dm_m2_fit(m200, **m2_prms))
    axm.set_ylim([7e-1,1])
    axm.set_xlabel(r'$m_{200\mathrm{m}} \, [\mathrm{M}_\odot]$')
    axm.legend(loc='best')
    axm.set_xscale('log')
    # axm.set_yscale('log')
    axm.set_title('Mass contribution')

    plt.show()

# ------------------------------------------------------------------------------
# End of plot_dm_fit_bahamas_median()
# ------------------------------------------------------------------------------

def compare_fit_dm_bahamas():
    '''
    Compare how the fit performs for binned profiles
    '''
    binned = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_binned_200_mean_M11_15p5.hdf5', 'r')

    rho = binned['PartType1/MedianDensity'][:]
    q16 = binned['PartType1/Q16'][:]
    q84 = binned['PartType1/Q84'][:]
    r200 = binned['PartType1/MedianR200'][:]
    m200 = binned['PartType1/MedianM200'][:]

    r_bins = binned['RBins_R_Mean200'][:]
    r = tools.bins2center(r_bins).reshape(-1)
    m_bins = binned['MBins_M_Mean200'][:]

    binned.close()

    rsf, rsf_err, c, c_err, rif, rif_err, bi, bi_err, m1, m2, mf, m200f, r200f, ri_prms, bi_prms, rs_prms, c_prms, m1_prms, m2_prms , f_prms = fit_dm_bahamas_bar()

    rs = dm_plaw_fit(m200, **rs_prms)
    ri = dm_plaw_fit(m200, **ri_prms) * r200
    bi = dm_plaw_fit(m200, **bi_prms)
    m = m200 * dm_f_fit(m200, **f_prms)
    m1 = m * dm_m1_fit(m200, **m1_prms)
    m2 = m * dm_m2_fit(m200, **m2_prms)
    sl1 = (r <= 0.05)
    sl2 = (r > 0.05)

    pl.set_style()
    fig = plt.figure(figsize=(20,6))
    ax1 = fig.add_axes([0.1, 0.1, 0.2, 0.8])
    ax2 = fig.add_axes([0.3, 0.1, 0.2, 0.8])
    ax3 = fig.add_axes([0.5, 0.1, 0.2, 0.8])
    ax4 = fig.add_axes([0.7, 0.1, 0.2, 0.8])

    masses = np.array([1e12, 1e13, 1e14, 1e15])
    axes = [ax1, ax2, ax3, ax4]

    for idx, mass in enumerate(masses):
        # find closest matching halo
        idx_match = np.argmin(np.abs(m200 - mass))
        prof = rho[idx_match]
        m_prof = tools.m_h(prof, r * r200[idx_match])

        prof1 = prof_dm_inner(r*r200[idx_match], sl1, ri[idx_match],
                              bi[idx_match], m1[idx_match])
        prof2 = prof_nfw(r*r200[idx_match], sl2, rs[idx_match], m2[idx_match])
        prof3 = prof1 + prof2
        mass3 = tools.m_h(prof3, r * r200[idx_match])
        scale = m_prof / mass3

        axes[idx].plot(r, (prof * r**2),
                       marker='o', lw=0, label='sim')
        axes[idx].plot(r, (prof1 * r**2) * scale, label=r'$\rho_i$')
        axes[idx].plot(r, (prof2 * r**2) * scale, label=r'$\rho_{\mathrm{NFW}}$')
        axes[idx].plot(r, (prof3 * r**2) * scale, label='total')
        axes[idx].set_title(r'$m_{200\mathrm{m}} = 10^{%.2f}\mathrm{M_\odot}$'
                        %np.log10(m200[idx_match]))
        axes[idx].set_ylim([4e11, 2e13])
        if idx == 0:
            text = axes[idx].set_ylabel(r'$\rho(r) \cdot (r/r_{200\mathrm{m}})^2 \, [\mathrm{M_\odot/Mpc^3}]$')
            font_properties = text.get_fontproperties()
        # need to set visibility to False BEFORE log scaling
        if idx > 0:
            ticks = axes[idx].get_xticklabels()
            # strange way of getting correct label
            ticks[-7].set_visible(False)

        axes[idx].set_xscale('log')
        axes[idx].set_yscale('log')
        if idx > 0:
            axes[idx].yaxis.set_ticklabels([])

    fig.text(0.5, 0.03,
             r'$r/r_{200\mathrm{m}}$', ha='center',
             va='center', rotation='horizontal', fontproperties=font_properties)
    ax1.legend(loc='best')
    plt.show()

# ------------------------------------------------------------------------------
# End of compare_fit_dm_bahamas()
# ------------------------------------------------------------------------------

def mass_fraction():
    profiles = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_200_mean.hdf5', 'r')

    r = tools.bins2center(profiles['RBins_R_Mean200'][:])
    # Get mass in all particle types
    rho0 = profiles['PartType0/Densities'][:]
    rho1 = profiles['PartType1/Densities'][:]
    rho4 = profiles['PartType4/Densities'][:]
    r200_0 = profiles['PartType0/R200'][:] * cm2mpc
    r200_1 = profiles['PartType1/R200'][:] * cm2mpc
    r200_4 = profiles['PartType4/R200'][:] * cm2mpc

    m_0 = tools.m_h(rho0, r.reshape(1,-1) * r200_0.reshape(-1,1))
    m_1 = tools.m_h(rho1, r.reshape(1,-1) * r200_1.reshape(-1,1))
    m_4 = tools.m_h(rho4, r.reshape(1,-1) * r200_4.reshape(-1,1))

    # Find matching groups between different particle types
    grnr0 = profiles['PartType0/GroupNumber'][:]
    grnr1 = profiles['PartType1/GroupNumber'][:]
    grnr4 = profiles['PartType4/GroupNumber'][:]
    m200_4 = profiles['PartType4/M200'][:]

    # stars have least amount of groups
    idx_matched0 = np.searchsorted(grnr0, grnr4, 'left')
    # idx_matched1 = np.searchsorted(grnr1, grnr4, 'left')
    idx_found0 = (grnr0[idx_matched0] == grnr4)
    # idx_found1 = (grnr1[idx_matched1] == grnr4)

    # less matches in gas than in dark matter
    idx0 = idx_matched0[idx_found0]
    idx1 = np.in1d(grnr1, grnr0[idx0],
                   assume_unique=True)
    idx4 = np.in1d(grnr4, grnr0[idx0], assume_unique=True)

    f_0 = m_0[idx0] / m200_4[idx4]
    f_1 = m_1[idx1] / m200_4[idx4]
    f_4 = m_4[idx4] / m200_4[idx4]

    f_tot = f_0 + f_1 + f_4

    m200 = m200_4[idx4]

    pl.set_style()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # bin relation
    m_bins = np.logspace(np.log10(m200).min(), np.log10(m200).max(), 20)
    m = tools.bins2center(m_bins)
    m_bin_idx = np.digitize(m200, m_bins)

    f_med = np.array([np.median(f_tot[m_bin_idx == m_bin])
                      for m_bin in np.arange(1, len(m_bins))])
    img = ax.hexbin(np.log10(m200), f_tot,
                    cmap='magma',
                    bins='log',
                    gridsize=40)
    ax.plot(np.log10(m), f_med, c='w', label=r'$f_{\mathrm{med}}$')
    # ax.plot(np.log10(m), np.log10(dm_c_fit(m, **fit_prms)), c='w',
    #         label=r'$c_{\mathrm{fit}}$')

    ax.set_xlabel(r'$m_{200\mathrm{m}} \, [\log_{10}\mathrm{M}_\odot]$')
    ax.set_ylabel(r'$f(m)$')
    ax.set_title(r'Mass fraction total')

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
# End of mass_fraction()
# ------------------------------------------------------------------------------

def fit_hot_gas_profiles_all():
    '''
    Fit the hot gas profiles in BAHAMAS with beta profiles and return the fit
    parameters in such a way that bahamas_obs_fit can easily read them
    '''
    profiles = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_500_crit_Tgt1e6_5rf500c.hdf5', 'r')

    # we need rho in terms of density * h**2
    rho = profiles['PartType0/Densities'][:] / 0.7**2
    # r500 is in Mpc but was normalized with r/h_70
    # we need r in terms of Mpc/h for halo model -> need to ``unnormalise''
    # same for m500, in terms of mass / h
    m500 = profiles['PartType0/M500'][:] * 0.7
    r500 = profiles['PartType0/R500'][:] * 0.7

    r_range = tools.bins2center(profiles['RBins_R_Crit500'][:])
    numpart = profiles['PartType0/NumPartGroup'][:]
    grnr = profiles['PartType0/GroupNumber'][:].astype(int)

    sl = (numpart[grnr] > 1e4)

    rho = rho[sl]
    m500 = m500[sl]
    r500 = r500[sl] * cm2mpc

    # # Need to express bins in easy way to extract median m500 in each bin
    # nbin = profiles['PartType0/NumBin'][:]
    # bins_idx = np.cumsum(np.concatenate([[0],nbin], axis=0))
    # bins = np.concatenate([bins_idx[:-1].reshape(-1,1),
    #                        bins_idx[1:].reshape(-1,1)], axis=1)

    # # take median value as the true one
    # m500 = np.array([np.median(m500[b[0]:b[1]]) for b in bins])
    # r500 = np.array([np.median(r500[b[0]:b[1]]) for b in bins])

    r_bins = profiles['RBins_R_Crit500'][:]

    profiles.close()

    rx = 0.5 * (r_bins[:-1] + r_bins[1:])

    a = np.empty((0,), dtype=float)
    b = np.empty((0,), dtype=float)
    m_sl = np.empty((0,), dtype=float)
    aerr = np.empty((0,), dtype=float)
    berr = np.empty((0,), dtype=float)
    for idx, prof in enumerate(rho):
        sl = ((prof > 0) & (rx >= 0.15) & (rx <= 1.))
        sl_500 = ((prof > 0) & (rx <= 1.))

        # need at least more than 4 points to make a fit
        if sl.sum() <= 3:
            # Final fit will need to reproduce the m500gas mass
            a = np.append(a, np.nan)
            b = np.append(b, np.nan)
            m_sl = np.append(m_sl, 0.)
            aerr = np.append(aerr, np.nan)
            berr = np.append(berr, np.nan)

        else:
            r = rx

            # Determine different profile masses
            mass = tools.m_h(prof[sl], r[sl] * r500[idx])
            m500gas = tools.m_h(prof[sl_500], r[sl_500] * r500[idx])

            # Need to perform the fit for [0.15,1] r500c -> mass within this
            # region need to match
            sl_fit = np.ones(sl.sum(), dtype=bool)
            popt, pcov = opt.curve_fit(lambda r, a, b: \
                                       # , c:\
                                       prof_gas_hot(r, sl_fit, a, b, # , c,
                                                      mass, r500[idx]),
                                       r[sl], prof[sl], bounds=([0, 0],
                                                                [1, 5]))

            # plt.plot(r, prof, label='obs')
            # plt.plot(r, prof_gas_hot(r, sl_500, popt[0], popt[1],
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
            aerr = np.append(aerr, np.sqrt(np.diag(pcov))[0])
            berr = np.append(berr, np.sqrt(np.diag(pcov))[1])

    sl = ~np.isnan(a)

    return a[sl], aerr[sl], b[sl], berr[sl], m_sl[sl], r500[sl], m500[sl]

# ------------------------------------------------------------------------------
# End of fit_hot_gas_profiles_all()
# ------------------------------------------------------------------------------

def fit_hot_gas_profiles_like_obs():
    '''
    Fit the hot gas profiles in BAHAMAS with beta profiles and return the fit
    parameters in such a way that bahamas_obs_fit can easily read them
    '''
    f = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_binned_500_crit_Tgt1e6_5r500c.hdf5', 'r')

    # this is m200c
    m200_med = f['PartType0/MedianM200'][:] * 0.7

    # Need to express bins in easy way to extract median m500 in each bin
    nbin = f['PartType0/NumBin'][:]
    bins_idx = np.cumsum(np.concatenate([[0],nbin], axis=0))
    bins = np.concatenate([bins_idx[:-1].reshape(-1,1),
                           bins_idx[1:].reshape(-1,1)], axis=1)

    # r500 is in Mpc but was normalized with r/h_70
    # we need r in terms of Mpc/h for halo model -> need to ``unnormalise''
    # same for m500, in terms of mass / h
    m500 = f['PartType0/M500'][:] * 0.7
    r500 = f['PartType0/R500'][:] * 0.7
    # take median value as the true one
    m500 = np.array([np.median(m500[b[0]:b[1]]) for b in bins])
    r500 = np.array([np.median(r500[b[0]:b[1]]) for b in bins])

    r_bins = f['RBins_R_Crit500'][:]
    m_bins = f['MBins_M_Crit500'][:]

    # same for rho, we need it in terms of density * h**2
    rho = f['PartType0/MedianDensity'][:] / 0.7**2

    f.close()

    rx = 0.5 * (r_bins[:-1] + r_bins[1:])
    m = np.sum(m_bins, axis=1).reshape(-1)

    a = np.empty((0,), dtype=float)
    b = np.empty((0,), dtype=float)
    m_sl = np.empty((0,), dtype=float)
    aerr = np.empty((0,), dtype=float)
    berr = np.empty((0,), dtype=float)
    for idx, prof in enumerate(rho):
        sl = ((prof > 0) & (rx >= 0.15) & (rx <= 1.))
        sl_500 = ((prof > 0) & (rx <= 1.))

        # need at least more than 4 points to make a fit
        if sl.sum() <= 3:
            # Final fit will need to reproduce the m500gas mass
            a = np.append(a, np.nan)
            b = np.append(b, np.nan)
            m_sl = np.append(m_sl, 0.)
            aerr = np.append(aerr, np.nan)
            berr = np.append(berr, np.nan)

        else:
            r = rx

            # Determine different profile masses
            mass = tools.m_h(prof[sl], r[sl] * r500[idx])
            m500gas = tools.m_h(prof[sl_500], r[sl_500] * r500[idx])

            # Need to perform the fit for [0.15,1] r500c -> mass within this
            # region need to match
            sl_fit = np.ones(sl.sum(), dtype=bool)
            popt, pcov = opt.curve_fit(lambda r, a, b: \
                                       # , c:\
                                       prof_gas_hot(r, sl_fit, a, b, # , c,
                                                      mass, r500[idx]),
                                       r[sl], prof[sl], bounds=([0, 0],
                                                                [1, 5]))

            # plt.plot(r, prof, label='obs')
            # plt.plot(r, prof_gas_hot(r, sl_500, popt[0], popt[1],
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
            aerr = np.append(aerr, np.sqrt(np.diag(pcov))[0])
            berr = np.append(berr, np.sqrt(np.diag(pcov))[1])

    sl = ~np.isnan(a)

    return a[sl], aerr[sl], b[sl], berr[sl], m_sl[sl], r500[sl], m500[sl]

# ------------------------------------------------------------------------------
# End of fit_hot_gas_profiles_like_obs()
# ------------------------------------------------------------------------------

def fit_prms_sim():
    '''
    Return beta profile fit parameters
    '''
    a, aerr, b, berr, m_sl, r500, m500 = fit_hot_gas_profiles_all()

    sl = ((m500 > 1e13) & (m500 < 1e15))

    m500 = m500[sl]
    a = a[sl]
    b = b[sl]
    m_sl = m_sl[sl]
    r500 = r500[sl]

    # bin relation
    m_bins = np.logspace(np.log10(m500).min(), np.log10(m500).max(), 30)
    m = tools.bins2center(m_bins)
    m_bin_idx = np.digitize(m500, m_bins)

    # median rc
    a_med = np.array([np.median(a[m_bin_idx == m_bin])
                      for m_bin in np.arange(1, len(m_bins))])

    b_med = np.array([np.median(b[m_bin_idx == m_bin])
                      for m_bin in np.arange(1, len(m_bins))])

    aopt, acov = opt.curve_fit(dm_plaw_fit, m, a_med)

    bopt, bcov = opt.curve_fit(dm_plaw_fit, m, b_med)


    rc_prms = {'a': aopt[0],
               'b': aopt[1]}

    b_prms = {'a': bopt[0],
              'b': bopt[1]}

    return rc_prms, b_prms

# ------------------------------------------------------------------------------
# End of fit_prms_sim()
# ------------------------------------------------------------------------------


def fit_prms(x=500, m_cut=1e13, prms=p.prms):
    '''
    Return beta profile fit parameters

    Parameters
    ----------
    x : int
      Overdensity threshold to compare rc to
    m_cut : float
      Mass cut for which to compute the median fit parameter
    prms : halo.Parameters object
      Contains all cosmological and halo model information

    Returns
    -------
    rc : float
      Median rc fit parameter
    beta : float
      Median beta fit parameter
    '''
    a, aerr, b, berr, m_sl, r500, m500 = fit_hot_gas_profiles_like_obs()

    if x == 500:
        # ! WATCH OUT ! This rc is rc / r500c, whereas in the paper we plot
        # rc / r200m
        rc = np.median(a[m500 > m_cut])
        beta = np.median(b[m500 > m_cut])

    elif x == 200:
        m200 = np.array([tools.m500c_to_m200m(m, prms.rho_crit, prms.rho_m)
                         for m in m500[m500 > m_cut]])
        r200 = tools.mass_to_radius(m200, 200 * prms.rho_m)

        rc = np.median(a[m500 > m_cut] * r500[m500 > m_cut] / r200)
        beta = np.median(b[m500 > m_cut])

    return rc, beta

# ------------------------------------------------------------------------------
# End of fit_prms()
# ------------------------------------------------------------------------------

def f_gas(m, log10mc, a, prms):
    x = np.log10(m) - log10mc
    return (prms.omegab/prms.omegam) * (0.5 * (1 + np.tanh(x / a)))

def f_gas_prms(prms):
    '''
    Get the gas fractions from the beta profile fits to the hot gas in the
    simulations
    '''
    a, aerr, b, berr, m_sl, r500, m500 = fit_hot_gas_profiles_like_obs()

    f_med = m_sl / m500

    fmopt, fmcov = opt.curve_fit(lambda m, log10mc, a: f_gas(m, log10mc, a, prms),
                                 m500, f_med,
                                 bounds=([10, 0],
                                         [15, 10]))

    fm_prms = {"log10mc": fmopt[0],
               "a": fmopt[1]}

    return fm_prms

# ------------------------------------------------------------------------------
# End of f_gas_prms()
# ------------------------------------------------------------------------------

def gas_plaw(r, rx, a, rhox):
    '''
    Return a power law with index a and density rhox at radius rx
    '''
    return rhox * (r/rx)**a

def mass_diff_plaw(a, m, rx, ry, rhox):
    '''
    Return the mass of a power law with index a between rx and ry
    with density rhox at rx
    '''
    if a == -3:
        return np.abs(m - (4 * np.pi * (rhox / rx**a) * np.log(ry / rx)))

    else:
        return np.abs(m - 4 * np.pi * (rhox / rx**a) * 1. / (a + 3) *
                      (ry**(a+3) - rx**(a+3)))

def fit_plaw_index_mass(m, rx, ry, rhox):
    '''
    Returns the power law index for a power law density with rhox at rx
    containing a mass m between rx and ry
    '''
    x0 = [-4]
    res = opt.minimize(mass_diff_plaw, x0, args=(m, rx, ry, rhox),
                       bounds=[(-20,20)])

    return res.x[0]

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
    fm, f1, f2 = f_gas_prms(prms)
    m200_b = opt.brentq(massdiff_dmo2bar, m_dmo / 10., m_dmo,
                        args=(m_dmo, f_b, prms, fm))

    return m200_b

# ------------------------------------------------------------------------------
# End of m200dmo_to_m200b()
# ------------------------------------------------------------------------------

def m200b_to_m200dmo(m_b, f_gas, prms):
    '''
    Get the relation between the measured halo mass and the dark matter only
    equivalent mass according to the relation

        m_dmo(m_b) = m_b / ( 1 - (f_b - f_gas(m_b)) )
    '''
    f_b = 1 - prms.f_dm
    m200dmo = m_b / (1 - (f_b - f_gas))

    return m200dmo

# ------------------------------------------------------------------------------
# End of m200b_to_m200dmo()
# ------------------------------------------------------------------------------

def plot_hot_gas_profiles_paper(prms):
    '''
    Plot the hot gas profiles extracted from the BAHAMAS simulations
    '''
    f = h5py.File('halo/data/BAHAMAS/eagle_subfind_particles_032_profiles_binned_500_crit_Tgt1e6_5r500c.hdf5', 'r')

    m200_med = f['PartType0/MedianM200'][:]

    # Need to express bins in easy way to extract median m500 in each bin
    nbin = f['PartType0/NumBin'][:]
    bins_idx = np.cumsum(np.concatenate([[0],nbin], axis=0))
    bins = np.concatenate([bins_idx[:-1].reshape(-1,1),
                           bins_idx[1:].reshape(-1,1)], axis=1)

    m500 = f['PartType0/M500'][:]
    m500_med = np.array([np.median(m500[b[0]:b[1]]) for b in bins])

    r_bins = f['RBins_R_Crit500'][:]
    m_bins = f['MBins_M_Crit500'][:]

    dens = f['PartType0/MedianDensity'][:] / prms.rho_crit
    dens[dens == 0] = np.nan

    f.close()

    r = 0.5 * (r_bins[:-1] + r_bins[1:])
    m = np.sum(m_bins, axis=1).reshape(-1)

    cmap = mpl.cm.get_cmap('magma', m.shape[0])
    lines = LineCollection([list(zip(r, d)) for d in dens],
                           cmap=cmap)

    pl.set_style('line')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(r.min(), r.max())
    ax.set_ylim(np.nanmin(dens[np.nonzero(dens)]),
                np.nanmax(dens))

    lines.set_array(np.log10(m))
    ax.add_collection(lines)
    axcb = fig.colorbar(lines)

    axcb.set_label(r'$\log_{10}(m_\mathrm{500c}) \, [\mathrm{M_\odot}/h]$',
                   rotation=270, labelpad=30)
    ax.set_xlabel(r'$r/r_\mathrm{500c}$')
    ax.set_ylabel(r'$\rho(r)/\rho_\mathrm{crit}$')

    plt.savefig('bahamas_hot_gas_profiles.pdf')

# ------------------------------------------------------------------------------
# End of plot_hot_gas_profiles_paper()
# ------------------------------------------------------------------------------

