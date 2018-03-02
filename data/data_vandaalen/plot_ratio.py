import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
from cycler import cycler
import palettable
import sys

# allow import of plot
import plot as pl


import numpy as np
import scipy.interpolate as interp

def plot_ratio(file1='BAHAMAS/tables/POWMES_DMONLY_nu0_L400N1024_WMAP9_032_table.dat',
               file2='BAHAMAS/tables/POWMES_AGN_TUNED_nu0_L400N1024_WMAP9_032_table.dat'):

    pl.set_style('line')
    k_dm, P_dm, D_dm = np.loadtxt(file1, unpack=True)
    k_ref, P_ref, D_ref = np.loadtxt(file2, unpack=True)
    # P_ref *= P_dm[0]/P_ref[0]
    # D_ref *= D_dm[0]/D_ref[0]

    plt.clf()
    fig = plt.figure(figsize=(11, 8))
    ax_P = fig.add_axes([0.1, 0.35, 0.8, 0.55])
    ax_r = fig.add_axes([0.1, 0.1, 0.8, 0.25])

    ax_P.plot(k_dm, D_dm, label=r'DMONLY')
    ax_P.plot(k_ref, D_ref, label=r'AGN')

    axd = ax_P.twiny()
    l = 2 * np.pi / k_dm
    axd.plot(l, D_dm)
    axd.set_xlim(axd.get_xlim()[::-1])
    axd.cla()
    axd.set_xscale('log')
    axd.set_xlabel(r'$\lambda \, [\mathrm{Mpc}/h]$', labelpad=10)
    axd.tick_params(axis='x', pad=5)

    # yticklabs = ax_P.get_yticklabels()
    # yticklabs[0] = ""
    # ax_P.set_yticklabels(yticklabs)
    ax_P.set_ylim(ymin=2e-3)
    ax_P.set_xlim([1e-2,1e2])
    ax_P.axes.set_xscale('log')
    ax_P.axes.set_yscale('log')
    ax_P.set_ylabel(r'$\Delta^2(k)$')
    # ax_P.set_title(r'Power spectra for BAHAMAS')
    ax_P.set_xticklabels([])
    ax_P.legend(loc='best')


    # # diff = P_i(k_dm) / P_dm - 1
    # diff = P_ref / P_dm - 1
    # diff_gt = diff * 1.
    # diff_gt[diff < 0] = np.nan
    # diff_lt = diff * 1.
    # diff_lt[diff >= 0] = np.nan

    # ax_r.plot(k_dm, diff_gt, c='k', ls='-',
    #           label=r'$P_{\mathrm{AGN}} \geq P_{\mathrm{DM}}$')
    # ax_r.plot(k_dm, np.abs(diff_lt), c='k', ls='--',
    #           label=r'$P_{\mathrm{AGN}} < P_{\mathrm{DM}}$')
    ax_r.plot(k_dm, P_ref / P_dm)
    ax_r.axhline(y=1, c='k', ls='--')
    ax_r.grid()
    ax_r.set_xlim([1e-2,1e2])
    # ax_r.set_ylim([1e-3,1])
    ax_r.set_ylim([0.82,1.2])
    ax_r.axes.set_xscale('log')
    # ax_r.axes.set_yscale('log')
    ax_r.minorticks_on()
    ax_r.tick_params(axis='x',which='minor',bottom='off')

    ax_r.legend(loc='best')
    ax_r.set_xlabel(r'$k \, [h/\mathrm{Mpc}]$')
    # ax_r.set_ylabel(r'$\frac{P_{\mathrm{AGN}} - P_{\mathrm{DM}}}{P_{\mathrm{DM}}}$',
    #                 labelpad=-2)
    ax_r.set_ylabel(r'$P_\mathrm{AGN}/P_\mathrm{DM}$')

    plt.savefig('ratio.pdf', dpi=900, transparent=True)
    # plt.close(fig)
