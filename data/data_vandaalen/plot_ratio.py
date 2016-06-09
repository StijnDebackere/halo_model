import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
from cycler import cycler
import palettable

import numpy as np
import scipy.interpolate as interp

def set_style():
    cycle_color = palettable.colorbrewer.qualitative.Paired_12.mpl_colors
    cycle_ls = ['-', '--', ':', '-.']
    cycle_lw = [2, 3, 4]

    cycle_tot = (cycler('color', cycle_color) +
                 (cycler('lw', cycle_lw) *
                  cycler('ls', cycle_ls)))

    # plot line settings
    mpl.rcParams['axes.prop_cycle'] = cycle_tot
    mpl.rcParams['errorbar.capsize'] = 0 # -> not valid

    # text settings
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['font.family'] = 'serif'
    # mpl.rcParams['font.serif'] = 'Times'
    mpl.rcParams['text.usetex'] = True

    mpl.rcParams['axes.labelsize'] = 'large'
    mpl.rcParams['axes.titlesize'] = 'x-large'
    # mpl.rcParams['axes.titleweight'] = 'bold'
    mpl.rcParams['figure.titlesize'] = 'x-large'
    mpl.rcParams['xtick.labelsize'] = 'large'
    mpl.rcParams['ytick.labelsize'] = 'large'

    # tick settings
    mpl.rcParams['xtick.major.size'] = 6
    mpl.rcParams['xtick.minor.size'] = 4
    mpl.rcParams['ytick.major.size'] = 6
    mpl.rcParams['ytick.minor.size'] = 4

    # legend settings
    mpl.rcParams['legend.frameon'] = False
    mpl.rcParams['legend.framealpha'] = 0
    mpl.rcParams['legend.scatterpoints'] = 1

    # general settings
    mpl.rcParams['savefig.transparent'] = True

    return

# ------------------------------------------------------------------------------
# End of set_style()
# ------------------------------------------------------------------------------

def plot_ratio(file1='BAHAMAS/tables/POWMES_DMONLY_nu0_L400N1024_WMAP9_032_table.dat',
               file2='BAHAMAS/tables/POWMES_AGN_TUNED_nu0_L400N1024_WMAP9_032_table.dat'):

    set_style()
    k_dm, P_dm, D_dm = np.loadtxt(file1, unpack=True)
    k_ref, P_ref, D_ref = np.loadtxt(file2, unpack=True)
    # P_ref *= P_dm[0]/P_ref[0]
    # D_ref *= D_dm[0]/D_ref[0]

    plt.clf()
    fig = plt.figure()
    ax_P = fig.add_axes([0.1, 0.35, 0.8, 0.55])
    ax_r = fig.add_axes([0.1, 0.1, 0.8, 0.25])

    ax_P.plot(k_dm, D_dm, label=r'DMONLY')
    ax_P.plot(k_ref, D_ref, label=r'AGN')
    # yticklabs = ax_P.get_yticklabels()
    # yticklabs[0] = ""
    # ax_P.set_yticklabels(yticklabs)
    ax_P.set_ylim(ymin=2e-3)
    ax_P.set_xlim([1e-2,1e2])
    ax_P.axes.set_xscale('log')
    ax_P.axes.set_yscale('log')
    ax_P.set_ylabel(r'$\Delta^2(k)$')
    ax_P.set_title(r'Power spectra for BAHAMAS')
    ax_P.set_xticklabels([])
    ax_P.legend(loc='best')

    # diff = P_i(k_dm) / P_dm - 1
    diff = P_ref / P_dm - 1
    diff_gt = diff * 1.
    diff_gt[diff < 0] = np.nan
    diff_lt = diff * 1.
    diff_lt[diff >= 0] = np.nan

    ax_r.plot(k_dm, diff_gt,
              label=r'$P_{\mathrm{AGN}} \geq P_{\mathrm{DM}}$')
    ax_r.plot(k_dm, np.abs(diff_lt),
              label=r'$P_{\mathrm{AGN}} < P_{\mathrm{DM}}$')    
    ax_r.grid()
    ax_r.set_xlim([1e-2,1e2])
    ax_r.set_ylim([1e-3,1])
    ax_r.axes.set_xscale('log')
    ax_r.axes.set_yscale('log')
    ax_r.legend(loc='best')
    ax_r.set_xlabel(r'$k$ in $h/$Mpc')
    ax_r.set_ylabel(r'$\frac{P_{\mathrm{AGN}} - P_{\mathrm{DM}}}{P_{\mathrm{DM}}}$',
                    labelpad=-2)

    plt.savefig('ratio.pdf', dpi=900)
    # plt.close(fig)
