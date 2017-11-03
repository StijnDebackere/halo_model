import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
from cycler import cycler
import palettable

import numpy as np
import scipy.interpolate as interp

def set_style(cycle='line'):
    cycle_options = ['line', 'mark']
    if cycle not in cycle_options:
        raise ValueError('cycle needs to be in %s'%cycle_options)
    if cycle == 'line':
        cycle_prop = cycle_line()
    elif cycle == 'mark':
        cycle_prop = cycle_mark()

    mpl.rcParams['axes.prop_cycle'] = cycle_prop
    mpl.rcParams['errorbar.capsize'] = 0 # -> not valid
    # text settings
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['font.size'] = 16
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = 'Times New Roman' # does not look like TNR
    # no type 3 fonts -> http://goo.gl/MppxTc
    # mpl.rcParams['ps.useafm'] = True
    # mpl.rcParams['pdf.use14corefonts'] = True

    mpl.rcParams['axes.labelsize'] = 'x-large'
    mpl.rcParams['axes.labelweight'] = 'normal'
    mpl.rcParams['axes.titlesize'] = 'x-large'
    mpl.rcParams['axes.labelpad'] = 1.0 # default=5.0
    # mpl.rcParams['axes.titleweight'] = 'bold'
    mpl.rcParams['figure.titlesize'] = 'x-large'
    mpl.rcParams['xtick.labelsize'] = 'x-large'
    mpl.rcParams['ytick.labelsize'] = 'x-large'
    # mpl.rcParams['xtick.direction'] = 'out'
    # mpl.rcParams['ytick.direction'] = 'out'

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

def cycle_line():
    '''
    Set plot prop_cycle for line plots
    '''
    # cycle_color = palettable.colorbrewer.qualitative.Paired_12.mpl_colors
    # cycle_ls = ['-', '--', ':', '-.']
    # cycle_lw = [2, 3, 4]

    # cycle_tot = (cycler('color', cycle_color) +
    #              (cycler('lw', cycle_lw) *
    #               cycler('ls', cycle_ls)))

    cycle_color = palettable.colorbrewer.qualitative.Set1_5.mpl_colors

    cycle_tot = cycler(c=cycle_color, lw=[2,2,2,2,2])

    # plot line settings
    # mpl.rcParams['axes.prop_cycle'] = cycle_tot
    # mpl.rcParams.update({'axes.prop_cycle': cycle_tot})
    return cycle_tot

# ------------------------------------------------------------------------------
# End of cycle_line()
# ------------------------------------------------------------------------------

def cycle_mark():
    '''
    Set plot prop_cycle for plot with markers
    '''

    cycle_color = palettable.colorbrewer.qualitative.Paired_12.mpl_colors
    cycle_lw = [0] * 12
    cycle_marker = ['o', 'x', '+', 's', '*', '^', 'v', 'h', 'D', '<', '>', 'p']
    # cycle_mfill = [None, 'none']

    cycle_tot = (cycler('marker', cycle_marker) +
                  # cycler('markerfacecolor', cycle_mfill) +
                  (cycler('color', cycle_color) + cycler('lw', cycle_lw)))

    return cycle_tot

# ------------------------------------------------------------------------------
# End of cycle_mark()
# ------------------------------------------------------------------------------

def plot_ratio(file1='BAHAMAS/tables/POWMES_DMONLY_nu0_L400N1024_WMAP9_032_table.dat',
               file2='BAHAMAS/tables/POWMES_AGN_TUNED_nu0_L400N1024_WMAP9_032_table.dat'):

    set_style()
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


    # diff = P_i(k_dm) / P_dm - 1
    diff = P_ref / P_dm - 1
    diff_gt = diff * 1.
    diff_gt[diff < 0] = np.nan
    diff_lt = diff * 1.
    diff_lt[diff >= 0] = np.nan

    ax_r.plot(k_dm, diff_gt, c='k', ls='-',
              label=r'$P_{\mathrm{AGN}} \geq P_{\mathrm{DM}}$')
    ax_r.plot(k_dm, np.abs(diff_lt), c='k', ls='--',
              label=r'$P_{\mathrm{AGN}} < P_{\mathrm{DM}}$')
    ax_r.grid()
    ax_r.set_xlim([1e-2,1e2])
    ax_r.set_ylim([1e-3,1])
    ax_r.axes.set_xscale('log')
    ax_r.axes.set_yscale('log')
    ax_r.legend(loc='best')
    ax_r.set_xlabel(r'$k \, [h/\mathrm{Mpc}]$')
    ax_r.set_ylabel(r'$\frac{P_{\mathrm{AGN}} - P_{\mathrm{DM}}}{P_{\mathrm{DM}}}$',
                    labelpad=-2)

    plt.savefig('ratio.pdf', dpi=900, transparent=True)
    # plt.close(fig)
