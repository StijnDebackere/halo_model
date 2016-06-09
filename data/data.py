import numpy as np
import astropy.constants as const
import astropy.units as u

import halo.parameters as p
import halo.tools as tools

import pdb

ddir = '/Volumes/Data/stijn/Documents/Universiteit/MR/code/halo/data/'
prms = p.prms

def data_vanDaalen(k_range_lin):
    # n-body data
    # z, k, P, D = np.loadtxt('vanDaalen_data/powtable_DMONLY_WMAP7_all.dat',
    #                         comments='%',unpack=True)
    z, k, P, D = np.loadtxt('data_vandaalen/OWLS+COSMO-OWLS/powtable_AGN_all.dat',
                            comments='%',unpack=True)
    idx = (z == 0)

    f_D = intrp.interp1d(k[idx], D[idx])
    f_P = intrp.interp1d(k[idx], P[idx])

    matched_D = f_D(k_range_lin)
    matched_P = f_P(k_range_lin)

    return matched_D, matched_P

# ------------------------------------------------------------------------------
# End of data_vanDaalen()
# ------------------------------------------------------------------------------
