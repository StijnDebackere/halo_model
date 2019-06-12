import numpy as np
import scipy.interpolate as interpolate
import asdf
from copy import copy

import pdb

if len(__file__.split("/")[:-1]) >= 1:
    table_dir = "/".join(__file__.split("/")[:-1]) + "/"
else:
    table_dir = "".join(__file__.split("/")[:-1])

table_dir += "tables/"



def arrays_to_coords(*xi):
    '''
    Convert a set of N 1-D coordinate arrays to a regular coordinate grid of 
    dimension (npoints, N) for the interpolator 
    '''
    # the meshgrid matches each of the *xi to all the other *xj
    Xi = np.meshgrid(*xi, indexing='ij')

    # now we create a column vector of all the coordinates
    coords = np.concatenate([X.reshape(X.shape + (1,)) for X in Xi], axis=-1)

    return coords.reshape(-1, len(xi))

# ------------------------------------------------------------------------------
# End of arrays_to_coords()
# ------------------------------------------------------------------------------

def arrays_to_ogrid(*xi):
    '''
    Return an ordered list of N arrays reshaped to an ogrid to match to the total
    dimension N with shape (n1, ..., nk, ..., nn)
    '''
    n = len(xi)

    # all arrays need to be reshaped to (1,...,-1,..,1)
    # where -1 corresponds to their own dimension
    shape = [1,] * n

    ogrid = []
    for i, x in enumerate(xi):
        s = copy(shape)
        s[i] = -1
        s = tuple(s)
        ogrid.append(np.reshape(x, s))

    return ogrid

# ------------------------------------------------------------------------------
# End of arrays_to_ogrid()
# ------------------------------------------------------------------------------

###################################
# Functions to interpolate tables #
###################################

def c200c_cosmo_interp(c_file=table_dir + "c200c_correa_cosmo.asdf"):
    '''
    Return the interpolator for the given file
    '''
    af = asdf.open(c_file)

    s8 = af.tree["sigma8"][:]
    om = af.tree["omegam"][:]
    ov = af.tree["omegav"][:]
    n = af.tree["n"][:]
    h = af.tree["h"][:]
    
    z = af.tree["z"][:]
    m = af.tree["m200c"][:]
    c = af.tree["c200c"][:]

    coords = (s8, om, ov, n, h, z, np.log10(m))
    c_interp = interpolate.RegularGridInterpolator(coords, c)

    return c_interp

# ------------------------------------------------------------------------------
# End of c200c_cosmo_interp()
# ------------------------------------------------------------------------------

def c200c_interp(c_file=table_dir + "c200c_correa.asdf"):
    '''
    Return the interpolator for the given file
    '''
    af = asdf.open(c_file)

    z = af.tree["z"][:]
    m = af.tree["m200c"][:]
    c = af.tree["c200c"][:]

    coords = (z, np.log10(m))

    c_interp = interpolate.RegularGridInterpolator(coords, c)

    return c_interp

# ------------------------------------------------------------------------------
# End of c200c_interp()
# ------------------------------------------------------------------------------

def c200m_cosmo_interp(c_file=table_dir + "c200m_correa_cosmo.asdf"):
    '''
    Return the interpolator for the given file
    '''
    af = asdf.open(c_file)

    s8 = af.tree["sigma8"][:]
    om = af.tree["omegam"][:]
    ov = af.tree["omegav"][:]
    n = af.tree["n"][:]
    h = af.tree["h"][:]
    
    z = af.tree["z"][:]
    m = af.tree["m200m"][:]
    c = af.tree["c200m"][:]

    coords = (s8, om, ov, n, h, z, np.log10(m))
    c_interp = interpolate.RegularGridInterpolator(coords, c)

    return c_interp

# ------------------------------------------------------------------------------
# End of c200m_cosmo_interp()
# ------------------------------------------------------------------------------

def c200m_interp(c_file=table_dir + "c200m_correa.asdf"):
    '''
    Return the interpolator for the c200m(m200m) relation
    '''
    af = asdf.open(c_file)

    z = af.tree["z"][:]
    m = af.tree["m200m"][:]
    c = af.tree["c200m"][:]

    coords = (z, np.log10(m))

    c_interp = interpolate.RegularGridInterpolator(coords, c)

    return c_interp
    
# ------------------------------------------------------------------------------
# End of c200m_interp()
# ------------------------------------------------------------------------------

def m200m_dmo_interp(f_c, sigma_lnc, m_file=None):
    '''
    Return the interpolator for the c200m(m200m) relation
    '''
    if m_file is None:
        m_file = table_dir + "m500c_to_m200m_dmo_fc_{}_slnc_{}.asdf".format(f_c,
                                                                            sigma_lnc)

    af = asdf.open(m_file)

    if f_c != af.tree["f_c"]:
        raise ValueError("f_c in {} is not {}".format(m_file, f_c))
    if sigma_lnc != af.tree["sigma_lnc"]:
        raise ValueError("sigma_lnc in {} is not {}".format(m_file, sigma_lnc))

    z = af.tree["z"][:]
    m = af.tree["m500c"][:]
    f = af.tree["fgas_500c"][:]
    m200m_dmo = af.tree["m200m_dmo"][:]

    coords = (z, np.log10(m), f)

    m200m_dmo_interp = interpolate.RegularGridInterpolator(coords, m200m_dmo)

    return m200m_dmo_interp
    
# ------------------------------------------------------------------------------
# End of m200m_dmo_interp()
# ------------------------------------------------------------------------------

def fcen500c_interp(f_c, sigma_lnc, m_file=None):
    '''
    Return the interpolator for the c200m(m200m) relation
    '''
    if m_file is None:
        m_file = table_dir + "m500c_to_m200m_dmo_fc_{}_slnc_{}.asdf".format(f_c,
                                                                            sigma_lnc)

    af = asdf.open(m_file)

    if f_c != af.tree["f_c"]:
        raise ValueError("f_c in {} is not {}".format(m_file, f_c))
    if sigma_lnc != af.tree["sigma_lnc"]:
        raise ValueError("sigma_lnc in {} is not {}".format(m_file, sigma_lnc))

    z = af.tree["z"][:]
    m = af.tree["m500c"][:]
    f = af.tree["fgas_500c"][:]
    fcen_500c = af.tree["fcen_500c"][:]

    coords = (z, np.log10(m), f)

    fcen_500c_interp = interpolate.RegularGridInterpolator(coords, fcen_500c)

    return fcen_500c_interp
    
# ------------------------------------------------------------------------------
# End of fcen_500c_interp()
# ------------------------------------------------------------------------------

def fsat500c_interp(f_c, sigma_lnc, m_file=None):
    '''
    Return the interpolator for the c200m(m200m) relation
    '''
    if m_file is None:
        m_file = table_dir + "m500c_to_m200m_dmo_fc_{}_slnc_{}.asdf".format(f_c,
                                                                            sigma_lnc)

    af = asdf.open(m_file)

    if f_c != af.tree["f_c"]:
        raise ValueError("f_c in {} is not {}".format(m_file, f_c))
    if sigma_lnc != af.tree["sigma_lnc"]:
        raise ValueError("sigma_lnc in {} is not {}".format(m_file, sigma_lnc))

    z = af.tree["z"][:]
    m = af.tree["m500c"][:]
    f = af.tree["fgas_500c"][:]
    fsat_500c = af.tree["fsat_500c"][:]

    coords = (z, np.log10(m), f)

    fsat_500c_interp = interpolate.RegularGridInterpolator(coords, fsat_500c)

    return fsat_500c_interp
    
# ------------------------------------------------------------------------------
# End of fsat_500c_interp()
# ------------------------------------------------------------------------------

def fbar500c_interp(f_c, sigma_lnc, m_file=None):
    '''
    Return the interpolator for the c200m(m200m) relation
    '''
    if m_file is None:
        m_file = table_dir + "m500c_to_m200m_dmo_fc_{}_slnc_{}.asdf".format(f_c,
                                                                            sigma_lnc)

    af = asdf.open(m_file)

    if f_c != af.tree["f_c"]:
        raise ValueError("f_c in {} is not {}".format(m_file, f_c))
    if sigma_lnc != af.tree["sigma_lnc"]:
        raise ValueError("sigma_lnc in {} is not {}".format(m_file, sigma_lnc))

    z = af.tree["z"][:]
    m = af.tree["m500c"][:]
    f = af.tree["fgas_500c"][:]
    fbar_500c = af.tree["fbar_500c"][:]
    
    coords = (z, np.log10(m), f)

    fbar_500c_interp = interpolate.RegularGridInterpolator(coords, fbar_500c)

    return fbar_500c_interp
    
# ------------------------------------------------------------------------------
# End of fbar_500c_interp()
# ------------------------------------------------------------------------------

def fstar_500c_max_interp(f_c, sigma_lnc, m_file=None):
    '''
    Return the interpolator for the c200m(m200m) relation
    '''
    if m_file is None:
        m_file = table_dir + "m500c_to_m200m_dmo_fc_{}_slnc_{}.asdf".format(f_c,
                                                                            sigma_lnc)

    af = asdf.open(m_file)

    if f_c != af.tree["f_c"]:
        raise ValueError("f_c in {} is not {}".format(m_file, f_c))
    if sigma_lnc != af.tree["sigma_lnc"]:
        raise ValueError("sigma_lnc in {} is not {}".format(m_file, sigma_lnc))

    z = af.tree["z"][:]
    m = af.tree["m500c"][:]
    fstar_500c_max = af.tree["fstar_500c_max"][:]

    coords = (z, np.log10(m))
    fstar_500c_max_interp = interpolate.RegularGridInterpolator(coords, fstar_500c_max)

    return fstar_500c_max_interp
    
# ------------------------------------------------------------------------------
# End of fstar_500c_max_interp()
# ------------------------------------------------------------------------------
