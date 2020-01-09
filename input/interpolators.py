import numpy as np
import scipy.interpolate as interpolate
import asdf
from copy import copy
import warnings
import dill

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


def arrays_to_ogrid(*xi):
    '''
    Return an ordered list of N arrays reshaped to an ogrid to match to the
    total dimension N with shape (n1, ..., nk, ..., nn)
    '''
    n = len(xi)

    # all arrays need to be reshaped to (1,...,-1,..,1)
    # where -1 corresponds to their own dimension
    shape = [1, ] * n

    ogrid = []
    for i, x in enumerate(xi):
        s = copy(shape)
        s[i] = -1
        s = tuple(s)
        ogrid.append(np.reshape(x, s))

    return ogrid


###################################
# Functions to interpolate tables #
###################################

def c200c_cosmo_interp(c_file="c200c_correa_cosmo.asdf"):
    '''
    Return the interpolator for the given file
    '''
    c_file = table_dir + c_file

    with asdf.open(c_file, copy_arrays=True) as af:
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


def c200c_interp(c_file="c200c_correa.asdf"):
    '''
    Return the interpolator for the given file
    '''
    c_file = table_dir + c_file

    with asdf.open(c_file, copy_arrays=True) as af:
        z = af.tree["z"][:]
        m = af.tree["m200c"][:]
        c = af.tree["c200c"][:]

    coords = (z, np.log10(m))

    c_interp = interpolate.RegularGridInterpolator(coords, c)

    return c_interp


def c200c_emu(m200c=np.logspace(10, 15, 100),
              z=np.linspace(0, 1, 10),
              sigma8=0.82,
              omegam=0.28,
              n=0.97,
              h=0.7):
    '''
    Calculate the c(m) relation from Correa+2015 for the given mass, z and
    cosmology range from our emulator

    Parameters
    ----------
    m200c : array [M_sun / h]
        halo mass at overdensity 200 rho_crit
    z : array
        redshifts
    sigma8 : float
        value of sigma8
    omegam : float
        value of omegam
    n : float
        value of n
    h : float
        value of h

    Returns
    ------
    c200c : concentration
    '''
    # load our saved interpolator info
    with open(table_dir + "c200c_cosmo_interpolator", "rb") as f:
        interp_info = dill.load(f)

    pcs = interp_info["Phi_pca"]
    weights_interp = interp_info["w_interp"]
    mu_interp = interp_info["mu_interp"]

    m200c_interp = interp_info["m200c"]
    z_interp = interp_info["z"]

    s8_interp = interp_info["sigma8"]
    om_interp = interp_info["omegam"]
    n_interp = interp_info["n"]
    h_interp = interp_info["h"]

    # warn about parameter ranges
    if np.all(omegam > om_interp) or np.all(omegam < om_interp):
        warnings.warn("omega_m outside of interpolated range [{}, {}]"
                      .format(om_interp.min(),
                              om_interp.max()),
                      UserWarning)

    if np.all(sigma8 > s8_interp) or np.all(sigma8 < s8_interp):
        warnings.warn("sigma_8 outside of interpolated range"
                      .format(s8_interp.min(),
                              s8_interp.max()),
                      UserWarning)

    if np.all(n > n_interp) or np.all(n < n_interp):
        warnings.warn("n outside of interpolated range"
                      .format(n_interp.min(),
                              n_interp.max()),
                      UserWarning)

    if np.all(h > h_interp) or np.all(h < h_interp):
        warnings.warn("h outside of interpolated range"
                      .format(h_interp.min(),
                              h_interp.max()),
                      UserWarning)

    mu = mu_interp(sigma8, omegam, n, h)
    weights = np.empty((pcs.shape[-1], ), dtype=float)
    for idx, wi in enumerate(weights_interp):
        weights[idx] = wi(sigma8, omegam, n, h)

    # the resulting c200c(z, m)
    c200c = (np.dot(pcs, weights) + mu)

    # interpolate along z
    c200c_interp_z = interpolate.interp1d(z_interp, c200c, axis=0)
    c200c_z = c200c_interp_z(z)

    # interpolate along m200m
    c200c_interp_m = interpolate.interp1d(np.log10(m200c_interp),
                                          c200c_z,
                                          axis=-1)
    c200c_mz = c200c_interp_m(np.log10(m200c))

    return c200c_mz


def c500c_interp(c_file="c500c_correa.asdf"):
    '''
    Return the interpolator for the given file
    '''
    c_file = table_dir + c_file

    with asdf.open(c_file, copy_arrays=True) as af:
        z = af.tree["z"][:]
        m = af.tree["m500c"][:]
        c = af.tree["c500c"][:]

    coords = (z, np.log10(m))

    c_interp = interpolate.RegularGridInterpolator(coords, c)

    return c_interp


def c200m_interp(c_file="c200m_correa.asdf"):
    '''
    Return the interpolator for the c200m(m200m) relation
    '''
    c_file = table_dir + c_file

    with asdf.open(c_file, copy_arrays=True) as af:
        z = af.tree["z"][:]
        m = af.tree["m200m"][:]
        c = af.tree["c200m"][:]

    coords = (z, np.log10(m))

    c_interp = interpolate.RegularGridInterpolator(coords, c)

    return c_interp


def c200m_emu(m200m=np.logspace(10, 15, 100),
              z=np.linspace(0, 1, 10),
              sigma8=0.82,
              omegam=0.28,
              n=0.97,
              h=0.7):
    '''
    Calculate the c(m) relation from Correa+2015 for the given mass, z and
    cosmology range from our emulator

    Parameters
    ----------
    m200m : array [M_sun / h]
        halo mass at overdensity 200 rho_crit
    z : array
        redshifts
    sigma8 : float
        value of sigma8
    omegam : float
        value of omegam
    n : float
        value of n
    h : float
        value of h

    Returns
    ------
    c200m : concentration
    '''
    # load our saved interpolator info
    with open(table_dir + "c200m_cosmo_interpolator", "rb") as f:
        interp_info = dill.load(f)

    pcs = interp_info["Phi_pca"]
    weights_interp = interp_info["w_interp"]
    mu_interp = interp_info["mu_interp"]

    m200m_interp = interp_info["m200m"]
    z_interp = interp_info["z"]

    s8_interp = interp_info["sigma8"]
    om_interp = interp_info["omegam"]
    n_interp = interp_info["n"]
    h_interp = interp_info["h"]

    # warn about parameter ranges
    if np.all(omegam > om_interp) or np.all(omegam < om_interp):
        warnings.warn("omega_m outside of interpolated range [{}, {}]"
                      .format(om_interp.min(),
                              om_interp.max()),
                      UserWarning)

    if np.all(sigma8 > s8_interp) or np.all(sigma8 < s8_interp):
        warnings.warn("sigma_8 outside of interpolated range"
                      .format(s8_interp.min(),
                              s8_interp.max()),
                      UserWarning)

    if np.all(n > n_interp) or np.all(n < n_interp):
        warnings.warn("n outside of interpolated range"
                      .format(n_interp.min(),
                              n_interp.max()),
                      UserWarning)

    if np.all(h > h_interp) or np.all(h < h_interp):
        warnings.warn("h outside of interpolated range"
                      .format(h_interp.min(),
                              h_interp.max()),
                      UserWarning)

    mu = mu_interp(sigma8, omegam, n, h)
    weights = np.empty((pcs.shape[-1], ), dtype=float)
    for idx, wi in enumerate(weights_interp):
        weights[idx] = wi(sigma8, omegam, n, h)

    # the resulting c200m(z, m)
    c200m = (np.dot(pcs, weights) + mu)

    # interpolate along z
    c200m_interp_z = interpolate.interp1d(z_interp, c200m, axis=0)
    c200m_z = c200m_interp_z(z)

    # interpolate along m200m
    c200m_interp_m = interpolate.interp1d(np.log10(m200m_interp),
                                          c200m_z,
                                          axis=-1)
    c200m_mz = c200m_interp_m(np.log10(m200m))

    return c200m_mz


def m200m_dmo_interp(f_c, sigma_lnc, m_file=None):
    '''
    Return the interpolator for the c200m(m200m) relation
    '''
    if m_file is None:
        m_file = "m500c_to_m200m_dmo_fc_{}_slnc_{}.asdf".format(f_c,
                                                                sigma_lnc)

    m_file = table_dir + m_file

    with asdf.open(m_file, copy_arrays=True) as af:
        if f_c != af.tree["f_c"]:
            raise ValueError("f_c in {} is not {}".format(m_file, f_c))
        if sigma_lnc != af.tree["sigma_lnc"]:
            raise ValueError("sigma_lnc in {} is not {}".format(m_file,
                                                                sigma_lnc))
        z = af.tree["z"][:]
        m = af.tree["m500c"][:]
        f = af.tree["fgas_500c"][:]
        m200m_dmo = af.tree["m200m_dmo"][:]

    coords = (z, np.log10(m), f)

    m200m_dmo_interp = interpolate.RegularGridInterpolator(coords, m200m_dmo)

    return m200m_dmo_interp


def fcen500c_interp(f_c, sigma_lnc, m_file=None):
    '''
    Return the interpolator for the c200m(m200m) relation
    '''
    if m_file is None:
        m_file = "m500c_to_m200m_dmo_fc_{}_slnc_{}.asdf".format(f_c,
                                                                sigma_lnc)

    m_file = table_dir + m_file

    with asdf.open(m_file, copy_arrays=True) as af:
        if f_c != af.tree["f_c"]:
            raise ValueError("f_c in {} is not {}".format(m_file, f_c))
        if sigma_lnc != af.tree["sigma_lnc"]:
            raise ValueError("sigma_lnc in {} is not {}".format(m_file,
                                                                sigma_lnc))
        z = af.tree["z"][:]
        m = af.tree["m500c"][:]
        f = af.tree["fgas_500c"][:]
        fcen_500c = af.tree["fcen_500c"][:]

    coords = (z, np.log10(m), f)

    fcen_500c_interp = interpolate.RegularGridInterpolator(coords, fcen_500c)

    return fcen_500c_interp


def fsat500c_interp(f_c, sigma_lnc, m_file=None):
    '''
    Return the interpolator for the c200m(m200m) relation
    '''
    if m_file is None:
        m_file = "m500c_to_m200m_dmo_fc_{}_slnc_{}.asdf".format(f_c,
                                                                sigma_lnc)

    m_file = table_dir + m_file

    with asdf.open(m_file, copy_arrays=True) as af:
        if f_c != af.tree["f_c"]:
            raise ValueError("f_c in {} is not {}".format(m_file, f_c))
        if sigma_lnc != af.tree["sigma_lnc"]:
            raise ValueError("sigma_lnc in {} is not {}".format(m_file,
                                                                sigma_lnc))
        z = af.tree["z"][:]
        m = af.tree["m500c"][:]
        f = af.tree["fgas_500c"][:]
        fsat_500c = af.tree["fsat_500c"][:]

    coords = (z, np.log10(m), f)

    fsat_500c_interp = interpolate.RegularGridInterpolator(coords, fsat_500c)

    return fsat_500c_interp


def fbar500c_interp(f_c, sigma_lnc, m_file=None):
    '''
    Return the interpolator for the c200m(m200m) relation
    '''
    if m_file is None:
        m_file = "m500c_to_m200m_dmo_fc_{}_slnc_{}.asdf".format(f_c,
                                                                sigma_lnc)

    m_file = table_dir + m_file

    with asdf.open(m_file, copy_arrays=True) as af:
        if f_c != af.tree["f_c"]:
            raise ValueError("f_c in {} is not {}".format(m_file, f_c))
        if sigma_lnc != af.tree["sigma_lnc"]:
            raise ValueError("sigma_lnc in {} is not {}".format(m_file,
                                                                sigma_lnc))
        z = af.tree["z"][:]
        m = af.tree["m500c"][:]
        f = af.tree["fgas_500c"][:]
        fbar_500c = af.tree["fbar_500c"][:]

    coords = (z, np.log10(m), f)

    fbar_500c_interp = interpolate.RegularGridInterpolator(coords, fbar_500c)

    return fbar_500c_interp


def fgas_500c_max_interp(f_c, sigma_lnc, m_file=None):
    '''
    Return the interpolator for the c200m(m200m) relation
    '''
    if m_file is None:
        m_file = "m500c_to_m200m_dmo_fc_{}_slnc_{}.asdf".format(f_c,
                                                                sigma_lnc)

    m_file = table_dir + m_file
    with asdf.open(m_file, copy_arrays=True) as af:
        if f_c != af.tree["f_c"]:
            raise ValueError("f_c in {} is not {}".format(m_file, f_c))
        if sigma_lnc != af.tree["sigma_lnc"]:
            raise ValueError("sigma_lnc in {} is not {}".format(m_file,
                                                                sigma_lnc))
        z = af.tree["z"][:]
        m = af.tree["m500c"][:]
        fgas_500c_max = af.tree["fgas_500c_max"][:]

    coords = (z, np.log10(m))
    fgas_500c_max_interp = interpolate.RegularGridInterpolator(coords,
                                                               fgas_500c_max)

    return fgas_500c_max_interp


def gamma_max_interp_test(f_c, sigma_lnc, r_c, beta, r_flat, m_file=None):
    '''
    Return the interpolator for the gamma_max(z, m500c, fgas500c) relation
    '''
    if m_file is None:
        fs = [f_c, sigma_lnc, r_c, beta, r_flat]
        fname_append = "_fc_{}_slnc_{}_rc_{}_beta_{}_rflat_{}.asdf".format(*fs)
        m_file = "m500c_to_gamma_max" + fname_append

    m_file = table_dir + m_file

    with asdf.open(m_file, copy_arrays=True) as af:
        if f_c != af.tree["f_c"]:
            raise ValueError("f_c in {} is not {}".format(m_file, f_c))
        if sigma_lnc != af.tree["sigma_lnc"]:
            raise ValueError("sigma_lnc in {} is not {}".format(m_file,
                                                                sigma_lnc))
        # if r_c != af.tree["r_c"]:
        #     raise ValueError("r_c in {} is not {}".format(m_file, r_c))
        # if beta != af.tree["beta"]:
        #     raise ValueError("beta in {} is not {}".format(m_file, beta))
        # if r_flat != af.tree["r_flat"]:
        #     raise ValueError("r_flat in {} is not {}".format(m_file, r_flat))
        m = af.tree["m500c"][:]
        f = af.tree["fgas_500c"][:]
        gamma_max = af.tree["gamma_max"][:].reshape(m.shape[0], f.shape[0])

    coords = (np.log10(m), f)

    gamma_max_interp = interpolate.RegularGridInterpolator(coords, gamma_max)

    return gamma_max_interp


def gamma_max_interp(f_c=0.86, sigma_lnc=0.0, r_c=0.21,
                     beta=0.71, r_flat=None, m_file=None):
    '''
    Return the interpolator for the gamma_max(z, m500c, fgas500c) relation
    '''
    if m_file is None:
        fs = [f_c, sigma_lnc, r_c, beta, r_flat]
        fname_append = "_fc_{}_slnc_{}_rc_{}_beta_{}_rflat_{}.asdf".format(*fs)
        m_file = "m500c_to_gamma_max" + fname_append

    m_file = table_dir + m_file

    with asdf.open(m_file, copy_arrays=True) as af:
        if f_c != af.tree["f_c"]:
            raise ValueError("f_c in {} is not {}".format(m_file, f_c))
        if sigma_lnc != af.tree["sigma_lnc"]:
            raise ValueError("sigma_lnc in {} is not {}".format(m_file,
                                                                sigma_lnc))
        # if r_c != af.tree["r_c"]:
        #     raise ValueError("r_c in {} is not {}".format(m_file, r_c))
        # if beta != af.tree["beta"]:
        #     raise ValueError("beta in {} is not {}".format(m_file, beta))
        # if r_flat != af.tree["r_flat"]:
        #     raise ValueError("r_flat in {} is not {}".format(m_file, r_flat))
        z = af.tree["z"][:]
        m = af.tree["m500c"][:]
        f = af.tree["fgas_500c"][:]
        gamma_max = af.tree["gamma_max"][:]

    coords = (z, np.log10(m), f)

    gamma_max_interp = interpolate.RegularGridInterpolator(coords, gamma_max)

    return gamma_max_interp
