import numpy as np
import scipy.optimize as opt
import scipy.interpolate as interpolate
import multiprocessing as multi
import asdf
from copy import copy

import pyDOE as pd
from sklearn.decomposition import PCA
import dill

import halo.tools as tools
import halo.input.interpolators as inp_interp
import halo.density_profiles as dp
import halo.cosmo as cosmo

import warnings
import sys
if sys.version_info[0] >= 3:
    print("We cannot import commah, use the supplied table")
else:
    import commah

import pdb

if len(__file__.split("/")[:-1]) >= 1:
    table_dir = "/".join(__file__.split("/")[:-1]) + "/"
else:
    table_dir = "".join(__file__.split("/")[:-1])

######################
# default parameters #
######################

m200c = np.logspace(7, 17, 500)
m200m = np.logspace(8, 17, 500)
m500c = np.logspace(8, 16, 500)

# z = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3.5, 5])
z = np.linspace(0, 3, 10)

cosmo = cosmo.Cosmology()
cosmo_ref = np.array([cosmo.sigma_8,
                      cosmo.omegam,
                      cosmo.omegav,
                      cosmo.n,
                      cosmo.h])

# 4/5D cosmological parameter space (closed universe)
sigma8_r=np.array([cosmo_ref[0] - 0.2,
                   cosmo_ref[0] + 0.2])
omegam_r=np.array([cosmo_ref[1] - 0.2,
                   cosmo_ref[1] + 0.2])
omegab_r=np.array([cosmo.omegab - 0.01,
                   cosmo.omegab + 0.01])
n_r=np.array([cosmo_ref[3] - 0.05,
              cosmo_ref[3] + 0.05])
h_r=np.array([cosmo_ref[4] - 0.1,
              cosmo_ref[4] + 0.1])


######################################
# Hypercubes for different functions #
######################################
# set up the latin hypercube for the concentration calculations
n_lh = 200
cosmo_coords = pd.lhs(4, n_lh, criterion="maximin")

sigma8_c = (cosmo_coords[:, 0] * (sigma8_r.max() - sigma8_r.min()) +
            sigma8_r.min())
omegam_c = (cosmo_coords[:, 1] * (omegam_r.max() - omegam_r.min()) +
            omegam_r.min())
n_c = (cosmo_coords[:, 2] * (n_r.max() - n_r.min()) +
       n_r.min())
h_c = (cosmo_coords[:, 3] * (h_r.max() - h_r.min()) +
       h_r.min())

# set up the latin hypercube for the mass calculations
cosmo_coords = pd.lhs(5, n_lh, criterion="maximin")

sigma8_m = (cosmo_coords[:, 0] * (sigma8_r.max() - sigma8_r.min()) +
            sigma8_r.min())
omegam_m = (cosmo_coords[:, 1] * (omegam_r.max() - omegam_r.min()) +
            omegam_r.min())
omegab_m = (cosmo_coords[:, 2] * (omegab_r.max() - omegab_r.min()) +
            omegab_r.min())
n_m = (cosmo_coords[:, 3] * (n_r.max() - n_r.min()) +
       n_r.min())
h_m = (cosmo_coords[:, 4] * (h_r.max() - h_r.min()) +
       h_r.min())


###############################################
# Tools for table and interpolation functions #
###############################################

def optimize(func, a, b, args, cond=None, fill=None,
             fill_low=np.nan, fill_hi=np.nan):
    """
    Helper function to brentq which should be called wrapped by
    np.vectorize(optimize, otypes=[np.float64])

    In the case cond is satisfied, return fill value

    In the case of a ValueError, fill_low is returned if both bounds
    are < 0, and fill_hi if both bounds > 0.
    """
    # t1 = time.time()
    if cond:
        result = fill
    else:
        try:
            result = opt.brentq(func, a, b, args=args)
        except ValueError:
            if func(a, *args) < 0 and func(b, *args) < 0:
                result = fill_low
            elif func(a, *args) > 0 and func(b, *args) > 0:
                result = fill_hi
            else:
                result = np.nan
                print("===============================")
                print("args: ", *args)
                print("===============================")

    # t2 = time.time()
    return result


def convert_cosmo_commah(cosmo):
    """
    Collect parameters into a dictionary suitable for cosmolopy.

    Returns
    -------
    dict
        Dictionary of values appropriate for cosmolopy
    """
    amap = {"h": "h",
            "omegav": "omega_lambda_0",
            "sigma8": "sigma_8",
            "omegam": "omega_M_0",
            "n": "n"}

    return_dict = {}
    for k, v in amap.items():
        return_dict.update({v: cosmo[k]})

    return return_dict


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
        ogrid.append(x.reshape(s))

    return ogrid


def interp(interpolator, *xi):
    '''
    Wrapping function around interpolator that automatically adjusts the
    1-D coordinates *xi to the regular grid required by the interpolator.

    The output is reshaped to the correctly matching ogrid
    '''
    # convert the given arrays into the correct coordinates for the
    # interpolator
    coords = arrays_to_coords(*xi)

    # create the shape that will match arrays_to_ogrid(*xi)
    shape = ()
    for x in xi:
        shape += x.shape

    return interpolator(coords).reshape(shape)


##############################
# Functions to create tables #
##############################

def table_c200c_correa(m200c=m200c,
                       z=z,
                       sigma8=0.821,
                       omegam=0.2793,
                       omegav=0.7207,
                       n=0.972,
                       h=0.7,
                       cpus=None):
    '''
    Calculate the c(m) relation from Correa+2015 for the given mass, z
    and cosmology.

    Parameters
    ----------
    m200c : array [M_sun / h]
        range of m200c for which to compute
    z : array
        redshifts to compute for
    sigma8 : float
        value of sigma_8 to compute for
    omegam : float
        value of omega_m to compute for
    omegav : float
        value of omega_lambda to compute for
    n : float
        value of n to compute for
    h : float
        value of h to compute for
    cpus : int
        number of cores to use

    Returns
    ------
    results : dict
        dict with c and all input values

        - this dict also gets saved to c_correa_200c.asdf

    '''
    def c_cosmo(procn, m200c, out_q):
        c_all = commah.run(cosmology=cosmo,
                           Mi=m200c/cosmo["h"],
                           z=z,
                           mah=False)['c'].T

        out_q.put([procn, c_all])

    # --------------------------------------------------
    cosmo = {}
    cosmo["sigma_8"] = sigma8
    cosmo["omega_M_0"] = omegam
    cosmo["omega_lambda_0"] = omegav
    cosmo["n"] = n
    cosmo["h"] = h

    if cpus is None:
        cpus = multi.cpu_count()

    manager = multi.Manager()
    out_q = manager.Queue()

    m200c_split = np.array_split(m200c, cpus)

    procs = []
    for i in range(cpus):
        process = multi.Process(target=c_cosmo,
                                args=(i, m200c_split[i], out_q))

        procs.append(process)
        process.start()

    results = []
    for i in range(cpus):
        results.append(out_q.get())

    # need to sort results
    results.sort()
    c = np.concatenate([item[1] for item in results], axis=-1)

    result_info = {
        "dims": np.array(["z", "m200c"]),
        "sigma8": sigma8,
        "omegam": omegam,
        "omegav": omegav,
        "n": n,
        "h": h,
        "z": z,
        "m200c": m200c,
        "c200c": c
    }

    af = asdf.AsdfFile(result_info)
    af.write_to(table_dir + "c200c_correa.asdf")
    af.close()

    return result_info


def table_c500c_correa(m500c=m500c,
                       z=z,
                       sigma8=0.821,
                       omegam=0.2793,
                       omegav=0.7207,
                       n=0.972,
                       h=0.7,
                       cpus=None):
    '''Interpolate the c500c(m500c, cosmo) relation from Correa+2015
    for the given mass, z and cosmology to a regular grid

    Parameters
    ----------
    m500c : array [M_sun / h]
        range of m500c for which to compute
    z : array
        redshifts to compute for
    sigma8 : array
        values of sigma_8 to compute for
    omegam : array
        values of omega_m to compute for
    omegav : array
        values of omega_lambda to compute for
    n : array
        values of n to compute for
    h : array
        values of h to compute for
    cpus : int
        number of cores to use

    Returns
    -------
    results : dict
        dict with c and all input values

        - this dict also gets saved to c_correa_200c.asdf

    '''
    def c_cosmo(procn, z_t, m500c_t, c500c_t, out_q):
        shape = z_t.shape + m500c.shape
        c_all = np.empty(shape)

        # need to tile redshifts to match the masses
        coords = np.vstack([np.tile(z_t.reshape(-1, 1),
                                    (1, m500c_t.shape[1])).flatten(),
                            np.log10(m500c_t).flatten()]).T
        c_interp = interpolate.LinearNDInterpolator(coords, c500c_t.flatten())

        # now interpolate to a regular and fixed grid in m500c
        # need to match m500c to each z
        tiled_z = np.tile(z_t.reshape(-1, 1), (1, m500c.shape[0]))
        tiled_m500c = np.log10(np.tile(m500c.reshape(1, -1),
                                       (z_t.shape[0], 1)))
        coords_new = np.vstack([tiled_z.flatten(),
                                tiled_m500c.flatten()]).T
        c_all = c_interp(coords_new).reshape(z_t.shape + m500c.shape)

        out_q.put([procn, c_all])

    # --------------------------------------------------
    if cpus is None:
        cpus = multi.cpu_count()

    if cpus > 8:
        cpus = 8

    manager = multi.Manager()
    out_q = manager.Queue()

    # load tables
    af = asdf.open(table_dir + "halo_200c_to_500c.asdf")
    z_tab = af.tree["z"]
    m500c_tab = af.tree["m500c"][:]
    c500c_tab = af.tree["c500c"][:]

    # we split along the redshift axis, only mass is not reg grid
    m500c_tab_split = np.array_split(m500c_tab, cpus, axis=-2)
    c500c_tab_split = np.array_split(c500c_tab, cpus, axis=-2)
    z_tab_split = np.array_split(z_tab, cpus, axis=0)

    procs = []
    for i in range(cpus):
        process = multi.Process(target=c_cosmo,
                                args=(i,
                                      z_tab_split[i],
                                      m500c_tab_split[i],
                                      c500c_tab_split[i],
                                      out_q))

        procs.append(process)
        process.start()

    results = []
    for i in range(cpus):
        results.append(out_q.get())

    # need to sort results
    results.sort()
    c500c = np.concatenate([item[1] for item in results], axis=-2)

    result_info = {
        "dims": np.array(["sigma8",
                          "omegam",
                          "omegav",
                          "n",
                          "h",
                          "z",
                          "m500c"]),
        "sigma8": sigma8,
        "omegam": omegam,
        "omegav": omegav,
        "n": n,
        "h": h,
        "z": z,
        "m500c": m500c,
        "c500c": c500c}

    af = asdf.AsdfFile(result_info)
    af.write_to(table_dir + "c500c_correa.asdf")
    af.close()

    return result_info


def massdiff_2m2c(m200m, m200c, c200c, r200c, rhom):
    '''
    Integrate an NFW halo with m200m up to r200c and return the mass difference
    between the integral and m200c
    '''
    r200m = tools.mass_to_radius(m200m, 200 * rhom)
    mass = dp.m_NFW_delta(r200c, c200c * r200m / r200c, r200m, rhom, Delta=200)

    return mass - m200c


@np.vectorize
def m200c_to_m200m(m200c, c200c, r200c, rhom):
    # these bounds should be reasonable for m200m < 1e18
    # 1e19 Msun is ~maximum for c_correa
    m200m = opt.brentq(massdiff_2m2c, m200c, 10. * m200c,
                       args=(m200c, c200c, r200c, rhom))
    r200m = tools.mass_to_radius(m200m, 200 * rhom)
    c200m = c200c * r200m / r200c

    return m200m, c200m, r200m


def table_m200c_to_m200m(m200c=m200c,
                         z=z,
                         sigma8=0.821,
                         omegam=0.2793,
                         omegav=0.7207,
                         n=0.972,
                         h=0.7):
    '''
    Create a table that converts from m200c to the corresponding halo
    properties m200m
    '''
    rhoc = 2.755 * 10**(11.)  # [h^2 M_sun / Mpc^3]
    rhom = omegam * rhoc

    # get interpolator and coordinates for c200c
    coords = arrays_to_coords(z, np.log10(m200c))
    c_interp = inp_interp.c200c_interp()

    # scaling for rhoc with redshift
    E2_z = omegam * (1+z)**3 + omegav

    c200c = c_interp(coords).reshape(z.shape + m200c.shape)
    r200c = tools.mass_to_radius(m200c.reshape(1, -1),
                                 200 * rhoc * E2_z.reshape(-1, 1))

    m200m, c200m, r200m = m200c_to_m200m(m200c=m200c.reshape(1, -1),
                                         c200c=c200c,
                                         r200c=r200c,
                                         rhom=rhom * (1 + z.reshape(-1, 1))**3)

    result_info = {
        "dims": np.array(["z", "m200c"]),
        "sigma8": sigma8,
        "omegam": omegam,
        "omegav": omegav,
        "n": n,
        "h": h,
        "z": z,
        "m200c": m200c,
        "r200c": r200c,
        "c200c": c200c,
        "m200m": m200m,
        "r200m": r200m,
        "c200m": c200m}

    af = asdf.AsdfFile(result_info)
    af.write_to(table_dir + "halo_200c_to_200m.asdf")
    af.close()

    return result_info


def massdiff_5c2c(m500c, m200c, c200c, r200c, rhoc):
    '''
    Integrate an NFW halo with c200c to r500c and return the mass difference
    '''
    r500c = tools.mass_to_radius(m500c, 500 * rhoc)
    mass = dp.m_NFW(r=r200c, r_x=r500c, c_x=c200c * r500c / r200c,
                    m_x=m500c)

    return mass - m200c


@np.vectorize
def m200c_to_m500c(m200c, c200c, r200c, rhoc):
    # these bounds should be reasonable for m500c < 1e18
    # 1e19 Msun is ~maximum for c_correa
    m500c = opt.brentq(massdiff_5c2c, 0.5 * m200c, 10. * m200c,
                       args=(m200c, c200c, r200c, rhoc))
    r500c = tools.mass_to_radius(m500c, 500 * rhoc)
    c500c = c200c * r500c / r200c

    return m500c, c500c, r500c


def table_m200c_to_m500c(m200c=m200c,
                         z=z,
                         sigma8=0.821,
                         omegam=0.2793,
                         omegav=0.7207,
                         n=0.972,
                         h=0.7):
    '''
    Create a table that converts from m200c to the corresponding halo
    properties m500c
    '''
    rhoc = 2.755 * 10**(11.)  # [h^2 M_sun / Mpc^3]

    # get interpolator and coordinates for c200c
    coords = arrays_to_coords(z, np.log10(m200c))
    c_interp = inp_interp.c200c_interp()

    # scaling for rhoc with redshift
    E2_z = omegam * (1+z)**3 + omegav

    c200c = c_interp(coords).reshape(z.shape + m200c.shape)
    r200c = tools.mass_to_radius(m200c, 200 * rhoc * E2_z.reshape(-1, 1))

    m500c, c500c, r500c = m200c_to_m500c(m200c=m200c.reshape(1, -1),
                                         c200c=c200c,
                                         r200c=r200c,
                                         rhoc=rhoc * E2_z.reshape(-1, 1))

    result_info = {
        "dims": np.array(["z", "m200c"]),
        "sigma8": sigma8,
        "omegam": omegam,
        "omegav": omegav,
        "n": n,
        "h": h,
        "z": z,
        "m200c": m200c,
        "r200c": r200c,
        "c200c": c200c,
        "m500c": m500c,
        "r500c": r500c,
        "c500c": c500c}

    af = asdf.AsdfFile(result_info)
    af.write_to(table_dir + "halo_200c_to_500c.asdf")
    af.close()

    return result_info


def table_c200m_correa(m200m=m200m,
                       z=z,
                       sigma8=0.821,
                       omegam=0.2793,
                       omegav=0.7207,
                       n=0.972,
                       h=0.7,
                       cpus=None):
    '''Interpolate the c200m(m200m, cosmo) relation from Correa+2015 for
    the given mass, z and cosmology to a regular grid

    Parameters
    ----------
    m200m : array [M_sun / h]
        range of m200m for which to compute
    z : array
        redshifts to compute for
    sigma8 : array
        values of sigma_8 to compute for
    omegam : array
        values of omega_m to compute for
    omegav : array
        values of omega_lambda to compute for
    n : array
        values of n to compute for
    h : array
        values of h to compute for
    cpus : int
        number of cores to use

    Returns
    -------
    results : dict
        dict with c and all input values

        - this dict also gets saved to c_correa_200c.asdf

    '''
    def c_cosmo(procn, z_t, m200m_t, c200m_t, out_q):
        shape = z_t.shape + m200m.shape
        c_all = np.empty(shape)

        # need to tile redshifts to match the masses
        coords = np.vstack([np.tile(z_t.reshape(-1, 1),
                                    (1, m200m_t.shape[1])).flatten(),
                            np.log10(m200m_t).flatten()]).T
        c_interp = interpolate.LinearNDInterpolator(coords, c200m_t.flatten())

        # now interpolate to a regular and fixed grid in m200m
        # need to match m200m to each z
        coords_new = np.vstack([np.tile(z_t.reshape(-1, 1),
                                        (1, m200m.shape[0])).flatten(),
                                np.log10(np.tile(m200m.reshape(1, -1),
                                                 (z_t.shape[0], 1))).flatten()]).T
        c_all = c_interp(coords_new).reshape(z_t.shape + m200m.shape)

        out_q.put([procn, c_all])

    # --------------------------------------------------
    if cpus is None:
        cpus = multi.cpu_count()

    if cpus > 8:
        cpus = 8

    manager = multi.Manager()
    out_q = manager.Queue()

    # load tables
    af = asdf.open(table_dir + "halo_200c_to_200m.asdf")
    z_tab = af.tree["z"]
    m200m_tab = af.tree["m200m"][:]
    c200m_tab = af.tree["c200m"][:]

    # we split along the redshift axis, only mass is not reg grid
    m200m_tab_split = np.array_split(m200m_tab, cpus, axis=-2)
    c200m_tab_split = np.array_split(c200m_tab, cpus, axis=-2)
    z_tab_split = np.array_split(z_tab, cpus, axis=0)

    procs = []
    for i in range(cpus):
        process = multi.Process(target=c_cosmo,
                                args=(i,
                                      z_tab_split[i],
                                      m200m_tab_split[i],
                                      c200m_tab_split[i],
                                      out_q))

        procs.append(process)
        process.start()

    results = []
    for i in range(cpus):
        results.append(out_q.get())

    # need to sort results
    results.sort()
    c200m = np.concatenate([item[1] for item in results], axis=-2)

    result_info = {
        "dims": np.array(["sigma8", "omegam", "omegav",
                          "n", "h", "z", "m200m"]),
        "sigma8": sigma8,
        "omegam": omegam,
        "omegav": omegav,
        "n": n,
        "h": h,
        "z": z,
        "m200m": m200m,
        "c200m": c200m
    }

    af = asdf.AsdfFile(result_info)
    af.write_to(table_dir + "c200m_correa.asdf")
    af.close()

    return result_info


def f_stars_interp(comp='all'):
    '''Return the stellar fraction interpolator as a function of halo
    mass as found by Zu & Mandelbaum (2015).

    For m200m < 1e10 M_sun/h we return f_stars=0
    For m200m > 1e16 M_sun/h we return f_stars=1.41e-2

    THIS INTERPOLATOR ASSUMES h=0.7 FOR EVERYTHING!!

    Returns
    -------
    f_stars : (m,) array [h_70^(-1)]
      total stellar fraction for the halo mass

    '''
    comp_options = ['all', 'cen', 'sat']
    if comp not in comp_options:
        raise ValueError('comp needs to be in {}'.format(comp_options))

    # m_h is in Hubble units
    # all the fractions have assumed h=0.7
    fs_file = table_dir + 'observations/m200m_fstar_fcen_fsat.txt'
    m_h, f_stars, f_cen, f_sat = np.loadtxt(fs_file, unpack=True)

    # we need to convert the halo mass to h=0.7 as well
    m_h = m_h / 0.7

    if comp == 'all':
        f_stars_interp = interpolate.interp1d(m_h, f_stars, bounds_error=False,
                                              fill_value=(0, f_stars[-1]))
    elif comp == 'cen':
        f_stars_interp = interpolate.interp1d(m_h, f_cen, bounds_error=False,
                                              fill_value=(0, f_cen[-1]))
    else:
        f_stars_interp = interpolate.interp1d(m_h, f_sat, bounds_error=False,
                                              fill_value=(0, f_sat[-1]))

    return f_stars_interp


def table_m500c_to_m200m_dmo(m500c=m500c,
                             z=z,
                             fg500c=np.linspace(0, 1, 100),
                             f_c=0.86,
                             sigma_lnc=0.0,
                             sigma8=0.821,
                             omegam=0.2793,
                             omegab=0.0463,
                             omegav=0.7207,
                             n=0.972,
                             h=0.7,
                             fname="m500c_to_m200m_dmo",
                             cpus=None):
    '''Create a table that computes the DMO equivalent halo mass
    m200m_dmo given the observed m500c & fgas_500c (normalized to the
    cosmic baryon fraction). The stellar fraction are also computed
    and saved.

    Parameters
    ----------
    m500c : array [M_sun / h]
        range of m500c for which to compute for
    z : array
        redshifts to compute for
    fg500c : array [normalised to omega_b / omega_m]
        range of gas fractions to compute for
    f_c : float
      ratio between satellite concentration and DM concentration
    sigma_lnc : float
      logarithmic offset to take the c(m) relation at
    sigma8 : array
        values of sigma_8 to compute for
    omegam : array
        values of omega_m to compute for
    omegav : array
        values of omega_lambda to compute for
    n : array
        values of n to compute for
    h : array
        values of h to compute for
    cpus : int
        number of cores to use

    Returns
    ------
    results : dict
        dict with m200m_dmo, c200m_dmo, fstar_500c and all input values

        - this dict gets saved to fname

    '''
    def m_diff(m200m_dmo, m500c, r500c, fg500c, fs200m, fc200m, c200m, z):
        # for a given halo mass, we know the concentration
        try:
            c200m_dmo = c200m(np.array([z, np.log10(m200m_dmo)]))
            c200m_dmo = c200m_dmo * np.e**sigma_lnc
        except ValueError:
            print(np.array([z, np.log10(m200m_dmo)]))
        r200m_dmo = tools.mass_to_radius(m200m_dmo,
                                         200 * rhom * (1+z)**3)

        # this give stellar fraction & concentration
        fcen_500c = fc200m(m200m_dmo / 0.7) * m200m_dmo / m500c
        fsat_200m = fs200m(m200m_dmo / 0.7)

        # which allows us to compute the stellar fraction at r500c
        fsat_500c = dp.m_NFW(r500c, m_x=fsat_200m*m200m_dmo, c_x=f_c*c200m_dmo,
                             r_x=r200m_dmo) / m500c

        # this is NOT m500c for our DMO halo, this is our DMO halo
        # evaluated at r500c for the observations, which when scaled
        # should match the observed m500c
        m_dmo_r500c = dp.m_NFW_delta(r500c, c200m_dmo, r200m_dmo,
                                     rhom * (1+z)**3, Delta=200)

        fb = omegab / omegam
        f500c = fg500c + fcen_500c + fsat_500c
        m500c_cor = m_dmo_r500c * (1 - fb) / (1 - f500c)
        return m500c_cor - m500c

    # --------------------------------------------------
    def calc_m_diff(procn, m500c, r500c, fg500c, fs200m,
                    fc200m, c200m, z, out_q):
        m200m_dmo = np.vectorize(optimize, otypes=[np.float64])(m_diff, m500c,
                                                                5. * m500c,
                                                                (m500c, r500c,
                                                                 fg500c,
                                                                 fs200m,
                                                                 fc200m,
                                                                 c200m, z))
        # now we have m200m_dmo, so we calculate again all the other
        # resulting variables

        # we need to tile the redshift to match m200m_dmo
        shape_final = m200m_dmo.shape
        z_tiled = np.tile(z, (1,) + shape_final[1:])
        coords = np.vstack([z_tiled.flatten(),
                            np.log10(m200m_dmo.flatten())]).T

        # calculate the concentration and reshape to match m200m_dmo
        c200m_dmo = c200m(coords).reshape(shape_final) * np.e**sigma_lnc
        r200m_dmo = tools.mass_to_radius(m200m_dmo, 200 * rhom * (1+z)**3)

        fcen_500c = fc200m(m200m_dmo / 0.7) * m200m_dmo / m500c
        fsat_200m = fs200m(m200m_dmo / 0.7)
        fsat_500c = dp.m_NFW(r500c, m_x=fsat_200m*m200m_dmo, c_x=f_c*c200m_dmo,
                             r_x=r200m_dmo) / m500c

        out_q.put([procn, m200m_dmo, fcen_500c, fsat_500c])

    # --------------------------------------------------
    if cpus is None:
        cpus = multi.cpu_count()

    manager = multi.Manager()
    out_q = manager.Queue()

    # reshape variables to match shapes
    (z_r, m500c_r, fg500c_r) = arrays_to_ogrid(z, m500c, fg500c)
    fgas_500c = fg500c_r * omegab / omegam
    fcen_200m = f_stars_interp(comp="cen")
    fsat_200m = f_stars_interp(comp="sat")

    # set background densities
    rhoc = 2.755 * 10**(11.)  # [h^2 M_sun / Mpc^3]
    rhom = omegam * rhoc

    E2_z_r = omegam * (1 + z_r)**3 + omegav
    r500c_r = tools.mass_to_radius(m500c_r, 500 * rhoc * E2_z_r)

    # otherwise the code gets upset when passing empty arrays to optimize
    if cpus > m500c.shape[0]:
        cpus = m500c.shape[0]

    m500c_split = np.array_split(m500c_r, cpus, axis=-2)
    r500c_split = np.array_split(r500c_r, cpus, axis=-2)
    c200m = inp_interp.c200m_interp()

    procs = []
    for i, (mi, ri) in enumerate(zip(m500c_split, r500c_split)):
        process = multi.Process(target=calc_m_diff,
                                args=(i, mi, ri, fgas_500c, fsat_200m,
                                      fcen_200m, c200m, z_r, out_q))

        procs.append(process)
        process.start()

    results = []
    for i in range(len(m500c_split)):
        results.append(out_q.get())

    # need to sort results
    results.sort()
    m200m_dmo = np.concatenate([item[1] for item in results], axis=-2)
    fcen_500c = np.concatenate([item[2] for item in results], axis=-2)
    fsat_500c = np.concatenate([item[3] for item in results], axis=-2)
    fbar_500c = fgas_500c + fsat_500c + fcen_500c

    # we cannot halo baryon fractions greater than cosmic
    mask = (fbar_500c > omegab / omegam)
    # m200m_dmo = np.ma.masked_array(m200m_dmo, mask=mask)
    # fcen_500c = np.ma.masked_array(fcen_500c, mask=mask)
    # fsat_500c = np.ma.masked_array(fsat_500c, mask=mask)
    # fbar_500c = np.ma.masked_array(fbar_500c, mask=mask)
    # m200m_dmo[mask] = np.nan
    # fcen_500c[mask] = np.nan
    # fsat_500c[mask] = np.nan
    # fbar_500c[mask] = np.nan

    # we need to get the asymptotic gas fraction where fbar500c = fbar
    fgas_500c_r = np.tile(fgas_500c, (np.shape(z)[0], np.shape(m500c)[0], 1))
    fgas_500c_r[mask] = np.nan
    fgas_500c_max = np.nanmax(fgas_500c_r, axis=-1)

    result_info = {
        "dims": np.array(["z", "m500c", "fgas_500c", "m200m_dmo"]),
        "sigma8": sigma8,
        "omegam": omegam,
        "omegab": omegab,
        "omegav": omegav,
        "n": n,
        "h": h,
        "z": z,
        "f_c": f_c,
        "sigma_lnc": sigma_lnc,
        "m500c": m500c,
        "fgas_500c_max": fgas_500c_max,
        "fgas_500c": fg500c * omegab / omegam,
        "fcen_500c": fcen_500c,
        "fsat_500c": fsat_500c,
        "fbar_500c": fbar_500c,
        "mask": mask,
        "m200m_dmo": m200m_dmo
    }

    af = asdf.AsdfFile(result_info)
    af.write_to(table_dir + fname + "_fc_{}_slnc_{}.asdf".format(f_c,
                                                                 sigma_lnc))
    af.close()

    return result_info


def table_m500c_to_gamma_max(m500c=m500c,
                             z=z,
                             fg500c=np.linspace(0, 1, 100),
                             r_c=0.21,
                             beta=0.71,
                             r_flat=None,
                             f_c=0.86,
                             sigma_lnc=0.0,
                             sigma8=0.821,
                             omegam=0.2793,
                             omegab=0.0463,
                             omegav=0.7207,
                             n=0.972,
                             h=0.7,
                             fname="m500c_to_gamma_max",
                             cpus=None):
    '''
    Create a table that computes the maximum gamma for each m500c & fgas_500c
    (normalized to the cosmic baryon fraction) with the requirement that the
    cosmic baryon fraction is just reached at r200m_obs

    Parameters
    ----------
    m500c : array [M_sun / h]
        range of m500c for which to compute for
    z : array
        redshifts to compute for
    fg500c : array [normalised to omega_b / omega_m]
        range of gas fractions to compute for
    gamma : array
        range of extrapolated slopes between r500c and r200m_obs
    beta : float
        best-fit slope of the beta profile
    r_c : float [in units of r500c]
        best-fit core radius of the beta profile
    r_flat : float [in units of r500c]
        radius at which the hot gas density profile goes flat
    f_c : float
      ratio between satellite concentration and DM concentration
    sigma_lnc : float
      logarithmic offset to take the c(m) relation at
    sigma8 : array
        values of sigma_8 to compute for
    omegam : array
        values of omega_m to compute for
    omegav : array
        values of omega_lambda to compute for
    n : array
        values of n to compute for
    h : array
        values of h to compute for
    cpus : int
        number of cores to use

    Returns
    ------
    results : dict
        dict with gamma_max(z, m500c, fg500c)

        - this dict gets saved to fname
    '''
    def m_dm_stars_from_m500c(z, m500c, fg500c, fc500c, fs500c, r500c,
                              m200m_dmo, c200m_dmo, f_c, sigma_lnc):
        # load the interpolated quantities
        m200m = m200m_dmo((z, np.log10(m500c), fg500c))
        r200m = tools.mass_to_radius(m200m, 200 * omegam * rhoc)
        c200m = c200m_dmo((z, np.log10(m200m))) * np.e**sigma_lnc
        fcen500c = fc500c((z, np.log10(m500c), fg500c))
        fsat500c = fs500c((z, np.log10(m500c), fg500c))
        fstars500c = fsat500c + fcen500c

        # convert c200m to c500c_sat
        c500c_sat = (f_c * c200m * r500c / r200m)

        # get dm mass and DMO mass at r500c
        m500c_dm = m500c * (1 - fg500c - fstars500c)
        m500c_dmo = m500c * (1 - fg500c - fstars500c) / (1 - omegab / omegam)

        # now we know
        cen_args = {'m_x': fcen500c * m500c}
        sat_args = {'m_x': fsat500c * m500c,
                    'c_x': c500c_sat,
                    'r_x': r500c}

        m_stars = lambda r, **kwargs: (dp.m_delta(r, **cen_args) +
                                       dp.m_NFW(r, **sat_args))

        # ## #
        # DM #
        # ## #
        c500c_dm = c200m * r500c / r200m
        dm_args = {'m_x': m500c_dm,
                   'c_x': c500c_dm,
                   'r_x': r500c}

        m_dm = lambda  r, **kwargs: dp.m_NFW(r, **dm_args)

        return m_dm, m_stars

    def m_gas_from_m500c(gamma, z, m500c, fg500c, r500c):
        rho_500c = dp.profile_beta(np.reshape(r500c, (-1, 1)),
                                   m_x=np.array([fg500c * m500c]),
                                   r_x=np.array([r500c]),
                                   r_c=np.array([r_c * r500c]),
                                   beta=np.array([beta])).reshape(-1)
        gas_args =  {'m_x': np.array([fg500c * m500c]),
                     'r_x': np.array([r500c]),
                     'r_c': np.array([r_c * r500c]),
                     'beta': np.array([beta]),
                     'gamma': np.array([gamma]),
                     'rho_x': np.array([rho_500c])}

        # this function should be called with a scalar
        if r_flat is None:
            m_gas = lambda r, **kwargs: float(dp.m_beta_plaw(r, **gas_args))
        else:
            gas_args['r_y'] = np.array([r_flat * r500c]),
            m_gas = lambda r, **kwargs: float(dp.m_beta_plaw_uni(r, **gas_args))
        return m_gas

    def r200m_from_m(m_f, r200m_dmo, z, **kwargs):
        def diff_m200m(r):
            m200m = 4. / 3 * np.pi * 200 * rhom * (1+z)**3 * r**3
            m_diff = m_f(r, **kwargs) - m200m
            return m_diff

        r200m = opt.brentq(diff_m200m, 0.5 * r200m_dmo, 10 * r200m_dmo)
        return r200m

    def fb_diff(gamma, z, m500c, fg500c, r500c, fc500c, fs500c, m200m_dmo,
                c200m_dmo):
        m_dm, m_stars = m_dm_stars_from_m500c(z=z, m500c=m500c, fg500c=fg500c,
                                              fc500c=fc500c, fs500c=fs500c,
                                              r500c=r500c, m200m_dmo=m200m_dmo,
                                              c200m_dmo=c200m_dmo, f_c=f_c,
                                              sigma_lnc=sigma_lnc)
        m_gas = m_gas_from_m500c(gamma=gamma, z=z, m500c=m500c, fg500c=fg500c,
                                 r500c=r500c)
        m_tot = lambda r: m_dm(r) + m_stars(r) + m_gas(r)
        # print(m_gas(r500c), m_stars(r500c), m_dm(r500c))
        # if fg500c == f_b:
        #     print(fg500c, m200m_dmo((z, np.log10(m500c), fg500c)))
        m200m = m200m_dmo((z, np.log10(m500c), fg500c))
        r200m = tools.mass_to_radius(m200m, 200 * omegam * rhoc * (1+z)**3)
        r200m_obs = np.vectorize(r200m_from_m, otypes=[np.float64])(m_tot,
                                                                    r200m,
                                                                    z)
        return (m_gas(r200m_obs) + m_stars(r200m_obs)) / m_tot(r200m_obs) - f_b

    # @np.vectorize
    # def compute_fbar_diff_gamma(gamma, z, m500c, fg500c, r500c, fc500c, fs500c, m200m_dmo, c200m_dmo):
    #     m_dm, m_stars = m_dm_stars_from_m500c(z=z, m500c=m500c, fg500c=fg500c,
    #                                           fc500c=fc500c, fs500c=fs500c,
    #                                           r500c=r500c, m200m_dmo=m200m_dmo,
    #                                           c200m_dmo=c200m_dmo, f_c=f_c,
    #                                           sigma_lnc=sigma_lnc)
    #     m_gas = m_gas_from_m500c(gamma=gamma, z=z, m500c=m500c, fg500c=fg500c,
    #                              r500c=r500c)
    #     m_bar = lambda r: m_stars(r) + m_gas(r)
    #     m_tot = lambda r: m_dm(r) + m_stars(r) + m_gas(r)
    #     m200m = m200m_dmo((z, np.log10(m500c), fg500c))
    #     r200m = tools.mass_to_radius(m200m, 200 * omegam * rhoc)
    #     r200m_obs = np.vectorize(r200m_from_m, otypes=[np.float64])(m_tot, r200m)
    #     fb_diff = ((m_gas(r200m_obs) + m_stars(r200m_obs)) / m_tot(r200m_obs)) - f_b

    #     if np.abs(fb_diff) > 1e-4 and fg500c > 0 and fg500c < f_b:
    #         print("z          :", z)
    #         print("m500c      :", np.log10(m500c))
    #         print("fg500c     :", fg500c)
    #         print("r200m_dmo  :", r200m / r500c)
    #         print("r200m_obs  :", r200m_obs / r500c)
    #         print("fb_200m_dmo:", m_bar(r200m) / m_tot(r200m))
    #         print("gamma      :", gamma)
    #         print("fb_diff    :", fb_diff)
    #         print("================")

    #     return fb_diff

    # ----------------------------------------------------------------------------
    def gamma_from_m500c(procn, z, m500c, fg500c, r500c,
                         fc500c, fs500c, m200m_dmo, c200m_dmo,
                         out_q):
        gamma = np.vectorize(optimize, otypes=[np.float64])(fb_diff, 0., 1000.,
                                                            *(z, m500c, fg500c,
                                                              r500c,
                                                              fc500c,
                                                              fs500c,
                                                              m200m_dmo,
                                                              c200m_dmo),
                                                            cond=(fg500c == 0),
                                                            fill=0, fill_low=0,
                                                            fill_hi=np.nan)
        # fb_200m_obs = compute_fbar_diff_gamma(gamma, z, m500c, fg500c, r500c, fc500c,
        #                                       fs500c, m200m_dmo, c200m_dmo)
        out_q.put([procn, gamma])
        # --------------------------------------------------
    if cpus is None:
        cpus = multi.cpu_count()

    manager = multi.Manager()
    out_q = manager.Queue()

    # reshape variables to match shapes
    (z_r, m500c_r, fg500c_r) = arrays_to_ogrid(z, m500c, fg500c)
    fgas500c_r = fg500c_r * omegab / omegam
    # set background density
    rhoc = 2.755 * 10**(11.)  # [h^2 M_sun / Mpc^3]
    rhom = omegam * rhoc
    f_b = omegab / omegam

    # scaling for rhoc with redshift
    E2_z_r = omegam * (1+z_r)**3 + omegav

    r500c_r = tools.mass_to_radius(m500c_r, 500 * rhoc * E2_z_r)

    # load interpolators
    fcen_500c = inp_interp.fcen500c_interp(f_c=f_c, sigma_lnc=sigma_lnc)
    fsat_500c = inp_interp.fsat500c_interp(f_c=f_c, sigma_lnc=sigma_lnc)
    m200m_dmo = inp_interp.m200m_dmo_interp(f_c=f_c, sigma_lnc=sigma_lnc)
    c200m_dmo = inp_interp.c200m_interp()

    # otherwise the code gets upset when passing empty arrays to optimize
    if cpus > m500c.shape[0]:
        cpus = m500c.shape[0]

    # split arrays along m500c axis
    m500c_split = np.array_split(m500c_r, cpus, axis=1)
    r500c_split = np.array_split(r500c_r, cpus, axis=1)

    procs = []
    for i, (mi, ri) in enumerate(zip(m500c_split, r500c_split)):
        process = multi.Process(target=gamma_from_m500c,
                                args=(i, z_r, mi, fgas500c_r, ri, fcen_500c,
                                      fsat_500c, m200m_dmo, c200m_dmo, out_q))

        procs.append(process)
        process.start()

    results = []
    for i in range(len(m500c_split)):
        results.append(out_q.get())

    # need to sort results
    results.sort()
    gamma_max = np.concatenate([item[1] for item in results], axis=1)

    # we still need to get our mask
    fbar_500c = inp_interp.fbar500c_interp(f_c=f_c, sigma_lnc=sigma_lnc)((z_r,
                                                                          np.log10(m500c_r),
                                                                          fgas500c_r))
    mask = (fbar_500c == np.nan)

    result_info = {
        "dims": np.array(["z", "m500c", "fgas_500c"]),
        "sigma8": sigma8,
        "omegam": omegam,
        "omegab": omegab,
        "omegav": omegav,
        "n": n,
        "h": h,
        "z": z,
        "f_c": f_c,
        "sigma_lnc": sigma_lnc,
        "r_c": r_c,
        "beta": beta,
        "r_flat": r_flat,
        "m500c": m500c,
        "fgas_500c": fg500c * omegab / omegam,
        "gamma_max": gamma_max,
        "mask": mask
    }

    af = asdf.AsdfFile(result_info)
    fname_append = "_fc_{}_slnc_{}_rc_{}_beta_{}_rflat_{}.asdf".format(f_c, sigma_lnc,
                                                                       r_c, beta, r_flat)
    af.write_to(table_dir + fname + fname_append)
    af.close()

    return result_info


#################################
# Cosmology dependent emulators #
#################################

def table_c200c_correa_cosmo(m200c=m200c,
                             z=z,
                             sigma8=sigma8_c,
                             omegam=omegam_c,
                             n=n_c,
                             h=h_c,
                             cpus=None):
    '''
    Calculate the c(m) relation from Correa+2015 for the given mass, z and
    cosmology range.

    Parameters
    ----------
    m200c : array [M_sun / h]
        halo mass at overdensity 200 rho_crit
    z : array
        redshifts
    sigma8 : array
        values of sigma8
    omegam : array
        values of omegam
    n : array
        values of n
    h : array
        values of h
    cpus : int
        number of cores to use

    Returns
    ------
    results : dict
        dict with c and all input values

        - this dict also gets saved to c_correa_200c.asdf

    '''
    def c_cosmo(procn, m200c, sigma8, omegam, omegav, n, h, out_q):
        cosmo = {}
        c_all = np.empty(z.shape + m200c.shape + sigma8.shape)

        for idx, s8 in enumerate(sigma8):
            cosmo["sigma_8"] = s8
            cosmo["omega_M_0"] = omegam[idx]
            cosmo["omega_lambda_0"] = omegav[idx]
            cosmo["n"] = n[idx]
            cosmo["h"] = h[idx]
            c_all[..., idx] = commah.run(cosmology=cosmo,
                                         Mi=m200c/cosmo["h"],
                                         z=z,
                                         mah=False)['c'].T

        out_q.put([procn, c_all])

    # --------------------------------------------------
    if cpus is None:
        cpus = multi.cpu_count()

    manager = multi.Manager()
    out_q = manager.Queue()

    m200c_split = np.array_split(m200c, cpus)

    # closed universe
    omegav = 1 - omegam

    procs = []
    for i in range(cpus):
        process = multi.Process(target=c_cosmo,
                                args=(i, m200c_split[i], sigma8, omegam,
                                      omegav, n, h, out_q))

        procs.append(process)
        process.start()

    results = []
    for i in range(cpus):
        results.append(out_q.get())

    # need to sort results
    results.sort()
    c = np.concatenate([item[1] for item in results], axis=-1)

    result_info = {
        "dims": np.array(["z",
                          "m200c",
                          "sigma8",
                          "omegam",
                          "n",
                          "h"]),
        "sigma8": sigma8,
        "omegam": omegam,
        "omegav": omegav,
        "n": n,
        "h": h,
        "z": z,
        "m200c": m200c,
        "c200c": c
    }

    af = asdf.AsdfFile(result_info)
    af.write_to(table_dir + "c200c_correa_cosmo.asdf")
    af.close()

    return result_info


def table_m200c_to_m200m_cosmo(cpus=None):
    '''
    Table that converts each table_c200c_correa_cosmo to the corresponding m200m

    Parameters
    ----------
    cpus : int
        number of cores to use

    Returns
    ------
    results : dict
        dict with c and all input values

        - this dict also gets saved to halo_200c_to_200m_cosmo.asdf
    '''
    with asdf.open(table_dir + "c200c_correa_cosmo.asdf",
                   copy_arrays=True) as af:
        c200c = af.tree["c200c"][:]
        m200c = af.tree["m200c"][:]
        z = af.tree["z"][:]
        sigma8 = af.tree["sigma8"][:]
        omegam = af.tree["omegam"][:]
        omegav = af.tree["omegav"][:]
        n = af.tree["n"][:]
        h = af.tree["h"][:]

    omegav = 1 - omegam
    # scaling for rhoc with redshift
    E2_z = (omegam.reshape(1, 1, -1) * (1+z.reshape(-1, 1, 1))**3 +
            omegav.reshape(1, 1, -1))

    rhoc = 2.755 * 10**(11.)  # [h^2 M_sun / Mpc^3]
    rhoc_z = rhoc * E2_z

    rhom = omegam * rhoc
    rhom_z = rhom * (1 + z.reshape(-1, 1, 1))**3

    r200c = tools.mass_to_radius(m200c.reshape(1, -1, 1),
                                 200 * rhoc_z)
    m200m, c200m, r200m = m200c_to_m200m(m200c=m200c.reshape(1, -1, 1),
                                         c200c=c200c,
                                         r200c=r200c,
                                         rhom=rhom_z)

    result_info = {
        "dims": np.array(["z", "m200c"]),
        "sigma8": sigma8,
        "omegam": omegam,
        "omegav": omegav,
        "n": n,
        "h": h,
        "z": z,
        "m200c": m200c,
        "r200c": r200c,
        "c200c": c200c,
        "m200m": m200m,
        "r200m": r200m,
        "c200m": c200m}

    af = asdf.AsdfFile(result_info)
    af.write_to(table_dir + "halo_200c_to_200m_cosmo.asdf")
    af.close()

    return result_info


def table_c200m_correa_cosmo(m200m=m200m,
                             cpus=None):
    '''
    Calculate the c(m) relation from Correa+2015 for the given mass, z and
    cosmology range.

    Parameters
    ----------
    m200m : array [M_sun / h]
        halo mass at overdensity 200 rho_m
    cpus : int
        number of cores to use

    Returns
    ------
    results : dict
        dict with c and all input values

        - this dict also gets saved to c_correa_200c.asdf

    '''
    def c_cosmo(procn, z_t, m200m_t, c200m_t, out_q):
        shape = (z_t.shape[0], m200m.shape[0], c200m_t.shape[-1])
        c_all = np.empty(shape)

        # loop over the cosmological dependence
        for idx_c in np.arange(c200m_t.shape[-1]):
            # need to tile redshifts to match the masses
            coords = np.vstack([np.tile(z_t.reshape(-1, 1),
                                        (1, m200m_t.shape[1])).flatten(),
                                np.log10(m200m_t[..., idx_c]).flatten()]).T

            c_interp = interpolate.LinearNDInterpolator(coords,
                                                        c200m_t[..., idx_c].flatten())

            # now interpolate to a regular and fixed grid in m200m
            # need to match m200m to each z
            coords_new = np.vstack([np.tile(z_t.reshape(-1, 1),
                                            (1, m200m.shape[0])).flatten(),
                                    np.log10(np.tile(m200m.reshape(1, -1),
                                                     (z_t.shape[0], 1))).flatten()]).T
            c_all[..., idx_c] = c_interp(coords_new).reshape(z_t.shape
                                                             + m200m.shape)

        out_q.put([procn, c_all])

    # --------------------------------------------------
    if cpus is None:
        cpus = multi.cpu_count()

    if cpus > 8:
        cpus = 8

    manager = multi.Manager()
    out_q = manager.Queue()

    with asdf.open(table_dir + "halo_200c_to_200m_cosmo.asdf",
                   copy_arrays=True) as af:
        z_tab = af.tree["z"][:]
        m200m_tab = af.tree["m200m"][:]
        c200m_tab = af.tree["c200m"][:]
        sigma8 = af.tree["sigma8"][:]
        omegam = af.tree["omegam"][:]
        omegav = af.tree["omegav"][:]
        n = af.tree["n"][:]
        h = af.tree["h"][:]

    # we split along the redshift axis, only mass is not reg grid
    m200m_tab_split = np.array_split(m200m_tab, cpus, axis=0)
    c200m_tab_split = np.array_split(c200m_tab, cpus, axis=0)
    z_tab_split = np.array_split(z_tab, cpus, axis=0)

    procs = []
    for i in range(cpus):
        process = multi.Process(target=c_cosmo,
                                args=(i,
                                      z_tab_split[i],
                                      m200m_tab_split[i],
                                      c200m_tab_split[i],
                                      out_q))

        procs.append(process)
        process.start()

    results = []
    for i in range(cpus):
        results.append(out_q.get())

    # need to sort results
    results.sort()
    c200m = np.concatenate([item[1] for item in results], axis=0)

    result_info = {
        "dims": np.array(["z", "m200m", "sigma8", "omegam", "n", "h"]),
        "sigma8": sigma8,
        "omegam": omegam,
        "omegav": omegav,
        "n": n,
        "h": h,
        "z": z,
        "m200m": m200m,
        "c200m": c200m
    }

    af = asdf.AsdfFile(result_info)
    af.write_to(table_dir + "c200m_correa_cosmo.asdf")
    af.close()

    return result_info


def setup_c200c_correa_emu(n_comp=7,
                           c_file=table_dir + "c200c_correa_cosmo.asdf"):
    """Fit principal components and cosmology dependent weights for the
    Correa+15 c(m) relation
    """
    # load the data
    with asdf.open(c_file, copy_arrays=True) as af:
        sigma8 = af.tree["sigma8"][:]
        omegam = af.tree["omegam"][:]
        n = af.tree["n"][:]
        h = af.tree["h"][:]
        z = af.tree["z"][:]
        m = af.tree["m200c"][:]
        # has shape(z, m, n_cosmo)
        c = af.tree["c200c"][:]

    # number of cosmologies
    n_coords = sigma8.shape[0]

    pca = PCA(n_components=n_comp)
    # first axis contains (z x m) coordinates, second axis cosmology
    c_pca = pca.fit_transform(c.reshape(-1, n_coords))
    # after transform n_cosmo -> n_comp
    c_pca = c_pca.reshape(c.shape[:-1] + (-1, ))

    # the pca.components_ just contain the weights to transform the PCAs back
    # to the space with the full cosmological information
    w_pca = pca.components_

    # interpolate the mean offset cosmology dependence
    mu_interp = interpolate.Rbf(sigma8, omegam, n, h, pca.mean_)

    w_interp = []
    for w in w_pca:
        w_i = interpolate.Rbf(sigma8, omegam, n, h, w)
        w_interp.append(w_i)

    interp_info = {"m200c": m,
                   "z": z,
                   "dims": np.array(["sigma8", "omegam", "n", "h"]),
                   "sigma8": sigma8,
                   "omegam": omegam,
                   "n": n,
                   "h": h,
                   "w_pca": w_pca,
                   "Phi_pca": c_pca,
                   "w_interp": w_interp,
                   "mu_interp": mu_interp}

    with open(table_dir + "c200c_cosmo_interpolator", "wb") as f:
        dill.dump(interp_info, f)


def setup_c200m_correa_emu(n_comp=7,
                           c_file=table_dir + "c200m_correa_cosmo.asdf"):
    """Fit principal components and cosmology dependent weights for the
    Correa+15 c(m) relation
    """
    # load the data
    with asdf.open(c_file, copy_arrays=True) as af:
        sigma8 = af.tree["sigma8"][:]
        omegam = af.tree["omegam"][:]
        n = af.tree["n"][:]
        h = af.tree["h"][:]
        z = af.tree["z"][:]
        m = af.tree["m200m"][:]
        # has shape(z, m, n_cosmo)
        c = af.tree["c200m"][:]

    # number of cosmologies
    n_coords = sigma8.shape[0]

    pca = PCA(n_components=n_comp)
    # first axis contains (z x m) coordinates, second axis cosmology
    c_pca = pca.fit_transform(c.reshape(-1, n_coords))
    # after transform n_cosmo -> n_comp
    c_pca = c_pca.reshape(c.shape[:-1] + (-1, ))

    # the pca.components_ just contain the weights to transform the PCAs back
    # to the space with the full cosmological information
    w_pca = pca.components_

    # interpolate the mean offset cosmology dependence
    mu_interp = interpolate.Rbf(sigma8, omegam, n, h, pca.mean_)

    w_interp = []
    for w in w_pca:
        w_i = interpolate.Rbf(sigma8, omegam, n, h, w)
        w_interp.append(w_i)

    interp_info = {"m200m": m,
                   "z": z,
                   "dims": np.array(["sigma8", "omegam", "n", "h"]),
                   "sigma8": sigma8,
                   "omegam": omegam,
                   "n": n,
                   "h": h,
                   "w_pca": w_pca,
                   "Phi_pca": c_pca,
                   "w_interp": w_interp,
                   "mu_interp": mu_interp}

    with open(table_dir + "c200m_cosmo_interpolator", "wb") as f:
        dill.dump(interp_info, f)


def table_m500c_to_m200m_dmo_cosmo(m500c=m500c,
                                   z=z,
                                   sigma8=sigma8_m,
                                   omegam=omegam_m,
                                   omegab=omegab_m,
                                   n=n_m,
                                   h=h_m,
                                   fg500c=np.linspace(0, 1, 100),
                                   f_c=0.86,
                                   sigma_lnc=0.0,
                                   fname="m500c_to_m200m_dmo",
                                   cpus=None):
    '''Create a table that computes the DMO equivalent halo mass
    m200m_dmo given the observed m500c & fgas_500c (normalized to the
    cosmic baryon fraction). The stellar fraction are also computed
    and saved.

    Parameters
    ----------
    m500c : array [M_sun / h]
        range of m500c for which to compute for
    z : array
        redshifts to compute for
    fg500c : array [normalised to omega_b / omega_m]
        range of gas fractions to compute for
    f_c : float
      ratio between satellite concentration and DM concentration
    sigma_lnc : float
      logarithmic offset to take the c(m) relation at
    sigma8 : array
        values of sigma_8 to compute for
    omegam : array
        values of omega_m to compute for
    omegav : array
        values of omega_lambda to compute for
    n : array
        values of n to compute for
    h : array
        values of h to compute for
    cpus : int
        number of cores to use

    Returns
    ------
    results : dict
        dict with m200m_dmo, c200m_dmo, fstar_500c and all input values

        - this dict gets saved to fname

    '''
    def m_diff(m200m_dmo, m500c, r500c, fg500c, fs200m, fc200m, c200m, z):
        # for a given halo mass, we know the concentration
        try:
            c200m_dmo = c200m(np.array([z, np.log10(m200m_dmo)]))
            c200m_dmo = c200m_dmo * np.e**sigma_lnc
        except ValueError:
            print(np.array([z, np.log10(m200m_dmo)]))
        r200m_dmo = tools.mass_to_radius(m200m_dmo,
                                         200 * rhom * (1+z)**3)

        # this give stellar fraction & concentration
        fcen_500c = fc200m(m200m_dmo / 0.7) * m200m_dmo / m500c
        fsat_200m = fs200m(m200m_dmo / 0.7)

        # which allows us to compute the stellar fraction at r500c
        fsat_500c = dp.m_NFW(r500c, m_x=fsat_200m*m200m_dmo, c_x=f_c*c200m_dmo,
                             r_x=r200m_dmo) / m500c

        # this is NOT m500c for our DMO halo, this is our DMO halo
        # evaluated at r500c for the observations, which when scaled
        # should match the observed m500c
        m_dmo_r500c = dp.m_NFW_delta(r500c, c200m_dmo, r200m_dmo,
                                     rhom * (1+z)**3, Delta=200)

        fb = omegab / omegam
        f500c = fg500c + fcen_500c + fsat_500c
        m500c_cor = m_dmo_r500c * (1 - fb) / (1 - f500c)
        return m500c_cor - m500c

    # --------------------------------------------------
    def calc_m_diff(procn, m500c, r500c, fg500c, fs200m,
                    fc200m, c200m, z, out_q):
        m200m_dmo = np.vectorize(optimize, otypes=[np.float64])(m_diff, m500c,
                                                                5. * m500c,
                                                                (m500c, r500c,
                                                                 fg500c,
                                                                 fs200m,
                                                                 fc200m,
                                                                 c200m, z))
        # now we have m200m_dmo, so we calculate again all the other
        # resulting variables

        # we need to tile the redshift to match m200m_dmo
        shape_final = m200m_dmo.shape
        z_tiled = np.tile(z, (1,) + shape_final[1:])
        coords = np.vstack([z_tiled.flatten(),
                            np.log10(m200m_dmo.flatten())]).T

        # calculate the concentration and reshape to match m200m_dmo
        c200m_dmo = c200m(coords).reshape(shape_final) * np.e**sigma_lnc
        r200m_dmo = tools.mass_to_radius(m200m_dmo, 200 * rhom * (1+z)**3)

        fcen_500c = fc200m(m200m_dmo / 0.7) * m200m_dmo / m500c
        fsat_200m = fs200m(m200m_dmo / 0.7)
        fsat_500c = dp.m_NFW(r500c, m_x=fsat_200m*m200m_dmo, c_x=f_c*c200m_dmo,
                             r_x=r200m_dmo) / m500c

        out_q.put([procn, m200m_dmo, fcen_500c, fsat_500c])

    # --------------------------------------------------
    if cpus is None:
        cpus = multi.cpu_count()

    manager = multi.Manager()
    out_q = manager.Queue()

    # reshape variables to match shapes
    (z_r, m500c_r, fg500c_r, ) = arrays_to_ogrid(z, m500c, fg500c)
    fgas_500c = fg500c_r * omegab / omegam
    fcen_200m = f_stars_interp(comp="cen")
    fsat_200m = f_stars_interp(comp="sat")

    omegav = 1 - omegam
    # scaling for rhoc with redshift
    E2_z = (omegam.reshape(1, 1, -1) * (1+z.reshape(-1, 1, 1))**3 +
            omegav.reshape(1, 1, -1))

    # set background densities
    rhoc = 2.755 * 10**(11.)  # [h^2 M_sun / Mpc^3]
    rhoc_z = rhoc * E2_z

    rhom = omegam * rhoc
    rhom_z = rhom * (1 + z.reshape(-1, 1, 1))**3

    E2_z_r = omegam * (1 + z_r)**3 + omegav
    r500c_r = tools.mass_to_radius(m500c_r, 500 * rhoc * E2_z_r)

    # otherwise the code gets upset when passing empty arrays to optimize
    if cpus > m500c.shape[0]:
        cpus = m500c.shape[0]

    m500c_split = np.array_split(m500c_r, cpus, axis=-2)
    r500c_split = np.array_split(r500c_r, cpus, axis=-2)
    c200m = inp_interp.c200m_interp()

    procs = []
    for i, (mi, ri) in enumerate(zip(m500c_split, r500c_split)):
        process = multi.Process(target=calc_m_diff,
                                args=(i, mi, ri, fgas_500c, fsat_200m,
                                      fcen_200m, c200m, z_r, out_q))

        procs.append(process)
        process.start()

    results = []
    for i in range(len(m500c_split)):
        results.append(out_q.get())

    # need to sort results
    results.sort()
    m200m_dmo = np.concatenate([item[1] for item in results], axis=-2)
    fcen_500c = np.concatenate([item[2] for item in results], axis=-2)
    fsat_500c = np.concatenate([item[3] for item in results], axis=-2)
    fbar_500c = fgas_500c + fsat_500c + fcen_500c

    # we cannot halo baryon fractions greater than cosmic
    mask = (fbar_500c > omegab / omegam)
    # m200m_dmo = np.ma.masked_array(m200m_dmo, mask=mask)
    # fcen_500c = np.ma.masked_array(fcen_500c, mask=mask)
    # fsat_500c = np.ma.masked_array(fsat_500c, mask=mask)
    # fbar_500c = np.ma.masked_array(fbar_500c, mask=mask)
    # m200m_dmo[mask] = np.nan
    # fcen_500c[mask] = np.nan
    # fsat_500c[mask] = np.nan
    # fbar_500c[mask] = np.nan

    # we need to get the asymptotic gas fraction where fbar500c = fbar
    fgas_500c_r = np.tile(fgas_500c, (np.shape(z)[0], np.shape(m500c)[0], 1))
    fgas_500c_r[mask] = np.nan
    fgas_500c_max = np.nanmax(fgas_500c_r, axis=-1)

    result_info = {
        "dims": np.array(["z", "m500c", "fgas_500c", "m200m_dmo"]),
        "sigma8": sigma8,
        "omegam": omegam,
        "omegab": omegab,
        "omegav": omegav,
        "n": n,
        "h": h,
        "z": z,
        "f_c": f_c,
        "sigma_lnc": sigma_lnc,
        "m500c": m500c,
        "fgas_500c_max": fgas_500c_max,
        "fgas_500c": fg500c * omegab / omegam,
        "fcen_500c": fcen_500c,
        "fsat_500c": fsat_500c,
        "fbar_500c": fbar_500c,
        "mask": mask,
        "m200m_dmo": m200m_dmo
    }

    af = asdf.AsdfFile(result_info)
    af.write_to(table_dir + fname + "_fc_{}_slnc_{}.asdf".format(f_c,
                                                                 sigma_lnc))
    af.close()

    return result_info


def table_m500c_to_gamma_max(m500c=m500c,
                             z=z,
                             fg500c=np.linspace(0, 1, 100),
                             r_c=0.21,
                             beta=0.71,
                             r_flat=None,
                             f_c=0.86,
                             sigma_lnc=0.0,
                             sigma8=0.821,
                             omegam=0.2793,
                             omegab=0.0463,
                             omegav=0.7207,
                             n=0.972,
                             h=0.7,
                             fname="m500c_to_gamma_max",
                             cpus=None):
    '''
    Create a table that computes the maximum gamma for each m500c & fgas_500c
    (normalized to the cosmic baryon fraction) with the requirement that the
    cosmic baryon fraction is just reached at r200m_obs

    Parameters
    ----------
    m500c : array [M_sun / h]
        range of m500c for which to compute for
    z : array
        redshifts to compute for
    fg500c : array [normalised to omega_b / omega_m]
        range of gas fractions to compute for
    gamma : array
        range of extrapolated slopes between r500c and r200m_obs
    beta : float
        best-fit slope of the beta profile
    r_c : float [in units of r500c]
        best-fit core radius of the beta profile
    r_flat : float [in units of r500c]
        radius at which the hot gas density profile goes flat
    f_c : float
      ratio between satellite concentration and DM concentration
    sigma_lnc : float
      logarithmic offset to take the c(m) relation at
    sigma8 : array
        values of sigma_8 to compute for
    omegam : array
        values of omega_m to compute for
    omegav : array
        values of omega_lambda to compute for
    n : array
        values of n to compute for
    h : array
        values of h to compute for
    cpus : int
        number of cores to use

    Returns
    ------
    results : dict
        dict with gamma_max(z, m500c, fg500c)

        - this dict gets saved to fname
    '''
    def m_dm_stars_from_m500c(z, m500c, fg500c, fc500c, fs500c, r500c,
                              m200m_dmo, c200m_dmo, f_c, sigma_lnc):
        # load the interpolated quantities
        m200m = m200m_dmo((z, np.log10(m500c), fg500c))
        r200m = tools.mass_to_radius(m200m, 200 * omegam * rhoc)
        c200m = c200m_dmo((z, np.log10(m200m))) * np.e**sigma_lnc
        fcen500c = fc500c((z, np.log10(m500c), fg500c))
        fsat500c = fs500c((z, np.log10(m500c), fg500c))
        fstars500c = fsat500c + fcen500c

        # convert c200m to c500c_sat
        c500c_sat = (f_c * c200m * r500c / r200m)

        # get dm mass and DMO mass at r500c
        m500c_dm = m500c * (1 - fg500c - fstars500c)
        m500c_dmo = m500c * (1 - fg500c - fstars500c) / (1 - omegab / omegam)

        # now we know
        cen_args = {'m_x': fcen500c * m500c}
        sat_args = {'m_x': fsat500c * m500c,
                    'c_x': c500c_sat,
                    'r_x': r500c}

        m_stars = lambda r, **kwargs: (dp.m_delta(r, **cen_args) +
                                       dp.m_NFW(r, **sat_args))

        # ## #
        # DM #
        # ## #
        c500c_dm = c200m * r500c / r200m
        dm_args = {'m_x': m500c_dm,
                   'c_x': c500c_dm,
                   'r_x': r500c}

        m_dm = lambda  r, **kwargs: dp.m_NFW(r, **dm_args)

        return m_dm, m_stars

    def m_gas_from_m500c(gamma, z, m500c, fg500c, r500c):
        rho_500c = dp.profile_beta(np.reshape(r500c, (-1, 1)),
                                   m_x=np.array([fg500c * m500c]),
                                   r_x=np.array([r500c]),
                                   r_c=np.array([r_c * r500c]),
                                   beta=np.array([beta])).reshape(-1)
        gas_args =  {'m_x': np.array([fg500c * m500c]),
                     'r_x': np.array([r500c]),
                     'r_c': np.array([r_c * r500c]),
                     'beta': np.array([beta]),
                     'gamma': np.array([gamma]),
                     'rho_x': np.array([rho_500c])}

        # this function should be called with a scalar
        if r_flat is None:
            m_gas = lambda r, **kwargs: float(dp.m_beta_plaw(r, **gas_args))
        else:
            gas_args['r_y'] = np.array([r_flat * r500c]),
            m_gas = lambda r, **kwargs: float(dp.m_beta_plaw_uni(r, **gas_args))
        return m_gas

    def r200m_from_m(m_f, r200m_dmo, z, **kwargs):
        def diff_m200m(r):
            m200m = 4. / 3 * np.pi * 200 * rhom * (1+z)**3 * r**3
            m_diff = m_f(r, **kwargs) - m200m
            return m_diff

        r200m = opt.brentq(diff_m200m, 0.5 * r200m_dmo, 10 * r200m_dmo)
        return r200m

    def fb_diff(gamma, z, m500c, fg500c, r500c, fc500c, fs500c, m200m_dmo,
                c200m_dmo):
        m_dm, m_stars = m_dm_stars_from_m500c(z=z, m500c=m500c, fg500c=fg500c,
                                              fc500c=fc500c, fs500c=fs500c,
                                              r500c=r500c, m200m_dmo=m200m_dmo,
                                              c200m_dmo=c200m_dmo, f_c=f_c,
                                              sigma_lnc=sigma_lnc)
        m_gas = m_gas_from_m500c(gamma=gamma, z=z, m500c=m500c, fg500c=fg500c,
                                 r500c=r500c)
        m_tot = lambda r: m_dm(r) + m_stars(r) + m_gas(r)
        # print(m_gas(r500c), m_stars(r500c), m_dm(r500c))
        # if fg500c == f_b:
        #     print(fg500c, m200m_dmo((z, np.log10(m500c), fg500c)))
        m200m = m200m_dmo((z, np.log10(m500c), fg500c))
        r200m = tools.mass_to_radius(m200m, 200 * omegam * rhoc * (1+z)**3)
        r200m_obs = np.vectorize(r200m_from_m, otypes=[np.float64])(m_tot,
                                                                    r200m,
                                                                    z)
        return (m_gas(r200m_obs) + m_stars(r200m_obs)) / m_tot(r200m_obs) - f_b

    # @np.vectorize
    # def compute_fbar_diff_gamma(gamma, z, m500c, fg500c, r500c, fc500c, fs500c, m200m_dmo, c200m_dmo):
    #     m_dm, m_stars = m_dm_stars_from_m500c(z=z, m500c=m500c, fg500c=fg500c,
    #                                           fc500c=fc500c, fs500c=fs500c,
    #                                           r500c=r500c, m200m_dmo=m200m_dmo,
    #                                           c200m_dmo=c200m_dmo, f_c=f_c,
    #                                           sigma_lnc=sigma_lnc)
    #     m_gas = m_gas_from_m500c(gamma=gamma, z=z, m500c=m500c, fg500c=fg500c,
    #                              r500c=r500c)
    #     m_bar = lambda r: m_stars(r) + m_gas(r)
    #     m_tot = lambda r: m_dm(r) + m_stars(r) + m_gas(r)
    #     m200m = m200m_dmo((z, np.log10(m500c), fg500c))
    #     r200m = tools.mass_to_radius(m200m, 200 * omegam * rhoc)
    #     r200m_obs = np.vectorize(r200m_from_m, otypes=[np.float64])(m_tot, r200m)
    #     fb_diff = ((m_gas(r200m_obs) + m_stars(r200m_obs)) / m_tot(r200m_obs)) - f_b

    #     if np.abs(fb_diff) > 1e-4 and fg500c > 0 and fg500c < f_b:
    #         print("z          :", z)
    #         print("m500c      :", np.log10(m500c))
    #         print("fg500c     :", fg500c)
    #         print("r200m_dmo  :", r200m / r500c)
    #         print("r200m_obs  :", r200m_obs / r500c)
    #         print("fb_200m_dmo:", m_bar(r200m) / m_tot(r200m))
    #         print("gamma      :", gamma)
    #         print("fb_diff    :", fb_diff)
    #         print("================")

    #     return fb_diff

    # ----------------------------------------------------------------------------
    def gamma_from_m500c(procn, z, m500c, fg500c, r500c,
                         fc500c, fs500c, m200m_dmo, c200m_dmo,
                         out_q):
        gamma = np.vectorize(optimize, otypes=[np.float64])(fb_diff, 0., 1000.,
                                                            *(z, m500c, fg500c,
                                                              r500c,
                                                              fc500c,
                                                              fs500c,
                                                              m200m_dmo,
                                                              c200m_dmo),
                                                            cond=(fg500c == 0),
                                                            fill=0, fill_low=0,
                                                            fill_hi=np.nan)
        # fb_200m_obs = compute_fbar_diff_gamma(gamma, z, m500c, fg500c, r500c, fc500c,
        #                                       fs500c, m200m_dmo, c200m_dmo)
        out_q.put([procn, gamma])
        # --------------------------------------------------
    if cpus is None:
        cpus = multi.cpu_count()

    manager = multi.Manager()
    out_q = manager.Queue()

    # reshape variables to match shapes
    (z_r, m500c_r, fg500c_r) = arrays_to_ogrid(z, m500c, fg500c)
    fgas500c_r = fg500c_r * omegab / omegam
    # set background density
    rhoc = 2.755 * 10**(11.)  # [h^2 M_sun / Mpc^3]
    rhom = omegam * rhoc
    f_b = omegab / omegam

    # scaling for rhoc with redshift
    E2_z_r = omegam * (1+z_r)**3 + omegav

    r500c_r = tools.mass_to_radius(m500c_r, 500 * rhoc * E2_z_r)

    # load interpolators
    fcen_500c = inp_interp.fcen500c_interp(f_c=f_c, sigma_lnc=sigma_lnc)
    fsat_500c = inp_interp.fsat500c_interp(f_c=f_c, sigma_lnc=sigma_lnc)
    m200m_dmo = inp_interp.m200m_dmo_interp(f_c=f_c, sigma_lnc=sigma_lnc)
    c200m_dmo = inp_interp.c200m_interp()

    # otherwise the code gets upset when passing empty arrays to optimize
    if cpus > m500c.shape[0]:
        cpus = m500c.shape[0]

    # split arrays along m500c axis
    m500c_split = np.array_split(m500c_r, cpus, axis=1)
    r500c_split = np.array_split(r500c_r, cpus, axis=1)

    procs = []
    for i, (mi, ri) in enumerate(zip(m500c_split, r500c_split)):
        process = multi.Process(target=gamma_from_m500c,
                                args=(i, z_r, mi, fgas500c_r, ri, fcen_500c,
                                      fsat_500c, m200m_dmo, c200m_dmo, out_q))

        procs.append(process)
        process.start()

    results = []
    for i in range(len(m500c_split)):
        results.append(out_q.get())

    # need to sort results
    results.sort()
    gamma_max = np.concatenate([item[1] for item in results], axis=1)

    # we still need to get our mask
    fbar_500c = inp_interp.fbar500c_interp(f_c=f_c, sigma_lnc=sigma_lnc)((z_r,
                                                                          np.log10(m500c_r),
                                                                          fgas500c_r))
    mask = (fbar_500c == np.nan)

    result_info = {
        "dims": np.array(["z", "m500c", "fgas_500c"]),
        "sigma8": sigma8,
        "omegam": omegam,
        "omegab": omegab,
        "omegav": omegav,
        "n": n,
        "h": h,
        "z": z,
        "f_c": f_c,
        "sigma_lnc": sigma_lnc,
        "r_c": r_c,
        "beta": beta,
        "r_flat": r_flat,
        "m500c": m500c,
        "fgas_500c": fg500c * omegab / omegam,
        "gamma_max": gamma_max,
        "mask": mask
    }

    af = asdf.AsdfFile(result_info)
    fname_append = "_fc_{}_slnc_{}_rc_{}_beta_{}_rflat_{}.asdf".format(f_c, sigma_lnc,
                                                                       r_c, beta, r_flat)
    af.write_to(table_dir + fname + fname_append)
    af.close()

    return result_info
