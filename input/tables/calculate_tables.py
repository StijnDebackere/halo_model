import numpy as np
import mpmath as mp
import scipy.optimize as opt
import scipy.interpolate as interpolate
import scipy.special as spec
import scipy.integrate as intg
import multiprocessing as multi
import asdf
from copy import copy
import time

import halo.tools as tools
import halo.input.initialize as init
import halo.density_profiles as dp

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

m200c = np.logspace(0, 17, 100)
m200m = np.logspace(1, 17, 100)
m500c = np.logspace(1, 16, 100)

z = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3.5, 5])

cosmo = init.default["parameters"]["cosmology"]
sigma8 = np.linspace(cosmo["sigma8"] - 0.02, cosmo["sigma8"] + 0.02, 3)
omegam = np.linspace(cosmo["omegam"] - 0.02, cosmo["omegam"] + 0.02, 3)
omegab = cosmo["omegab"]
omegav = np.linspace(cosmo["omegav"] - 0.02, cosmo["omegav"] + 0.02, 3)
n = np.linspace(cosmo["n"] - 0.02, cosmo["n"] + 0.02, 3)
h = np.linspace(cosmo["h"] - 0.02, cosmo["h"] + 0.02, 3)

###############################################
# Tools for table and interpolation functions #
###############################################

@np.vectorize
def optimize(func, a, b, *args):
    # t1 = time.time()
    result = opt.brentq(func, a, b, args=args)
    # t2 = time.time()
    # print(t2 - t1)
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
        return_dict.update({v:cosmo[k]})

    return return_dict

# ------------------------------------------------------------------------------
# End of convert_cosmo_commah()
# ------------------------------------------------------------------------------

def arrays_to_coords(*xi):
    '''
    Convert a set of coordinate arrays to a coordinate grid for the interpolator
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
    Return an ordered list of arrays reshaped to an ogrid
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
        ogrid.append(x.reshape(s))

    return ogrid

# ------------------------------------------------------------------------------
# End of arrays_to_ogrid()
# ------------------------------------------------------------------------------

def interp(interpolator, *xi):
    '''
    Wrapping function around interpolator that automatically adjusts the
    coordinates *xi to the interpolator and changes the output to an ogrid
    '''
    # convert the given arrays into the correct coordinates for the interpolator
    coords = arrays_to_coords(*xi)

    # create the shape that will match arrays_to_ogrid(*xi)
    shape = ()
    for x in xi:
        shape += x.shape

    pdb.set_trace()
    return interpolator(coords).reshape(shape)

# ------------------------------------------------------------------------------
# End of interp()
# ------------------------------------------------------------------------------

##############################
# Functions to create tables #
##############################

def table_c200c_correa_cosmo(m200c=m200c,
                             z=z,
                             sigma8=sigma8,
                             omegam=omegam,
                             omegav=omegav,
                             n=n,
                             h=h,
                             cpus=None):
    '''Calculate the c(m) relation from Correa+2015 for the given mass, z
    and cosmology range.

    Parameters
    ----------
    m200c : array [M_sun / h]
        range of m200c for which to compute
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
    ------
    results : dict
        dict with c and all input values

        - this dict also gets saved to c_correa_200c.asdf

    '''
    def c_cosmo(procn, m200c, sigma8, omegam, omegav, n, h, out_q):
        cosmo = {}
        c_all = np.empty(sigma8.shape + omegam.shape + omegav.shape + n.shape +
                         h.shape + z.shape + m200c.shape)
        for idx_s8, s8 in enumerate(sigma8):
            cosmo["sigma_8"] = s8
            for idx_om, om in enumerate(omegam):
                cosmo["omega_M_0"] = om
                for idx_ov, ov in enumerate(omegav):
                    cosmo["omega_lambda_0"] = ov
                    for idx_n, n_s in enumerate(n):
                        cosmo["n"] = n_s
                        for idx_h, h0 in enumerate(h):
                            cosmo["h"] = h0
                            c_all[idx_s8, idx_om, idx_ov, idx_n, idx_h] = commah.run(cosmology=cosmo,
                                                                                     Mi=m200c/cosmo["h"],
                                                                                     z=z,
                                                                                     mah=False)['c'].T
        out_q.put([procn, c_all])

    # --------------------------------------------------
    if cpus == None:
        cpus = multi.cpu_count()

    manager = multi.Manager()
    out_q = manager.Queue()

    m200c_split = np.array_split(m200c, cpus)    

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
        "dims": np.array(["sigma8", "omegam", "omegav", "n", "h", "z", "m200c"]),
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

# ------------------------------------------------------------------------------
# End of table_c200c_correa_cosmo()
# ------------------------------------------------------------------------------

def table_c200c_correa(m200c=m200c,
                       z=z,
                       sigma8=0.821,
                       omegam=0.2793,
                       omegav=0.7207,
                       n=0.972,
                       h=0.7,
                       cpus=None):
    '''Calculate the c(m) relation from Correa+2015 for the given mass, z
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
    
    if cpus == None:
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

# ------------------------------------------------------------------------------
# End of table_c200c_correa()
# ------------------------------------------------------------------------------

def massdiff_2m2c(m200m, m200c, c200c, r200c, rhom, h, z):
    '''
    Integrate an NFW halo with m200m up to r200c and return the mass difference
    between the integral and m200c
    '''
    r200m = tools.mass_to_radius(m200m, 200 * rhom)
    mass = dp.m_NFW_delta(r200c, c200c * r200m / r200c, r200m, rhom, Delta=200)

    return mass - m200c

@np.vectorize
def m200c_to_m200m(m200c, c200c, r200c, rhom, h, z):
    # these bounds should be reasonable for m200m < 1e18
    # 1e19 Msun is ~maximum for c_correa
    m200m = opt.brentq(massdiff_2m2c, m200c, 10. * m200c,
                       args=(m200c, c200c, r200c, rhom, h, z))
    r200m = tools.mass_to_radius(m200m, 200 * rhom)
    c200m = c200c * r200m / r200c

    return m200m, c200m, r200m

# ------------------------------------------------------------------------------
# End of m200c_to_m200m()
# ------------------------------------------------------------------------------

def table_m200c_to_m200m(m200c=m200c,
                         z=z,
                         sigma8=0.821,
                         omegam=0.2793,
                         omegav=0.7207,
                         n=0.972,
                         h=0.7):
    '''
    Create a table that converts from m200c to the corresponding halo properties
    m200m
    '''
    rhoc = 2.755 * 10**(11.) # [h^2 M_sun / Mpc^3]
    rhom = omegam * rhoc

    # get interpolator and coordinates for c200c
    coords = arrays_to_coords(z, np.log10(m200c))
    c_interp = c200c_interp(c_file=table_dir + "c200c_correa.asdf")

    c200c = c_interp(coords).reshape(z.shape + m200c.shape)
    r200c = tools.mass_to_radius(m200c, 200 * rhoc)
    
    m200m, c200m, r200m = m200c_to_m200m(m200c=m200c.reshape(1,-1),
                                         c200c=c200c,
                                         r200c=r200c.reshape(1,-1),
                                         rhom=rhom,
                                         h=h,
                                         z=z.reshape(-1,1))

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

# ------------------------------------------------------------------------------
# End of table_m200c_to_m200m()
# ------------------------------------------------------------------------------

def massdiff_2m2c_cosmo(m200m, m200c, c200c, r200c, rhom, h, z):
    '''
    Integrate an NFW halo with m200m up to r200c and return the mass difference
    between the integral and m200c
    '''
    r200m = tools.mass_to_radius(m200m, 200 * rhom)
    mass = dp.m_NFW_delta(r200c, c200c * r200m / r200c, r200m, rhom, Delta=200)

    return mass - m200c

@np.vectorize
def m200c_to_m200m_cosmo(m200c, c200c, r200c, rhom, h, z):
    # these bounds should be reasonable for m200m < 1e18
    # 1e19 Msun is ~maximum for c_correa
    m200m = opt.brentq(massdiff_2m2c, m200c, 10. * m200c,
                       args=(m200c, c200c, r200c, rhom, h, z))
    r200m = tools.mass_to_radius(m200m, 200 * rhom)
    c200m = c200c * r200m / r200c

    return m200m, c200m, r200m

# ------------------------------------------------------------------------------
# End of m200c_to_m200m_cosmo()
# ------------------------------------------------------------------------------

def table_m200c_to_m200m_cosmo(m200c=m200c,
                               z=z,
                               sigma8=sigma8,
                               omegam=omegam,
                               omegav=omegav,
                               n=n,
                               h=h):
    '''
    Create a table that converts from m200c and the given cosmology to the corresponding
    halo properties m200m
    '''
    # get interpolator and coordinates for c200c
    coords = arrays_to_coords(sigma8, omegam, omegav, n, h, z, np.log10(m200c))
    c_interp = c200c_cosmo_interp(c_file=table_dir + "c200c_correa_cosmo.asdf")

    # set background densities
    rhoc = 2.755 * 10**(11.) # [h^2 M_sun / Mpc^3]
    rhom = omegam * rhoc

    c200c = c_interp(coords).reshape(sigma8.shape + omegam.shape + omegav.shape +
                                     n.shape + h.shape + z.shape + m200c.shape)
    r200c = tools.mass_to_radius(m200c, 200 * rhoc)
    
    m200m, c200m, r200m = m200c_to_m200m_cosmo(m200c=m200c.reshape(1,1,1,1,1,1,-1),
                                               c200c=c200c,
                                               r200c=r200c.reshape(1,1,1,1,1,1,-1),
                                               rhom=rhom.reshape(1,-1,1,1,1,1,1),
                                               h=h.reshape(1,1,1,1,-1,1,1),
                                               z=z.reshape(1,1,1,1,1,-1,1))

    result_info = {
        "dims": np.array(["sigma8", "omegam", "omegav", "n", "h", "z", "m200c"]),
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

# ------------------------------------------------------------------------------
# End of table_m200c_to_m200m_cosmo()
# ------------------------------------------------------------------------------

def table_c200m_correa(m200m=m200m,
                       z=z,
                       sigma8=0.821,
                       omegam=0.2793,
                       omegav=0.7207,
                       n=0.972,
                       h=0.7,
                       cpus=None):
    '''Interpolate the c200m(m200m, cosmo) relation from Correa+2015 for the given mass, z
    and cosmology to a regular grid

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
    def c_cosmo(procn, m200m_t, c200m_t, out_q):
        c_all = np.empty((1,) + m200m.shape)
        
        c_interp = interpolate.interp1d(np.log10(m200m_t[0]),
                                        c200m_t[0])
                            
        c_all[0] = c_interp(np.log10(m200m))

        out_q.put([procn, c_all])

    # --------------------------------------------------
    if cpus == None:
        cpus = multi.cpu_count()

    if cpus > 8:
        cpus = 8

    manager = multi.Manager()
    out_q = manager.Queue()

    # load tables
    af = asdf.open(table_dir + "halo_200c_to_200m.asdf")
    m200m_tab = af.tree["m200m"][:]
    c200m_tab = af.tree["c200m"][:]

    # we split along the redshift axis, only mass is not reg grid
    m200m_tab_split = np.array_split(m200m_tab, cpus, axis=-2)
    c200m_tab_split = np.array_split(c200m_tab, cpus, axis=-2)

    procs = []
    for i in range(cpus):
        process = multi.Process(target=c_cosmo,
                                args=(i,
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
        "dims": np.array(["sigma8", "omegam", "omegav", "n", "h", "z", "m200m"]),
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

# ------------------------------------------------------------------------------
# End of table_c200m_correa()
# ------------------------------------------------------------------------------

def table_c200m_correa_cosmo(m200m=m200m,
                             z=z,
                             sigma8=sigma8,
                             omegam=omegam,
                             omegav=omegav,
                             n=n,
                             h=h,
                             cpus=None):
    '''Interpolate the c200m(m200m, cosmo) relation from Correa+2015 for the given mass, z
    and cosmology to a regular grid

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
    ------
    results : dict
        dict with c and all input values

        - this dict also gets saved to c_correa_200c.asdf

    '''
    def c_cosmo(procn, m200m_t, c200m_t, out_q):
        c_all = np.empty(m200m_t.shape[:-1] + m200m.shape)
        for idx_s8 in range(m200m_t.shape[0]):
            for idx_om in range(m200m_t.shape[1]):
                for idx_ov in range(m200m_t.shape[2]):
                    for idx_n in range(m200m_t.shape[3]):
                        for idx_h in range(m200m_t.shape[4]):
                            c_interp = interpolate.interp1d(np.log10(m200m_t[idx_s8, idx_om, idx_ov,
                                                                        idx_n, idx_h, 0]),
                                                       c200m_t[idx_s8, idx_om, idx_ov, idx_n, idx_h, 0])
                            
                            
                            c_all[idx_s8, idx_om, idx_ov, idx_n, idx_h, 0] = c_interp(np.log10(m200m))

        out_q.put([procn, c_all])

    # --------------------------------------------------
    if cpus == None:
        cpus = multi.cpu_count()

    if cpus > 8:
        cpus = 8

    manager = multi.Manager()
    out_q = manager.Queue()

    # load tables
    af = asdf.open(table_dir + "halo_200c_to_200m_cosmo.asdf")
    m200m_tab = af.tree["m200m"][:]
    c200m_tab = af.tree["c200m"][:]

    # we split along the redshift axis, only mass is not reg grid
    m200m_tab_split = np.array_split(m200m_tab, cpus, axis=-2)
    c200m_tab_split = np.array_split(c200m_tab, cpus, axis=-2)

    procs = []
    for i in range(cpus):
        process = multi.Process(target=c_cosmo,
                                args=(i,
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
        "dims": np.array(["sigma8", "omegam", "omegav", "n", "h", "z", "m200m"]),
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

# ------------------------------------------------------------------------------
# End of table_c200m_correa_cosmo()
# ------------------------------------------------------------------------------

def f_stars_interp(comp='all'):
    '''
    Return the stellar fraction interpolator as a function of halo mass as found 
    by Zu & Mandelbaum (2015).

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
                                              fill_value=(0,f_stars[-1]))
    elif comp == 'cen':
        f_stars_interp = interpolate.interp1d(m_h, f_cen, bounds_error=False,
                                              fill_value=(0,f_cen[-1]))
    else:
        f_stars_interp = interpolate.interp1d(m_h, f_sat, bounds_error=False,
                                              fill_value=(0,f_sat[-1]))

    return f_stars_interp

# ------------------------------------------------------------------------------
# End of f_stars_interp()
# ------------------------------------------------------------------------------

def table_m500c_to_m200m_dmo(m500c=m500c,
                             z=z,
                             fg500c=np.linspace(0, 1, 20),
                             f_c=0.86,
                             sigma8=0.821,
                             omegam=0.2793,
                             omegab=0.0463,
                             omegav=0.7207,
                             n=0.972,
                             h=0.7,
                             fname="m500c_to_m200m_dmo.asdf",
                             cpus=None):
    '''
    Create a table that computes the DMO equivalent halo mass given the observations
    m500c & fg500c
    '''
    
    
    def m_diff(m200m_dmo, m500c, r500c, fg500c, fs200m, fc500c, c200m, z):
        # for a given halo mass, we know the concentration
        try:
            c200m_dmo = c200m(np.array([z, np.log10(m200m_dmo)]))
        except ValueError:
            print(np.array([z, np.log10(m200m_dmo)]))
        r200m_dmo = tools.mass_to_radius(m200m_dmo, 200 * omegam * rhoc)

        # this give stellar fraction & concentration
        fcen_500c = fc500c(m200m_dmo / 0.7) * m200m_dmo / m500c
        fsat_200m = fs200m(m200m_dmo / 0.7)

        # which allows us to compute the stellar fraction at r500c
        fsat_500c = dp.m_NFW(r500c, m_x=fsat_200m*m200m_dmo, c_x=f_c*c200m_dmo,
                             r_x=r200m_dmo) / m500c

        # this is NOT m500c for our DMO halo, this is our DMO halo
        # evaluated at r500c for the observations, which when scaled
        # should match the observed m500c
        m_dmo_r500c = dp.m_NFW_delta(r500c, c200m_dmo, r200m_dmo,
                                     omegam * rhoc, Delta=200)

        f500c = fg500c + fcen_500c + fsat_500c
        fb = omegab / omegam
        m500c_cor = m_dmo_r500c * (1 - fb) / (1 - f500c)
        return m500c_cor - m500c

    # --------------------------------------------------
    def calc_m_diff(procn, m500c, r500c, fg500c, fs200m,
                    fc500c, c200m, z, out_q):
        m200m_dmo = optimize(m_diff, m500c, 5. * m500c,
                             *(m500c, r500c, fg500c, fs200m,
                               fc500c, c200m, z))
        # now we have m200m_dmo, so we calculate again all the other
        # resulting variables

        # we need to tile the redshift to match m200m_dmo
        shape_final = m200m_dmo.shape
        z_tiled = np.tile(z, (1,) + shape_final[1:])
        coords = np.vstack([z_tiled.flatten(), m200m_dmo.flatten()]).T

        # calculate the concentration and reshape to match m200m_dmo
        c200m_dmo = c200m(coords).reshape(shape_final)
        r200m_dmo = tools.mass_to_radius(m200m_dmo, 200 * omegam * rhoc)

        fcen_500c = fc500c(m200m_dmo / 0.7) * m200m_dmo / m500c
        fsat_200m = fs200m(m200m_dmo / 0.7)
        fsat_500c = dp.m_NFW(r500c, m_x=fsat_200m*m200m_dmo, c_x=f_c*c200m_dmo,
                            r_x=r200m_dmo) / m500c
        
        out_q.put([procn, m200m_dmo, fcen_500c, fsat_500c])

    # --------------------------------------------------
    if cpus == None:
        cpus = multi.cpu_count()

    manager = multi.Manager()
    out_q = manager.Queue()

    # reshape variables to match shapes
    (z_r, m500c_r, fg500c_r) = arrays_to_ogrid(z, m500c, fg500c)
    fgas_500c = fg500c_r * omegab / omegam
    fcen_500c = f_stars_interp(comp="cen")
    fsat_200m = f_stars_interp(comp="sat")

    # set background densities
    rhoc = 2.755 * 10**(11.) # [h^2 M_sun / Mpc^3]
    
    r500c_r = tools.mass_to_radius(m500c_r, 500 * rhoc)

    # otherwise the code gets upset when passing empty arrays to optimize
    if cpus > m500c.shape[0]:
        cpus = m500c.shape[0]

    m500c_split = np.array_split(m500c_r, cpus, axis=-2)
    r500c_split = np.array_split(r500c_r, cpus, axis=-2)
    c200m = c200m_interp(c_file=table_dir + "c200m_correa.asdf")

    procs = []
    for i, (mi, ri) in enumerate(zip(m500c_split, r500c_split)):
        process = multi.Process(target=calc_m_diff,
                                args=(i, mi, ri, fgas_500c, fsat_200m,
                                      fcen_500c, c200m, z_r, out_q))

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

    result_info = {
        "dims": np.array(["z", "m500c", "fg500c", "m200m_dmo"]),
        "sigma8": sigma8,
        "omegam": omegam,
        "omegab": omegab,
        "omegav": omegav,
        "n": n,
        "h": h,
        "z": z,
        "m500c": m500c,
        "fgas_500c": fg500c,
        "fcen_500c": fcen_500c,
        "fsat_500c": fsat_500c,
        "m200m_dmo": m200m_dmo
    }

    af = asdf.AsdfFile(result_info)
    af.write_to(table_dir + fname)
    af.close()

    return result_info

# ------------------------------------------------------------------------------
# End of table_m500c_to_m200m_dmo()
# ------------------------------------------------------------------------------

def table_m500c_to_m200m_dmo_cosmo(m500c=m500c,
                                   z=z,
                                   f500c=np.linspace(0, 1, 20),
                                   sigma8=sigma8,
                                   omegam=omegam,
                                   omegab=omegab,
                                   omegav=omegav,
                                   n=n,
                                   h=h,
                                   fname="m500c_to_m200m_dmo_cosmo.asdf",
                                   cpus=None):
    '''
    Create a table that computes the DMO equivalent halo mass given the observations
    m500c & f500c
    '''
    
    
    def m_diff(m200m_dmo, m500c, r500c, f500c, c200m,
               z, sigma8, omegam, omegab, omegav, n, h):
        # for a given halo mass, we know the concentration
        try:
            c200m_dmo = c200m(np.array([sigma8, omegam, omegav, n, h, z, np.log10(m200m_dmo)]))
        except ValueError:
            print(np.array([sigma8, omegam, omegav, n, h, z, np.log10(m200m_dmo)]))
        r200m_dmo = tools.mass_to_radius(m200m_dmo, 200 * omegam * rhoc)

        # this is NOT m500c for our DMO halo, this is our DMO halo
        # evaluated at r500c for the observations, which when scaled
        # should match the observed m500c
        m_dmo_r500c = dp.m_NFW_delta(r500c, c200m_dmo, r200m_dmo,
                                     omegam * rhoc, Delta=200)

        m500c_cor = m_dmo_r500c * (1 - omegab / omegam) / (1 - f500c)
        return m500c_cor - m500c

    # --------------------------------------------------
    def calc_m_diff(procn, m500c, r500c, f500c, c200m, z, sigma8, omegam, omegab,
                    omegav, n, h, out_q):
        m200m_dmo = optimize(m_diff, m500c, 5. * m500c,
                             *(m500c, r500c, f500c, c200m, z, sigma8, omegam,
                               omegab, omegav, n, h))

        out_q.put([procn, m200m_dmo])

    # --------------------------------------------------
    if cpus == None:
        cpus = multi.cpu_count()

    manager = multi.Manager()
    out_q = manager.Queue()

    # reshape variables to match shapes
    (sigma8_r, omegam_r, omegav_r,
     n_r, h_r, z_r, m500c_r, f500c_r) = arrays_to_ogrid(sigma8,
                                                        omegam,
                                                        omegav,
                                                        n, h, z,
                                                        m500c,
                                                        f500c)
    fb_500c = f500c_r * omegab / omegam_r

    # set background densities
    rhoc = 2.755 * 10**(11.) # [h^2 M_sun / Mpc^3]
    
    r500c_r = tools.mass_to_radius(m500c_r, 500 * rhoc)

    # otherwise the code gets upset when passing empty arrays to optimize
    if cpus > m500c.shape[0]:
        cpus = m500c.shape[0]

    m500c_split = np.array_split(m500c_r, cpus, axis=-2)
    r500c_split = np.array_split(r500c_r, cpus, axis=-2)
    c200m_cosmo = c200m_cosmo_interp(c_file=table_dir + "c200m_correa_cosmo.asdf")

    procs = []
    for i, (mi, ri) in enumerate(zip(m500c_split, r500c_split)):
        process = multi.Process(target=calc_m_diff,
                                args=(i, mi, ri,
                                      fb_500c, c200m_cosmo, z_r,
                                      sigma8_r, omegam_r, omegab,
                                      omegav_r, n_r, h_r, out_q))

        procs.append(process)
        process.start()

    results = []
    for i in range(len(m500c_split)):
        results.append(out_q.get())

    # need to sort results
    results.sort()
    m200m_dmo = np.concatenate([item[1] for item in results], axis=-2)

    result_info = {
        "dims": np.array(["sigma8", "omegam", "omegav", "n", "h", "z",
                          "m500c", "f500c", "m200m_dmo"]),
        "sigma8": sigma8,
        "omegam": omegam,
        "omegab": omegab,
        "omegav": omegav,
        "n": n,
        "h": h,
        "z": z,
        "m500c": m500c,
        "f500c": f500c,
        "m200m_dmo": m200m_dmo
    }

    af = asdf.AsdfFile(result_info)
    af.write_to(table_dir + fname)
    af.close()

    return result_info

# ------------------------------------------------------------------------------
# End of table_m500c_to_m200m_dmo_cosmo()
# ------------------------------------------------------------------------------

def table_m500c_to_m200m_obs(m500c=m500c,
                             z=z,
                             f500c=np.linspace(0, 1, 20),
                             r_c=np.linspace(0.05, 0.5, 20),
                             beta=np.linspace(0, 2, 20),
                             gamma=np.linspace(0, 3, 20),
                             sigma8=0.821,
                             omegam=0.2793,
                             omegab=0.0463,
                             omegav=0.7207,
                             n=0.972,
                             h=0.7,
                             fname="m500c_to_m200m_obs.asdf",
                             cpus=None):
    """
    Calculate the table linking each input m500c & observed gas profile to its
    halo radius r200m_obs
    """
    def diff_m200m(r):
        m200m = 4. /3 * np.pi * 200 * cosmo.rho_m * r**3
        m_diff = m_f(r, **kwargs) - m200m
        return m_diff

    def r200m_from_m(procn, m500c, r500c, f500c, rc, beta, gamma, c200m,
                     z, out_q):
        # m_gas = lambda r: dp.m_beta_plaw_uni(r, m500c, r500c, rc, beta, r, gamma)
        # m_stars = lambda r: dp.m_NFW(r, )
        # m_dm = lambda r: dp.m_NFW()

        # m_b = lambda r:
        pass

    # --------------------------------------------------
    if cpus == None:
        cpus = multi.cpu_count()

    manager = multi.Manager()
    out_q = manager.Queue()

    # reshape variables to match shapes
    (z_r, m500c_r, f500c_r, rc_r, b_r, g_r) = arrays_to_ogrid(z, m500c, f500c,
                                                              r_c, beta, gamma)

    f_stars = d.f_stars(m200m_dmo / 0.7, 'all')

    fb_500c = f500c_r * omegab / omegam

    # set background densities
    rhoc = 2.755 * 10**(11.) # [h^2 M_sun / Mpc^3]
    
    r500c_r = tools.mass_to_radius(m500c_r, 500 * rhoc)

    # otherwise the code gets upset when passing empty arrays to optimize
    if cpus > m500c.shape[0]:
        cpus = m500c.shape[0]

    m500c_split = np.array_split(m500c_r, cpus, axis=-2)
    r500c_split = np.array_split(r500c_r, cpus, axis=-2)
    c200m_cosmo = c200m_interp(c_file=table_dir + "c200m_correa.asdf")
    m200m_dmo = m200m_dmo_interp(m_file=table_dir + "m500c_to_m200m_dmo.asdf")

    procs = []
    for i, (mi, ri) in enumerate(zip(m500c_split, r500c_split)):
        process = multi.Process(target=calc_m_diff,
                                args=(i, mi, ri,
                                      fb_500c, m200m_dmo, z_r, out_q))

        procs.append(process)
        process.start()

    results = []
    for i in range(len(m500c_split)):
        results.append(out_q.get())

    # need to sort results
    results.sort()
    m200m_obs = np.concatenate([item[1] for item in results], axis=-2)

    result_info = {
        "dims": np.array(["z", "m500c", "f500c", "r_c", "beta", "gamma", "m200m_obs"]),
        "sigma8": sigma8,
        "omegam": omegam,
        "omegab": omegab,
        "omegav": omegav,
        "n": n,
        "h": h,
        "z": z,
        "m500c": m500c,
        "f500c": f500c,
        "r_c": r_c,
        "beta": beta,
        "gamma": gamma,
        "m200m_obs": m200m_obs
    }

    af = asdf.AsdfFile(result_info)
    af.write_to(table_dir + fname)
    af.close()

    return result_info

# ------------------------------------------------------------------------------
# End of table_m500c_to_m200m_obs()
# ------------------------------------------------------------------------------

def table_m500c_fstar_500c(m500c=m500c,
                           z=z,
                           f500c=np.linspace(0, 1, 20),
                           sigma8=0.821,
                           omegam=0.2793,
                           omegab=0.0463,
                           omegav=0.7207,
                           n=0.972,
                           h=0.7,
                           f_c=0.86,
                           fname="m500c_fstar_500c.asdf",
                           cpus=None):
    '''
    Create a table that computes the central and satellite fractions at r500c 
    given the observations m500c & f500c
    '''
    
    
    def calc_fstar_500c(procn, m500c, r500c, f500c, c200m, z, out_q):
        m200m_dmo = optimize(m_diff, m500c, 5. * m500c,
                             *(m500c, r500c, f500c, c200m, z))

        out_q.put([procn, m200m_dmo])

    # --------------------------------------------------
    if cpus == None:
        cpus = multi.cpu_count()

    manager = multi.Manager()
    out_q = manager.Queue()

    # reshape variables to match shapes
    (z_r, m500c_r, f500c_r) = arrays_to_ogrid(z, m500c, f500c)
    fb_500c = f500c_r * omegab / omegam

    # now we calculate the DMO equivalent halo mass and the stellar fractions
    m200m_dmo = interp(m200m_dmo_interp(m_file=table_dir + "m500c_to_m200m_dmo.asdf"),
                       z, m500c, f500c)

    # m200m_dmo has higher dimensionality, need to fill z to match correctly
    # then flatten and reshape final output
    coords = np.vstack([np.tile(z.reshape(-1,1,1),
                                (1,) + m200m_dmo.shape[1:]).flatten(),
                        np.log10(m200m_dmo).flatten()]).T
    c200m_dmo = c200m_interp(c_file=table_dir + "c200m_correa.asdf")(coords)
    c200m_dmo = c200m_dmo.reshape(m200m_dmo.shape)
    c200m_sat = f_c * c200m_dmo
    
    # f_stars takes values in actual units with h=0.7!
    # do not need to convert result, since independent of h in our model
    fcen_500c = d.f_stars(m200m_dmo / 0.7, comp='cen')
    fsat_200m = d.f_stars(m200m_dmo / 0.7, comp='sat')
    
    # set background density
    rhoc = 2.755 * 10**(11.) # [h^2 M_sun / Mpc^3]

    # calculate r500c for the satellite fraction conversion
    r500c_r = tools.mass_to_radius(m500c_r, 500 * rhoc)

    # otherwise the code gets upset when passing empty arrays to optimize
    if cpus > m500c.shape[0]:
        cpus = m500c.shape[0]

    m500c_split = np.array_split(m500c_r, cpus, axis=-2)
    r500c_split = np.array_split(r500c_r, cpus, axis=-2)

    procs = []
    for i, (mi, ri) in enumerate(zip(m500c_split, r500c_split)):
        process = multi.Process(target=calc_m_diff,
                                args=(i, mi, ri,
                                      fb_500c, c200m, z_r, out_q))

        procs.append(process)
        process.start()

    results = []
    for i in range(len(m500c_split)):
        results.append(out_q.get())

    # need to sort results
    results.sort()
    m200m_dmo = np.concatenate([item[1] for item in results], axis=-2)

    result_info = {
        "dims": np.array(["z", "m500c", "f500c", "m200m_dmo"]),
        "sigma8": sigma8,
        "omegam": omegam,
        "omegab": omegab,
        "omegav": omegav,
        "n": n,
        "h": h,
        "z": z,
        "m500c": m500c,
        "f500c": f500c,
        "m200m_dmo": m200m_dmo
    }

    af = asdf.AsdfFile(result_info)
    af.write_to(table_dir + fname)
    af.close()

    return result_info

# ------------------------------------------------------------------------------
# End of table_m500c_fstar_500c()
# ------------------------------------------------------------------------------


def table_rho_gas(m500c=m500c,
                  m200m_obs=m200m,
                  f500c=np.linspace(0, 1, 100),
                  r_c=np.linspace(0.05, 0.5, 50),
                  beta=np.linspace(0, 2, 50),
                  gamma=np.linspace(0, 3, 50),
                  fname="gamma_m500c.asdf",
                  cpus=None):
    """
    Calculate the table linking each input m500c & observed gas profile to its
    halo radius r200m_obs
    """
    pass

    
def extrapolate_plaw(x_range, func, verbose=False):
    '''
    Extrapolate func NaN values as a powerlaw. Works best if power law behaviour
    is already apparent, extrapolates from largest change/bump in func.

    Parameters
    ----------
    x_range : array
      range for func
    func : array
      function where np.nan will be extrapolated

    Returns
    -------
    func : array
      function with np.nan extrapolated as power law
    '''
    def plaw_pos(x, slope):
        return slope * x

    def plaw_neg(x, slope):
        return np.power(x, slope)

    # find largest change in func, will extrapolate from there
    idx_xs = np.argmin(np.diff(func[~np.isnan(func)], axis=-1))
    idx_nan = np.argmax(np.isnan(func), axis=-1) - 1

    if idx_nan != 0:
        x_fit = x_range[~np.isnan(func)]/x_range[idx_xs]
        func_fit = func[~np.isnan(func)]/func[idx_xs]

        x_fit = x_fit[...,idx_xs:]
        func_fit = func_fit[...,idx_xs:]
        if (func_fit < 0).any():
            slope, cov = opt.curve_fit(plaw_neg,
                                       (x_fit).astype(float),
                                       (func_fit).astype(float))
        else:
            slope, cov = opt.curve_fit(plaw_pos,
                                       np.log10(x_fit).astype(float),
                                       np.log10(func_fit).astype(float))

        func[idx_nan:] = func[idx_nan] * \
                         (x_range[idx_nan:]/x_range[idx_nan])**slope
    if verbose: print('Power law slope: %f'%slope)
    return func

# ------------------------------------------------------------------------------
# End of extrapolate_plaw()
# ------------------------------------------------------------------------------

def _taylor_expansion_multi(n, r_range, profile, cpus):
    '''
    Computes the Taylor coefficients for the profile expansion for n_range.

        F_n = 1 / (2n+1)! int_r r^(2n+2) * profile[M,r]

    Parameters
    ----------
    n : int
      number of Taylor coefficients to compute
    r_range : (m,r) array
      radius range to integrate over
    profile : array
      density profile with M along axis 0 and r along axis 1
    cpus : int
      number of cpus to use

    Returns
    -------
    taylor_coefs : (m,k,n) array
      array containing Taylor coefficients of Fourier expansion
    '''
    def _taylor_expansion(procn, n_range, r, profile, out_q):
        '''
        Parameters
        ----------
        procn : int
          process id
        n_range : array
          array containing the index of the Taylor coefficients
        r : array
          radius range to integrate over
        profile : array
          density profile with M along axis 0 and r along axis 1
        out_q : queue
          queue to output results
        '''
        # (m,n) array
        F_n = np.empty((profile.shape[0],) + n_range.shape,dtype=np.longdouble)
        r = np.longdouble(r)

        for idx, n in enumerate(n_range):
            prefactor = 1./spec.factorial(2*n+1, exact=True)
            result = prefactor * intg.simps(y=np.power(r, (2.0*n+2)) *
                                            profile,
                                            x=r,
                                            axis=1,
                                            even='first')

            F_n[:,idx] = result

        results = [procn,F_n]
        out_q.put(results)
        return
    # --------------------------------------------------------------------------
    manager = multi.Manager()
    out_q = manager.Queue()

    taylor = np.arange(0,n+1)
    # Split array in number of CPUs
    taylor_split = np.array_split(taylor,cpus)

    # Start the different processes
    procs = []

    for i in range(cpus):
        process = multi.Process(target=_taylor_expansion,
                                args=(i, taylor_split[i],
                                      r_range,
                                      profile,
                                      out_q))
        procs.append(process)
        process.start()

    # Collect all results
    result = []
    for i in range(cpus):
        result.append(out_q.get())

    result.sort()
    taylor_coefs = np.concatenate([item[1] for item in result],
                                  axis=-1)

    # Wait for all worker processes to finish
    for p in procs:
        p.join()

    return taylor_coefs

# ------------------------------------------------------------------------------
# End of taylor_expansion_multi()
# ------------------------------------------------------------------------------

def ft_taylor(k_range, r_range, rho_r, n=84, cpus=4, extrap=True,
              taylor_err=1e-50):
    '''
    Computes the Fourier transform of the density profile, using a Taylor
    expansion of the sin(kr)/(kr) term. We have

        u[M,k] = sum_n (-1)^n F_n[M] k^(2n)

    Returns
    -------
    u : (m,k) array
      Fourier transform of density profile
    '''
    def F_n(r_range, rho_r, n, cpus):
        '''
        Computes the Taylor coefficients in the Fourier expansion:

            F_n[M] = 4 * pi * 1 / (2n+1)! int_r r^(2n+2) * profile[M,r] dr

        Returns
        -------
        F_n : (m,n+1) array
          Taylor coefficients of Fourier expansion
        '''
        # Prefactor only changes along axis 0 (Mass)
        prefactor = (4.0 * np.pi)

        # F_n is (m,n+1) array
        F_n = _taylor_expansion_multi(n=n, r_range=r_range,
                                      profile=rho_r,
                                      cpus=cpus)
        F_n *= prefactor

        return F_n
    # --------------------------------------------------------------------------
    # define shapes for readability
    n_s = n
    m_s = r_range.shape[0]
    k_s = k_range.shape[0]

    Fn = F_n(r_range, rho_r, n, cpus)
    # need (1,n+1) array to match F_n
    n_arr = np.arange(0,n_s+1,dtype=np.longdouble).reshape(1,n_s+1)
    # -> (m,n) array
    c_n = np.power(-1,n_arr) * Fn

    # need (k,n+1) array for exponent
    k_n = np.power(np.tile(np.longdouble(k_range).reshape(k_s,1),
                           (1,n_s+1)),
                   (2 * n_arr))

    # need to match n terms and sum over them
    # result is (k,m) array -> transpose
    T_n = c_n.reshape(1,m_s,n_s+1) * k_n.reshape(k_s,1,n_s+1)
    u = np.sum(T_n,axis=-1).T

    # k-values which do not converge anymore will have coefficients
    # that do not converge to zero. Convergence to zero is determined
    # by taylor_err.
    indices = np.argmax((T_n[:,:,-1] > taylor_err), axis=0)
    indices[indices == 0] = k_s
    for idx, idx_max in enumerate(indices):
        u[idx,idx_max:] = np.nan
        # this extrapolation is not really very good...
        if (idx_max != k_s) and extrap:
            u[idx] = extrapolate_plaw(k_range, u[idx])

    # # normalize spectrum so that u[k=0] = 1, otherwise we get a small
    # # systematic offset, while we know that theoretically u[k=0] = 1
    # if (np.abs(u[:,0]) - 1. > 1.e-2).any():
    #     print('-------------------------------------------------',
    #           '! Density profile mass does not match halo mass !',
    #           '-------------------------------------------------',
    #           sep='\n')

    # nonnil = (u[:,0] != 0)
    # u[nonnil] = u[nonnil] / u[nonnil,0].reshape(-1,1)

    return u

# ------------------------------------------------------------------------------
# End of ft_taylor()
# ------------------------------------------------------------------------------

def profile_beta(r_range, m_x, r_x, r_y, rc, beta):
    '''
    Return a beta profile with mass m_x inside r_range <= r_x

        rho[r] =  rho_c[m_x, rc, r_x] / (1 + ((r/r_x)/rc)^2)^(beta / 2)

    rho_c is determined by the mass of the profile.

    Parameters
    ----------
    r_range : (m,r) array
      array containing r_range for each m
    m_x : (m,) array
      array containing masses to match at r_x
    r_x : (m,) array
      x overdensity radius to match m_x at, in units of r_range
    r_y : (m,) array
      cutoff radius for profile
    beta : (m,) array
      power law slope of profile
    rc : (m,) array
      physical core radius of beta profile in as a fraction

    Returns
    -------
    profile : (m,r) array
      array containing beta profile
    '''
    m = m_x.shape[0]

    # analytic enclosed mass inside r_x gives normalization rho_0
    rho_0 = m_x / (4./3 * np.pi * r_x**3 * spec.hyp2f1(3./2, 3. * beta / 2,
                                                       5./2, -(r_x / rc)**2))

    rc = rc.reshape(m,1)
    beta = beta.reshape(m,1)
    r_x = r_x.reshape(m,1)
    r_y = r_y.reshape(m,1)
    m_x = m_x.reshape(m,1)
    rho_0 = rho_0.reshape(m,1)

    profile = rho_0 / (1 + (r_range / rc)**2)**(3*beta/2)
    profile[r_range > r_y] = 0.

    return profile

# ------------------------------------------------------------------------------
# End of profile_beta()
# ------------------------------------------------------------------------------

def profile_uni(r_range, rho_x, r_x, r_y):
    '''
    Return a uniform profile with density rho_x at r_x between r_x and r_y

        rho[r] =  rho_x for r_x <= r <= y

    Parameters
    ----------
    r_range : (m,r) array
      array containing r_range for each m
    rho_x : (m,) array
      array containing densities to match at r_x
    r_x : (m,) array
      x overdensity radius to match rho_x at, in units of r_range
    r_y : (m,) array
      y radius, in units of r_range
    
    Returns
    -------
    profile : (m,r) array
      array containing uniform profile
    '''
    m = r_x.shape[0]

    r_x = r_x.reshape(m,1)
    r_y = r_y.reshape(m,1)
    rho_x = rho_x.reshape(m,1)

    profile = rho_x * np.ones_like(r_range)
    profile[((r_range < r_x) | (r_range > r_y))] = 0.

    return profile

# ------------------------------------------------------------------------------
# End of profile_uni()
# ------------------------------------------------------------------------------

def profile_uni_k(k_range, rho_x, r_x, r_y):
    '''
    Return the analytic 3D radially symmetric FT of a uniform profile with density 
    rho_x at r_x between r_x and r_y

    Parameters
    ----------
    k_range : (k,) array
      array containing k_range
    rho_x : (m,) array
      array containing densities to match at r_x
    r_x : (m,) array
      x overdensity radius to match rho_x at, in units of r_range
    r_y : (m,) array
      y radius, in units of r_range
    
    Returns
    -------
    profile_k : (m,k) array
      array containing uniform profile_k
    '''
    k_range = k_range.reshape(1,-1)
    rho_x = rho_x.reshape(-1,1)
    r_x = r_x.reshape(-1,1)
    r_y = r_y.reshape(-1,1)

    krx = k_range * r_x
    kry = k_range * r_y

    sincoskry = np.sin(kry) / kry**3 - np.cos(kry) / (kry)**2
    sincoskrx = np.sin(krx) / krx**3 - np.cos(krx) / (krx)**2

    profile_k = 4 * np.pi * rho_x * (r_y**3 * sincoskry - r_x**3 * sincoskrx)
    
    return profile_k

# ------------------------------------------------------------------------------
# End of profile_uni_k()
# ------------------------------------------------------------------------------

def profile_beta_plaw_uni(r_range, m_x, r_x, rc, beta, r_y, gamma,
                          rho_x=None):
    '''
    Return a beta profile with mass m_x inside r_range <= r_x

        rho[r] =  rho_c[m_x, rc, r_x] / (1 + ((r/r_x)/rc)^2)^(beta / 2)

    and a power law outside

        rho[r] = rho_x (r/r_x)^(-gamma)

    rho_c is determined by the mass of the profile.

    Parameters
    ----------
    r_range : (m,r) array
      array containing r_range for each m
    m_x : (m,) array
      array containing masses to match at r_x
    r_x : (m,) array
      x overdensity radius to match m_x at, in units of r_range
    rc : (m,) array
      physical core radius of beta profile in as a fraction
    beta : (m,) array
      power law slope of profile
    r_y : (m,) array
      radius out to which power law holds
    gamma : (m,) array
      power law index

    Returns
    -------
    profile : (m,r) array
      array containing beta profile
    '''
    m = m_x.shape[0]

    # analytic enclosed mass inside r_x gives normalization rho_0
    rho_0 = m_x / (4./3 * np.pi * r_x**3 * spec.hyp2f1(3./2, 3 * beta / 2,
                                                       5./2, -(r_x / rc)**2))

    rc = rc.reshape(m,1)
    beta = beta.reshape(m,1)
    r_x = r_x.reshape(m,1)
    m_x = m_x.reshape(m,1)
    r_y = r_y.reshape(m,1)
    rho_0 = rho_0.reshape(m,1)

    if rho_x is None:
        rho_x = profile_beta(r_x, m_x=m_x, r_x=r_x, r_y=np.tile(np.inf, (m, 1)),
                             rc=rc, beta=beta)

    rho_x = rho_x.reshape(m,1)
    profile = np.zeros_like(r_range)
    for idx, r in enumerate(r_range):
        # create slices for the different profiles
        sl_beta = (r <= r_x[idx])
        sl_plaw = ((r > r_x[idx]) & (r <= r_y[idx]))
        sl_uni = (r > r_y[idx])
        profile[idx][sl_beta] = rho_0[idx] / (1 + (r[sl_beta] / rc[idx])**2)**(3*beta[idx]/2)
        profile[idx][sl_plaw] = rho_x[idx] * (r[sl_plaw]/r_x[idx])**(-gamma[idx])
        profile[idx][sl_uni] = rho_x[idx] * (r_y[idx] / r_x[idx])**(-gamma[idx])

    return profile

# ------------------------------------------------------------------------------
# End of profile_beta_plaw_uni()
# ------------------------------------------------------------------------------

def profile_beta_plaw_uni_k(k_range, fgas500c, rc, beta, gamma):
    '''
    Calculate 
    '''
    r_range = np.logspace(-2, 1, 200)

    return

# ------------------------------------------------------------------------------
# End of profile_beta_plaw_uni_k()
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

def m200m_dmo_interp(m_file=table_dir + "m500c_to_m200m_dmo.asdf"):
    '''
    Return the interpolator for the c200m(m200m) relation
    '''
    af = asdf.open(m_file)

    z = af.tree["z"][:]
    m = af.tree["m500c"][:]
    f = af.tree["f500c"][:]
    m200m_dmo = af.tree["m200m_dmo"][:]

    coords = (z, np.log10(m), f)
    print(m200m_dmo.shape)

    m200m_dmo_interp = interpolate.RegularGridInterpolator(coords, m200m_dmo)

    return m200m_dmo_interp
    
# ------------------------------------------------------------------------------
# End of m200m_dmo_interp()
# ------------------------------------------------------------------------------
