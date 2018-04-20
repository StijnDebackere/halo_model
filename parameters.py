'''Main definitions to ease use of the package

TODO
----
add methods to Parameters docstring

'''

import numpy as np
import scipy.interpolate as interp
import halo.hmf as hmf

import halo.tools as tools
from halo.model._cache import Cache, cached_property, parameter

import pdb

class Parameters(Cache):
    '''
    An object containing all parameters for the halo model.

    Parameters
    ----------
    m_range : float
      Halo mass range [units Msun h^-1]
    r_min : float
      Minimum log radius [units h^-1 Mpc]
    r_bins : int
      Number of r bins
    k_min : float
      Minimum log wavenumber [units h Mpc^-1]
    k_max : float
      Maximum log wavenumber [units h Mpc^-1]
    k_bins : int
      Number of k bins to compute tranfer function for
    sigma_8 : float
      sigma_8 normalization of power spectrum
    H0 : float
      Hubble constant at present
    omegab : float
      Baryon matter density
    omegac : float
      CDM matter density
    omegav : float
      Vacuum density
    n : float
      Spectral index
    tranfer_fit : string (accepted by hmf.Transfer)
      Transfer function fit
    transfer_options : string (accepted by hmf.Transfer)
      Extra Transfer options
    z : float
      Redshift to compute for
    mf_fit : string (accepted by hmf.MassFunction)
      Fitting function for halo mass function
    cut_fit : bool
      Whether to forcibly cut f(sigma) at bounds in literature. If false,
      will use whole range of M.
    delta_h : float
      Overdensity for halo definition
    delta_wrt : str (accpected by hmf.MassFunction)
      With respect to which density is delta_h
    delta_c : float
      Linear critical overdensity for collapse
    rho_crit : float [units Msun h^2 / Mpc^3]
      Critical density of the universe

    Methods
    -------
    dlog10m : float
      log10m mass interval for mass function

    rho_m : float [units Msun h^2 / Mpc^3]
      mean matter density of the universe

    '''
    def __init__(self, m200m,
                 r_min=-4.0, r_bins=1000,
                 k_min=-1.8, k_max=2., k_bins=1000,
                 sigma_8=0.821, H0=70.0, omegab=0.0463, omegac=0.233,
                 omegav=0.7207, n=0.972,
                 transfer_fit='FromFile',
                 transfer_options={'fname': 'camb/wmap9_transfer_out.dat'},
                 z=0.,
                 delta_h=200.,
                 mf_fit='Tinker10', cut_fit=False,
                 delta_wrt='mean', delta_c=1.686,
                 rho_crit=2.7763458 * (10.0**11.0)):
        super(Parameters, self).__init__()
        self.m200m = m200m
        self.r_min = r_min
        self.r_bins = r_bins
        self.k_min = k_min
        self.k_max = k_max
        self.k_bins = k_bins
        self.sigma_8 = sigma_8
        self.H0 = H0
        self.omegab = omegab
        self.omegac = omegac
        self.omegav = omegav
        self.n = n
        self.transfer_fit = transfer_fit
        self.transfer_options = transfer_options
        self.z = z
        self.mf_fit = mf_fit
        self.cut_fit = cut_fit
        self.delta_h = delta_h
        self.delta_wrt = delta_wrt
        self.delta_c = delta_c
        self.rho_crit = rho_crit

    #===========================================================================
    # Parameters
    #===========================================================================
    @parameter
    def m200m(self, val):
        return val

    @parameter
    def r_min(self, val):
        return val

    @parameter
    def r_bins(self, val):
        return val

    @parameter
    def k_min(self, val):
        return val

    @parameter
    def k_max(self, val):
        return val

    @parameter
    def k_bins(self, val):
        return val

    @parameter
    def sigma_8(self, val):
        return val

    @parameter
    def H0(self, val):
        return val

    @parameter
    def omegab(self, val):
        return val

    @parameter
    def omegac(self, val):
        return val

    @parameter
    def omegav(self, val):
        return val

    @parameter
    def n(self, val):
        return val

    @parameter
    def transfer_fit(self, val):
        return val

    @parameter
    def transfer_options(self, val):
        return val

    @parameter
    def z(self, val):
        if val >= 0:
            return val
        else:
            raise ValueError('z needs to be >= 0')

    @parameter
    def mf_fit(self, val):
        return val

    @parameter
    def cut_fit(self, val):
        return val

    @parameter
    def delta_h(self, val):
        return val

    @parameter
    def delta_wrt(self, val):
        return val

    @parameter
    def delta_c(self, val):
        return val

    @parameter
    def rho_crit(self, val):
        if val > 0:
            return val
        else:
            raise ValueError('rho_crit needs to be > 0')

    #===========================================================================
    # Methods
    #===========================================================================
    @cached_property('m200m', 'rho_m')
    def r200m(self):
        return tools.mass_to_radius(self.m200m, self.rho_m * 200)

    @cached_property('m200m', 'rho_crit', 'rho_m')
    def m200c(self):
        return np.array([tools.m200m_to_m200c(m, self.rho_crit, self.rho_m, prms.h)
                         for m in self.m200m])

    @cached_property('m200c', 'rho_crit')
    def r200c(self):
        return tools.mass_to_radius(self.m200c, 200 * self.rho_crit)

    @cached_property('m200m', 'rho_crit', 'rho_m', 'm200c')
    def m500c(self):
        return np.array([tools.m200m_to_m500c(mm, self.rho_crit, self.rho_m, prms.h, mc)
                         for mm, mc in zip(self.m200m, self.m200c)])

    @cached_property('m500c', 'rho_crit')
    def r500c(self):
        return tools.mass_to_radius(self.m500c, 500 * self.rho_crit)

    @cached_property('m200c', 'h', 'r200c', 'r200m')
    def c_correa(self):
        '''
        The density profiles always assume cosmology dependent variables
        '''
        return (np.array([tools.c_correa(m, h=self.h)
                         for m in self.m200c]).reshape(-1)
                * self.r200m / self.r200c)

    @cached_property('omegab', 'omegac')
    def omegam(self):
        return self.omegab + self.omegac

    @cached_property('k_min', 'k_max', 'k_bins')
    def dlnk(self):
        return np.log(10) * (self.k_max - self.k_min)/np.float(self.k_bins)

    @cached_property('k_min', 'k_max', 'k_bins')
    def k_range_lin(self):
        return np.logspace(self.k_min, self.k_max, self.k_bins)

    @cached_property('sigma_8', 'H0', 'omegab', 'omegac', 'omegav', 'n')
    def cosmo_prms(self):
        return {
            "sigma_8": self.sigma_8,
            "H0": self.H0,
            "omegab": self.omegab,
            "omegac": self.omegac,
            "omegav": self.omegav,
            "n": self.n}

    @cached_property('k_min', 'k_max', 'dlnk', 'z', 'transfer_fit',
                     'transfer_options', 'cosmo_prms')
    def trans_prms(self):
        trans = {
            "lnk_min": np.log(10) * self.k_min,
            "lnk_max": np.log(10) * self.k_max,
            "dlnk": self.dlnk,
            "transfer_fit": self.transfer_fit,
            "transfer_options": self.transfer_options,
            "z": self.z}
        trans = tools.merge_dicts(self.cosmo_prms, trans)

        return trans

    @cached_property('m200m', 'mf_fit', 'delta_h', 'cut_fit',
                     'delta_wrt', 'delta_c', 'trans_prms')
    def m_fn_prms(self):
        massf = {
            "m_range": self.m200m,
            "mf_fit": self.mf_fit,
            "cut_fit": self.cut_fit,
            "delta_h": self.delta_h,
            "delta_wrt": self.delta_wrt,
            "delta_c": self.delta_c}
        massf = tools.merge_dicts(massf, self.trans_prms)

        return massf

    @cached_property('m_fn_prms')
    def m_fn(self):
        return hmf.MassFunction(**self.m_fn_prms)

    @cached_property('k_range_lin', 'm_fn')
    def p_lin(self):
        plin = np.exp(self.m_fn.power)

        return plin

    @cached_property('k_range_lin', 'm_fn', 'p_lin')
    def delta_lin(self):
        delta_lin = 1. / (2 * np.pi**2) * self.k_range_lin**3 * self.p_lin

        return delta_lin

    @cached_property('m_fn', 'm200m')
    def dndm(self):
        return self.m_fn.dndm

    @cached_property('omegab', 'omegac', 'rho_crit', 'h')
    def rho_m(self):
        return (self.omegab + self.omegac) * self.rho_crit

    @cached_property('m200m', 'dndm')
    def rho_hm(self):
        return tools.Integrate(y=self.m200m * self.dndm, x=self.m200m)

    @cached_property('omegab', 'omegam')
    def f_dm(self):
        return 1 - self.omegab/self.omegam

    @cached_property('f_dm', 'rho_m')
    def rho_dm(self):
        return self.f_dm * self.rho_m

    @cached_property('r_min', 'r200m', 'r_bins')
    def r_range_lin(self):
        return np.array([np.logspace(self.r_min, np.log10(r_max), self.r_bins)
                         for r_max in self.r200m])

    @cached_property('H0')
    def h(self):
        return self.H0 / 100.

# ------------------------------------------------------------------------------
# End of Parameters()
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Typical parameters for our simulations
# ------------------------------------------------------------------------------
prms1 = Parameters(m200m=np.logspace(11,12,101))
prms2 = Parameters(m200m=np.logspace(12,13,101))
prms3 = Parameters(m200m=np.logspace(13,14,101))
prms4 = Parameters(m200m=np.logspace(14,15,101))
prmst = Parameters(m200m=np.logspace(10,15,101))

prms = Parameters(m200m=np.logspace(11,15,101),
                  k_min=-1.8, k_max=2, k_bins=1000)
