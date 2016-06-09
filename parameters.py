'''
Main definitions to ease use of the package

TODO
----
add methods to Parameters docstring
'''

import numpy as np
import scipy.interpolate as interp
import hmf

import halo.tools as tools
from halo.model._cache import Cache, cached_property, parameter

import pdb

class Parameters(Cache):
    '''
    An object containing all parameters for the halo model.

    Parameters
    ----------
    m_min : float
      Minimum mass at which to perform analysis [units log10 Msun h^-1]
    m_max : float
      Maximum mass at which to perform analysis [units log10 Msun h^-1]
    m_bins : int
      Number of mass bins to compute mass function for
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
    p_lin_file : string
      Location of linear matter power spectrum file
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

    Methods
    -------
    dlog10m : float
      log10m mass interval for mass function

    '''
    def __init__(self, m_min=10, m_max=15, m_bins=101,
                 r_min=-4.0, r_bins=1000,
                 k_min=-1.8, k_max=2., k_bins=1000,
                 sigma_8=0.821, H0=70.0, omegab=0.0463, omegac=0.233,
                 omegav=0.7207, n=0.972,
                 # transfer_fit='EH',
                 # transfer_options=None,
                 # transfer_fit='FromFile',
                 # transfer_options={'fname': 'camb/wmap9_transfer_out.dat'},
                 z=0.,
                 p_lin_file='HMcode/plin.dat',
                 fnu_file='HMcode/nu_fnu.dat',
                 delta_h=200.,
                 # mf_fit='Tinker10', cut_fit=False,
                 # delta_wrt='mean', delta_c=1.686,
                 delta_c=1.686,
                 rho_crit=2.7763458 * (10.0**11.0)):
        super(Parameters, self).__init__()
        self.m_min = m_min
        self.m_max = m_max
        self.m_bins = m_bins
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
        # self.transfer_fit = transfer_fit
        # self.transfer_options = transfer_options
        self.p_lin_file = p_lin_file
        self.fnu_file = fnu_file
        self.z = z
        # self.mf_fit = mf_fit
        # self.cut_fit = cut_fit
        self.delta_h = delta_h
        # self.delta_wrt = delta_wrt
        self.delta_c = delta_c
        self.rho_crit = rho_crit

    #===========================================================================
    # Parameters
    #===========================================================================
    @parameter
    def m_min(self, val):
        return val

    @parameter
    def m_max(self, val):
        return val

    @parameter
    def m_bins(self, val):
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

    # @parameter
    # def transfer_fit(self, val):
    #     return val

    # @parameter
    # def transfer_options(self, val):
    #     return val

    @parameter
    def p_lin_file(self, val):
        return val

    @parameter
    def fnu_file(self, val):
        return val

    @parameter
    def z(self, val):
        if val >= 0:
            return val
        else:
            raise ValueError('z needs to be >= 0')

    # @parameter
    # def mf_fit(self, val):
    #     return val

    # @parameter
    # def cut_fit(self, val):
    #     return val

    @parameter
    def delta_h(self, val):
        return val

    # @parameter
    # def delta_wrt(self, val):
    #     return val

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
    @cached_property('omegab', 'omegac')
    def omegam(self):
        return self.omegab + self.omegac

    # @cached_property('m_min', 'm_max', 'm_bins')
    # def dlog10m(self):
    #     return (self.m_max - self.m_min)/np.float(self.m_bins)

    @cached_property('m_min', 'm_max', 'm_bins')
    def m_range_lin(self):
        return np.logspace(self.m_min, self.m_max, self.m_bins)

    # @cached_property('m_range')
    # def m_range_lin(self):
    #     return np.power(10, self.m_range)

    # @cached_property('k_min', 'k_max', 'k_bins')
    # def dlnk(self):
    #     return (self.k_max - self.k_min)/np.float(self.k_bins)

    @cached_property('k_min', 'k_max', 'k_bins')
    def k_range_lin(self):
        return np.logspace(self.k_min, self.k_max, self.k_bins)

    # @cached_property('k_range')
    # def k_range_lin(self):
    #     return np.exp(self.k_range)

    @cached_property('sigma_8', 'H0', 'omegab', 'omegac', 'omegav', 'n')
    def cosmo_prms(self):
        return {
            "sigma_8": self.sigma_8,
            "H0": self.H0,
            "omegab": self.omegab,
            "omegac": self.omegac,
            "omegav": self.omegav,
            "n": self.n}

    # @cached_property('k_min', 'k_max', 'dlnk', 'z', 'transfer_fit',
    #                  'transfer_options', 'cosmo_prms')
    # def trans_prms(self):
    #     trans = {
    #         "lnk_min": self.k_min,
    #         "lnk_max": self.k_max,
    #         "dlnk": self.dlnk,
    #         "transfer_fit": self.transfer_fit,
    #         "transfer_options": self.transfer_options,
    #         "z": self.z}
    #     trans = tools.merge_dicts(self.cosmo_prms, trans)

    #     return trans

    # @cached_property('m_min', 'm_max', 'dlog10m', 'mf_fit', 'delta_h', 'cut_fit',
    #                  'delta_wrt', 'delta_c', 'trans_prms')
    # def m_fn_prms(self):
    #     massf = {
    #         "Mmin": self.m_min,
    #         "Mmax": self.m_max,
    #         "dlog10m": self.dlog10m,
    #         "mf_fit": self.mf_fit,
    #         "cut_fit": self.cut_fit,
    #         "delta_h": self.delta_h,
    #         "delta_wrt": self.delta_wrt,
    #         "delta_c": self.delta_c}
    #     massf = tools.merge_dicts(massf, self.trans_prms)

    #     return massf

    # @cached_property('m_fn_prms')
    # def m_fn(self):
    #     return hmf.MassFunction(**self.m_fn_prms)

    @cached_property('p_lin_file', 'k_range_lin')
    def p_lin(self):
        k, P = np.loadtxt(self.p_lin_file, unpack=True)
        p_int = interp.interp1d(k, P)
        plin = p_int(self.k_range_lin)
        return plin

    @cached_property('fnu_file')
    def nu(self):
        nu, fnu = np.loadtxt(self.fnu_file, unpack=True)
        return nu

    @cached_property('fnu_file')
    def fnu(self):
        nu, fnu = np.loadtxt(self.fnu_file, unpack=True)
        return fnu

    # @cached_property('m_range_lin', 'm_fn_prms')
    # def rho_m(self):
    #     return tools.Integrate(self.m_fn.dndlnm, self.m_range_lin)
    @cached_property('omegab', 'omegac', 'rho_crit', 'h')
    def rho_m(self):
        return (self.omegab + self.omegac) * self.rho_crit

    @cached_property('omegab', 'omegac')
    def f_dm(self):
        return 1 - self.omegab/self.omegac

    @cached_property('f_dm', 'rho_m')
    def rho_dm(self):
        return self.f_dm * self.rho_m

    @cached_property('m_range_lin', 'rho_m', 'delta_h')
    def r_h(self):
        return tools.mass_to_radius(self.m_range_lin, self.rho_m * self.delta_h)

    @cached_property('r_min', 'r_h', 'r_bins')
    def r_range_lin(self):
        return np.array([np.logspace(self.r_min, np.log10(r_max), self.r_bins)
                         for r_max in self.r_h])

    # @cached_property('r_range')
    # def r_range_lin(self):
    #     return np.power(10, self.r_range)

    @cached_property('H0')
    def h(self):
        return self.H0 / 100.

# ------------------------------------------------------------------------------
# End of Parameters()
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Typical parameters for our simulations
# ------------------------------------------------------------------------------
prms1 = Parameters(m_min=10., m_max=15.)
prms2 = Parameters(m_min=11., m_max=15.)
prms3 = Parameters(m_min=12., m_max=15.)
prms4 = Parameters(m_min=13., m_max=15.)
prms5 = Parameters(m_min=14., m_max=15.)

prms = Parameters(m_min=6.5, m_max=15.5)
# Power spectrum converges for lower cutoff m_min = 6.5
# Power spectrum converges for upper cutoff m_max = 15.5
