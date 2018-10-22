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
    m_range : (m,) array
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
    def __init__(self, m500c,
                 m200m_dmo=None,
                 # r_min needs to be low in order to get correct masses from
                 # integration
                 r_min=-4, r_bins=1000,
                 k_min=-1.8, k_max=2., k_bins=1000,
                 cosmo=hmf.cosmo.Cosmology(**{'sigma_8': 0.821,
                                              'H0': 70.0,
                                              'omegab': 0.0463,
                                              'omegac': 0.233,
                                              'omegam': 0.0463 + 0.233,
                                              'omegav': 0.7207,
                                              'n': 0.972}),
                 z=0.):
        super(Parameters, self).__init__()
        self.m500c = m500c
        if m200m_dmo is None:
            self.m200m_dmo = tools.m500c_to_m200m_duffy(m500c, cosmo.rho_crit, cosmo.rho_m)
        else:
            self.m200m_dmo = m200m_dmo
        self.r_min = r_min
        self.r_bins = r_bins
        self.k_min = k_min
        self.k_max = k_max
        self.k_bins = k_bins
        self.cosmo = cosmo
        self.z = z

    #===========================================================================
    # Parameters
    #===========================================================================
    @parameter
    def m500c(self, val):
        return val

    @parameter
    def m200m_dmo(self, val):
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
    def cosmo(self, val):
        return val

    @parameter
    def z(self, val):
        if val >= 0:
            return val
        else:
            raise ValueError('z needs to be >= 0')

    #===========================================================================
    # Methods
    #===========================================================================
    @cached_property('m500c', 'cosmo')
    def r500c(self):
        return tools.mass_to_radius(self.m500c, 500 * self.cosmo.rho_crit)

    @cached_property('m200m_dmo', 'cosmo')
    def r200m_dmo(self):
        return tools.mass_to_radius(self.m200m_dmo, self.cosmo.rho_m * 200)

    # @cached_property('m200m', 'cosmo')
    # def m200c(self):
    #     return np.array([tools.m200m_to_m200c_duffy(m, self.cosmo.rho_crit,
    #                                                 self.cosmo.rho_m)
    #                      for m in self.m200m])

    # @cached_property('m200c', 'cosmo')
    # def r200c(self):
    #     return tools.mass_to_radius(self.m200c, 200 * self.cosmo.rho_crit)

    # @cached_property('m200m', 'cosmo', 'm200c')
    # def m500c(self):
    #     return np.array([tools.m200m_to_m500c_duffy(m, self.cosmo.rho_crit,
    #                                                 self.cosmo.rho_m)
    #                      for m in self.m200m])

    # @cached_property('m200m', 'cosmo', 'r200m')
    # def c200m(self):
    #     '''
    #     The density profiles always assume cosmology dependent variables
    #     '''
    #     return tools.c_duffy(self.m200m).reshape(-1)

    # @cached_property('m200m', 'cosmo')
    # def r200m(self):
    #     return tools.mass_to_radius(self.m200m, self.cosmo.rho_m * 200)

    # @cached_property('m200m', 'cosmo')
    # def m200c(self):
    #     return np.array([tools.m200m_to_m200c_correa(m, self.cosmo.rho_crit,
    #                                                  self.cosmo.rho_m,
    #                                                  self.cosmo.h)
    #                      for m in self.m200m])

    # @cached_property('m200c', 'cosmo')
    # def r200c(self):
    #     return tools.mass_to_radius(self.m200c, 200 * self.cosmo.rho_crit)

    # @cached_property('m200m', 'cosmo', 'm200c')
    # def m500c(self):
    #     return np.array([tools.m200m_to_m500c_correa(mm, self.cosmo.rho_crit,
    #                                                  self.cosmo.rho_m,
    #                                                  self.cosmo.h, mc)
    #                      for mm, mc in zip(self.m200m, self.m200c)])

    # @cached_property('m500c', 'cosmo')
    # def r500c(self):
    #     return tools.mass_to_radius(self.m500c, 500 * self.cosmo.rho_crit)

    # @cached_property('m200c', 'cosmo', 'r200c', 'r200m')
    # def c200m(self):
    #     '''
    #     The density profiles always assume cosmology dependent variables
    #     '''
    #     return (np.array([tools.c_correa(m, h=self.cosmo.h)
    #                      for m in self.m200c]).reshape(-1)
    #             * self.r200m / self.r200c)

    @cached_property('k_min', 'k_max', 'k_bins')
    def dlnk(self):
        return np.log(10) * (self.k_max - self.k_min)/np.float(self.k_bins)

    @cached_property('k_min', 'k_max', 'k_bins')
    def k_range(self):
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

# ------------------------------------------------------------------------------
# End of Parameters()
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Typical parameters for our simulations
# ------------------------------------------------------------------------------
prms1 = Parameters(m500c=np.logspace(11,12,101), k_min=-1.)
prms2 = Parameters(m500c=np.logspace(12,13,101), k_min=-1.)
prms3 = Parameters(m500c=np.logspace(13,14,101), k_min=-1.)
prms4 = Parameters(m500c=np.logspace(14,15,101), k_min=-1.)

prms = Parameters(m500c=np.logspace(11,15,101), k_min=-1.)
