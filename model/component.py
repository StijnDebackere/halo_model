import numpy as np
import hmf
import scipy.special as spec
import halo.density_profiles as profs
import halo.parameters as p
from halo.tools import Integrate
import halo.model.density as dens
from halo.model._cache import cached_property, parameter

import pdb

class Component(dens.Profile):
    '''
    An object containing information about a matter component.

    Parameters
    ----------
    name : str
      name of the component
    rho_m : float
      mean matter density of the universe
    m : (m,) array
      halo masses
    p_lin : (k,) array
      linear power spectrum
    dndm : (m,) array
      halo mass function, adjusted for lower m200m in baryon case
    f_comp : (m,) array
      fractional contribution of component to halo mass m200m for an object of
      equivalent mass m at redshift z
    bias_fn : function or array
      function to compute the bias (should have first arg for m200m) or array
      containing profile with m along axis 0
    bias_fn_args : dict
      dictionary containing the additional arguments to the bias function
    profile_kwargs : keywords
      keyword arguments are passed to density.Profile class

    Methods
    -------
    p_1h : (k,) array
      1 halo term of power spectrum
    p_2h : (k,) array
      2 halo term of power spectrum
    p_tot : (k,) array
      power spectrum
    delta_1h : (k,) array
      1 halo term of dimensionless power spectrum
    delta_2h : (k,) array
      2 halo term of dimensionless power spectrum
    delta_tot: (k,) array
      dimensionless power spectrum
    '''
    def __init__(self, name, rho_m, m, p_lin, dndm, f_comp,
                 # bias_fn, bias_fn_args,
                 **profile_kwargs):
        super(Component, self).__init__(**profile_kwargs)
        self.name = name
        self.rho_m = rho_m
        self.m = m
        self.p_lin = p_lin
        self.f_comp = f_comp
        self.dndm = dndm
        # self.bias_fn = bias_fn
        # self.bias_fn_args = bias_fn_args

    def __add__(self, other):
        if not (np.allclose(self.dndm, other.dndm) or
                np.allclose(self.p_lin, other.p_lin)):
            raise AttributeError('dndm/p_lin need to be the same')
        if not (np.allclose(self.r_range, other.r_range) or
                np.allclose(self.rho_m, other.rho_m) or
                np.allclose(self.m, other.m) or
                np.allclose(self.k_range, other.k_range)):
            raise AttributeError('r_range/rho_m/m/k_range need to be the same')

        prof1 = self.rho_r
        prof2 = other.rho_r

        prof1_f = self.rho_k
        prof2_f = other.rho_k

        f_comp1 = self.f_comp
        comp1_mask = (self.f_comp == 0.)
        f_comp2 = other.f_comp
        comp2_mask = (other.f_comp == 0.)
        f_new = f_comp1 + f_comp2

        prof_new = 1. / f_new.reshape(-1,1) * (f_comp1.reshape(-1,1) * prof1 +
                                               f_comp2.reshape(-1,1) * prof2)
        prof_f_new = 1. / f_new.reshape(-1,1) * (f_comp1.reshape(-1,1) * prof1_f +
                                                 f_comp2.reshape(-1,1) * prof2_f)

        profile_kwargs = {"r_range": self.r_range,
                          "m_tot": self.m_tot,
                          "k_range": self.k_range,
                          "profile": prof_new,
                          "f_comp": f_new,
                          "profile_f": prof_f_new}
        return Component(self.name, self.rho_m, self.m, self.p_lin,
                         self.dndm,
                         #self.bias_fn,
                         # self.bias_fn_args,
                         **profile_kwargs)

    #===========================================================================
    # Parameters
    #===========================================================================
    @parameter
    def name(self, val):
        return val

    @parameter
    def rho_m(self, val):
        return val

    @parameter
    def m(self, val):
        return val

    @parameter
    def p_lin(self, val):
        return val

    @parameter
    def dndm(self, val):
        return val

    # @parameter
    # def bias_fn(self, val):
    #     return val

    # @parameter
    # def bias_fn_args(self, val):
    #     return val

    #===========================================================================
    # Methods
    #===========================================================================
    @cached_property('m', 'f_comp')
    def m200m(self):
        return self.m * self.f_comp

    @cached_property('k_range', 'p_lin')
    def delta_lin(self):
        return 0.5/np.pi**2 * self.k_range**3 * self.p_lin

    @cached_property('f_comp', 'm', 'dndm')
    def rho_comp(self):
        '''
        Compute the average density of the component as a fraction of rho_m

            rho_comp(z) = int_m m n(m,z) f_comp(m,z)
            rho_comp(z) / rho_m = int_nu f(nu, z) f_comp(nu, z)
        '''
        norm = Integrate(y=self.m * self.dndm, x=self.m, axis=0)
        result = Integrate(y=self.m * self.dndm * self.f_comp,
                           x=self.dndm,
                           axis=0) / norm

        return result

    # @cached_property('bias_fn', 'bias_fn_args', 'm', 'nu', 'fnu')
    # def bias(self):
    #     if hasattr(self.bias_fn,'__call__'):
    #         b = self.bias_fn(self.m,
    #                          **self.bias_fn_args)
    #     else:
    #         b = self.bias_fn

    #     if len(b.shape) != 1:
    #         raise ValueError('bias_fn should be an (m,) array. ')

    #     else:
    #         # correct bias such that p_2h -> p_lin for k -> 0
    #         # this correction assumes Simpson integration!!!
    #         # pdb.set_trace()
    #         bias_int = Integrate(y=self.fnu * b * self.f_comp,
    #                              x=self.nu,
    #                              axis=0)
    #         diff = self.rho_comp - bias_int
    #         # diff < 0 will cause strange features at high k in 2 halo term
    #         # if a high m_max is chosen. This is of no importance, since 1 halo
    #         # term dominates there.
    #         if diff < 0:
    #             print('-------------------------------------------------------',
    #                   '! 2-halo term wrong for large k, but 1-halo dominates  ',
    #                   '! for component %s                                     '\
    #                   %(self.name),
    #                   '-------------------------------------------------------',
    #                   sep='\n')


    #         # notation follows scipy.integrate.simps source code
    #         h = np.diff(self.nu)
    #         h0 = h[0]
    #         h1 = h[1]
    #         hsum = h0 + h1
    #         h0divh1 = h0/np.float(h1)

    #         if (self.fnu[0] * self.f_comp[0]) != 0:
    #             b[0] += (diff * 6./(hsum * (2 - 1.0/h0divh1)) *
    #                      1./(self.fnu[0] * self.f_comp[0]))


    #         bias_int = Integrate(y=self.fnu * b * self.f_comp,
    #                              x=self.nu,
    #                              axis=0)
    #         diff = self.rho_comp - bias_int
    #         return b

    @cached_property('rho_r', 'r_range', 'f_comp')
    def m_h(self):
        '''
        Compute mass of the integrated density profile.

        Returns
        -------
        m_h : (m,) array
          array containing halo mass for each profile
        '''
        m_s = self.m.shape[0]

        f_comp = self.f_comp.reshape(m_s,1)
        return 4*np.pi * Integrate(self.rho_r * f_comp * self.r_range**2,
                                   self.r_range,
                                   axis=1)

    @cached_property('rho_m', 'm', 'k_range', 'rho_k', 'f_comp', 'dndm')
    def p_1h(self):
        '''
        Compute the 1-halo term of the power spectrum.

        P_1h = int_m (m/rho_m)^2 n(m_dmo(m)) f(m)^2 |u(k|m)|^2 dm
        '''
        # define shapes for readability
        m_s = self.m.shape[0]

        rho_m = self.rho_m
        m = self.m.reshape(m_s,1)
        f_comp = self.f_comp.reshape(m_s,1)

        dndm = self.dndm.reshape(m_s,1)

        prefactor = rho_m**(-2)
        result = Integrate(y=m**2 * dndm * f_comp**2 * np.abs(self.rho_k)**2,
                           x=m,
                           axis=0)
        result *= prefactor

        return result

    @cached_property('m', 'k_range', 'p_lin')
    def p_2h(self):
        '''
        Compute the 2-halo term of the power spectrum.
        '''
        # define shapes for readability
        m_s = self.m.shape[0]
        k_s = self.k_range.shape[0]

        m = self.m.reshape(m_s,1)
        f_comp = self.f_comp.reshape(m_s,1)

        prefactor = self.p_lin
        # result = (Integrate(y=fnu * f_comp * self.rho_k * bias,
        #                     x=nu,
        #                     axis=0))**2
        # result *= prefactor
        result = prefactor

        return result

    @cached_property('p_1h', 'p_2h')
    def p_tot(self):
        return self.p_1h + self.p_2h

    @cached_property('k_range', 'p_1h')
    def delta_1h(self):
        '''
        Return the dimensionless power spectrum for 1-halo term of component

               delta_1h[k] = 1/(2*pi^2) * k^3 p_1h(k)
        '''
        return 1./(2*np.pi**2) * self.k_range**3 * self.p_1h

    @cached_property('k_range', 'p_1h')
    def delta_2h(self):
        '''
        Return the dimensionless power spectrum for 2-halo term of component

               delta_2h[k] = 1/(2*pi^2) * k^3 p_2h(k)
        '''
        return 1./(2*np.pi**2) * self.k_range**3 * self.p_2h

    @cached_property('delta_1h', 'delta_2h')
    def delta_tot(self):
        return self.delta_1h + self.delta_2h

# ------------------------------------------------------------------------------
# End of Component()
# ------------------------------------------------------------------------------
