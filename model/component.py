from __future__ import print_function

import numpy as np
import hmf

import halo.density_profiles as profs
from halo.tools import Integrate
import halo.model.density as dens
from halo.model._cache import Cache, cached_property, parameter

import pdb

class Component(dens.Profile):
    '''
    An object containing information about a matter component.
    
    Parameters
    ----------
    name : str
      name of the component
    m_fn : hmf.MassFunction object
      the halo mass function
    profile_kwargs : keywords
      keyword arguments are passed to density.Profile class
    f_comp : (m,) array
      fractional contribution of component to total mass for an object of
      equivalent mass m at redshift z
    bias_fn : function or array
      function to compute the bias (should have first arg for m_range) or array 
      containing profile with m along axis 0
    bias_fn_args : dict
      dictionary containing the additional arguments to the bias function
    
    Methods
    -------
    P_1h : (k,) array
      1 halo term of power spectrum
    P_2h : (k,) array
      2 halo term of power spectrum
    P_tot : (k,) array
      power spectrum
    Delta_1h : (k,) array
      1 halo term of dimensionless power spectrum
    Delta_2h : (k,) array
      2 halo term of dimensionless power spectrum
    Delta_tot: (k,) array
      dimensionless power spectrum
    '''
    def __init__(self, name, m_fn, f_comp, bias_fn, bias_fn_args,
                 **profile_kwargs):
        super(Component, self).__init__(**profile_kwargs)
        # self.dndlnm = m_fn.dndlnm
        # self.power_lin = m_fn.power
        self.name = name
        self.m_fn = m_fn
        self.f_comp = f_comp
        self.bias_fn = bias_fn
        self.bias_fn_args = bias_fn_args

    #===========================================================================
    # Parameters
    #===========================================================================    
    # @parameter
    # def dndlnm(self, val):
    #     return val

    # @parameter
    # def power_lin(self, val):
    #     return val

    @parameter
    def name(self, val):
        return val

    @parameter
    def m_fn(self, val):
        if not isinstance(val, hmf.MassFunction):
            raise TypeError('m_fn should be hmf.MassFunction instance.')
        else: 
            return val
    
    @parameter
    def f_comp(self, val):
        return val

    @parameter
    def bias_fn(self, val):
        return val
    
    @parameter
    def bias_fn_args(self, val):
        return val

    #===========================================================================
    # Methods
    #===========================================================================    
    @cached_property('m_range', 'f_comp', 'm_fn')
    def rho_comp(self):
        '''
        Compute the average density of the component.

            rho_comp(z) = int_m m n(m,z) f_comp(m,z)
        '''
        return Integrate(y=self.m_fn.dndlnm * self.f_comp,
                         x=self.m_range,
                         axis=0)

    @cached_property('bias_fn', 'bias_fn_args', 'm_range', 'm_fn', 'rho_comp')
    def bias(self):
        if hasattr(self.bias_fn,'__call__'):
            b = self.bias_fn(self.m_range,
                             **self.bias_fn_args)
        else:
            b = self.bias_fn

        if len(b.shape) != 1:
            raise ValueError('bias_fn should be an (m,) array. ')
                
        else:
            # correct bias such that P_2h -> P_lin for k -> 0
            # this correction assumes Simpson integration!!!
            # pdb.set_trace()
            dndlnm = self.m_fn.dndlnm

            bias_int = Integrate(y=dndlnm * b * self.f_comp,
                                 x=self.m_range,
                                 axis=0)
            diff = self.rho_comp - bias_int
            # diff < 0 will cause strange features in at high k in 2 halo term
            # if a high m_max is chosen. This is of no importance, since 1 halo
            # term dominates there.
            if diff < 0:
                print('-------------------------------------------------------',
                      '! 2-halo term wrong for large k, but 1-halo dominates  ',
                      '! for component %s                                     '\
                      %(self.name),
                      '-------------------------------------------------------',
                      sep='\n')


            # notation follows scipy.integrate.simps source code
            h = np.diff(self.m_range)
            h0 = h[0]
            h1 = h[1]
            hsum = h0 + h1
            h0divh1 = h0/np.float(h1)

            if (dndlnm[0] * self.f_comp[0]) != 0:
                b[0] += (diff * 6./(hsum * (2 - 1.0/h0divh1)) *
                         1./(dndlnm[0] * self.f_comp[0]))

            
            bias_int = Integrate(y=dndlnm * b * self.f_comp,
                                 x=self.m_range,
                                 axis=0)
            diff = self.rho_comp - bias_int
            return b

    @cached_property('rho_r', 'r_range', 'f_comp')
    def m_h(self):
        '''
        Compute mass of the integrated density profile.

        Returns
        -------
        m_h : (m,) array
          array containing halo mass for each profile
        '''
        m = self.m_range.shape[0]
        
        f_comp = self.f_comp.reshape(m,1)
        return 4*np.pi * Integrate(self.rho_r * f_comp * self.r_range**2,
                                   self.r_range,
                                   axis=1)

    @cached_property('m_range', 'k_range', 'm_fn', 'rho_comp', 'rho_k')
    def P_1h(self):
        '''
        Compute the 1-halo term of the power spectrum.
        '''
        # define shapes for readability
        m = self.m_range.shape[0]
        k = self.k_range.shape[0]
        
        dndlnm = self.m_fn.dndlnm.reshape(m,1)
        m_range = self.m_range.reshape(m,1)
        f_comp = self.f_comp.reshape(m,1)

        prefactor = 1./self.rho_comp**2
        result = Integrate(y=dndlnm * f_comp**2 * np.abs(self.rho_k)**2
                           * m_range,
                           x=m_range,
                           axis=0)
        result *= prefactor

        return result
        
    @cached_property('m_range', 'k_range', 'm_fn', 'rho_comp', 'rho_k')
    def P_2h(self):
        '''
        Compute the 2-halo term of the power spectrum.
        '''
        # define shapes for readability
        m = self.m_range.shape[0]
        k = self.k_range.shape[0]
        
        dndlnm = self.m_fn.dndlnm.reshape(m,1)
        m_range = self.m_range.reshape(m,1)
        f_comp = self.f_comp.reshape(m,1)
        bias = self.bias.reshape(m,1)

        prefactor = np.exp(self.m_fn.power)
        result = 1./self.rho_comp**2 * (Integrate(y=dndlnm * f_comp *
                                                  self.rho_k * bias,
                                                  x=m_range,
                                                  axis=0))**2
        result *= prefactor

        return result

    @cached_property('P_1h', 'P_2h')
    def P_tot(self):
        return self.P_1h + self.P_2h

    @cached_property('k_range', 'P_1h')
    def Delta_1h(self):
        '''
        Return the dimensionless power spectrum for 1-halo term of component
        
               Delta[k] = 1/(2*pi^2) * k^3 P(k)
        '''
        return 1./(2*np.pi**2) * self.k_range**3 * self.P_1h

    @cached_property('k_range', 'P_1h')
    def Delta_2h(self):
        '''
        Return the dimensionless power spectrum for 2-halo term of component
        
               Delta[k] = 1/(2*pi^2) * k^3 P(k)
        '''
        return 1./(2*np.pi**2) * self.k_range**3 * self.P_2h

    @cached_property('Delta_1h', 'Delta_2h')
    def Delta_tot(self):
        return self.Delta_1h + self.Delta_2h

# ------------------------------------------------------------------------------
# End of Component()
# ------------------------------------------------------------------------------

# class DiffuseComponent(Cache):
#     '''
#     An object containing information about a diffuse matter component. This 
#     component traces the linear matter power spectrum with a bias factor.
    
#     Parameters
#     ----------
#     name : str
#       name of the component
#     m_fn : hmf.MassFunction object
#       the halo mass function
#     f_comp : (m,) array
#       fractional contribution of component to total mass for an object of
#       equivalent mass m at redshift z
#     bias : float
#       bias factor between diffuse component and linear power spectrum

#     Methods
#     -------
#     P_2h : (k,) array
#       2 halo term of power spectrum
#     Delta_2h : (k,) array
#       2 halo term of dimensionless power spectrum
#     '''
#     def __init__(self, name, m_range, k_range, m_fn, f_comp, bias):
#         super(DiffuseComponent, self).__init__()
#         self.name = name
#         self.m_range = m_range
#         self.k_range = k_range
#         self.m_fn = m_fn
#         self.f_comp = f_comp
#         self.bias = bias

#     #===========================================================================
#     # Parameters
#     #===========================================================================    
#     @parameter
#     def name(self, val):
#         return val

#     @parameter
#     def m_range(self, val):
#         return val

#     @parameter
#     def k_range(self, val):
#         return val

#     @parameter
#     def m_fn(self, val):
#         if not isinstance(val, hmf.MassFunction):
#             raise TypeError('m_fn should be hmf.MassFunction instance.')
#         else: 
#             return val

#     @parameter
#     def f_comp(self, val):
#         return val

#     @parameter
#     def bias(self, val):
#         return val
    
#     #===========================================================================
#     # Methods
#     #===========================================================================    
#     @cached_property('m_range', 'f_comp', 'm_fn')
#     def rho_comp(self):
#         '''
#         Compute the average density of the component.

#             rho_comp(z) = int_m m n(m,z) f_comp(m,z)
#         '''
#         return Integrate(y=self.m_fn.dndlnm * self.f_comp,
#                          x=self.m_range,
#                          axis=0)

#     @cached_property('m_range', 'k_range', 'm_fn')
#     def P_lin(self):
#         '''
#         Compute linear power spectrum
#         '''
#         return np.exp(self.m_fn.power)
    
#     @cached_property('P_lin', 'bias')
#     def P_2h(self):
#         '''
#         Compute the 2-halo term of the power spectrum.
#         '''
#         return np.power(self.bias, 2) * self.P_lin


#     @cached_property('k_range', 'P_1h')
#     def Delta_2h(self):
#         '''
#         Return the dimensionless power spectrum for 2-halo term of component
        
#                Delta[k] = 1/(2*pi^2) * k^3 P(k)
#         '''
#         return 1./(2*np.pi**2) * self.k_range**3 * self.P_2h
        
# # ------------------------------------------------------------------------------
# # End of DiffuseComponent()
# # ------------------------------------------------------------------------------

