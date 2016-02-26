import numpy as np

import halo.parameters as p
from halo.tools import Integrate
import halo.model.component as comp
from halo.model._cache import Cache, cached_property, parameter

class Power(Cache):
    '''
    Object containing info on the total matter power spectrum of all matter
    components. Components should have same range properties.

    Parameters
    ----------
    components : list
      list of components
      

    Methods
    -------
    
    '''
    def __init__(self, components):
        super(Power, self).__init__()
        self.comps = {}
        for comp in components:
            self.comps[comp.name] = comp
            
    #===========================================================================
    # Parameters
    #===========================================================================    
    @parameter
    def comps(self, val):
        return val

    #===========================================================================
    # Methods
    #===========================================================================    
    @staticmethod
    def _cross_1halo(comp_1, comp_2):
        '''
        Compute the 1-halo cross-correlation between comp_1 and comp_2
        '''
        # define shapes for readability
        m = comp_1.m_range.shape[0]
        k = comp_1.k_range.shape[0]

        dndlnm = comp_1.m_fn.dndlnm.reshape(m,1)
        m_range = comp_1.m_range.reshape(m,1)
        f_comp_1 = comp_1.f_comp.reshape(m,1)
        f_comp_2 = comp_2.f_comp.reshape(m,1)

        prefactor = 1./(comp_1.rho_comp * comp_2.rho_comp)
        result = Integrate(y=dndlnm * f_comp_1 * comp_1.rho_k *
                           f_comp_2 * comp_2.rho_k * m_range,
                           x=m_range,
                           axis=0)
        result *= prefactor

        return result
        
    @staticmethod
    def _cross_2halo_cc(comp_1, comp_2):
        '''
        Compute the 2-halo cross-correlation between components comp_1 and
        comp_2
        '''
        # define shapes for readability
        m = comp_1.m_range.shape[0]
        k = comp_1.k_range.shape[0]
        
        dndlnm = comp_1.m_fn.dndlnm.reshape(m,1)
        m_range = comp_1.m_range.reshape(m,1)
        f_comp_1 = comp_1.f_comp.reshape(m,1)
        f_comp_2 = comp_2.f_comp.reshape(m,1)
        bias_1 = comp_1.bias.reshape(m,1)
        bias_2 = comp_2.bias.reshape(m,1)

        prefactor = (1./(comp_1.rho_comp * comp_2.rho_comp) *
                     np.exp(comp_1.m_fn.power))
        result = (Integrate(y=dndlnm * f_comp_1 * comp_1.rho_k * bias_1,
                            x=m_range,
                            axis=0) *
                  Integrate(y=dndlnm * f_comp_2 * comp_2.rho_k * bias_2,
                            x=m_range,
                            axis=0))
        result *= prefactor

        return result
        
    @staticmethod
    def _cross_2halo_dc(comp_1, comp_2):
        '''
        Compute the 2-halo cross-correlation between diffuse and normal 
        components comp_1 and comp_2
        '''
        # define shapes for readability
        m = comp_2.m_range.shape[0]
        k = comp_2.k_range.shape[0]
        
        dndlnm = comp_2.m_fn.dndlnm.reshape(m,1)
        m_range = comp_2.m_range.reshape(m,1)
        f_comp_2 = comp_2.f_comp.reshape(m,1)
        bias_2 = comp_2.bias.reshape(m,1)

        prefactor = comp_1.bias * comp_1.P_lin
        result = 1./comp_2.rho_comp * Integrate(y=dndlnm * bias_2 *
                                                f_comp_2 * comp_2.rho_k,
                                                x=m_range,
                                                axis=0)
        result *= prefactor

        return result
        
    @staticmethod
    def _cross_2halo_dd(comp_1, comp_2):
        '''
        Compute the 2-halo cross-correlation between diffuse components
        comp_1 and comp_2
        '''
        return (comp_1.bias * comp_1.P_lin * comp_2.bias * comp_2.P_lin)

    @staticmethod
    def _cross_power(comp_1, comp_2):
        '''
        Compute the cross-correlation between comp_1 and comp_2.
        '''
        if comp_1.__class__ == comp.DiffuseComponent:
            if comp_2.__class__ == comp.DiffuseComponent:
                P_2h = Power._cross_2halo_dd(comp_1, comp_2)

            elif comp_2.__class__ == comp.Component:
                P_2h = Power._cross_2halo_dc(comp_1, comp_2)

            P_tot = P_2h

        elif comp_1.__class__ == comp.Component:
            if comp_2.__class__ == comp.DiffuseComponent:
                P_2h = Power._cross_2halo_dc(comp_2, comp_1)
                P_tot = P_2h

            elif comp_2.__class__ == comp.Component:
                P_2h = Power._cross_2halo_cc(comp_1, comp_2)
                P_1h = Power._cross_1halo(comp_1, comp_2)
                P_tot = P_2h + P_1h

        return P_tot

    @cached_property('comps')
    def cross_P(self):
        '''
        Compute all cross-correlation terms between different components.
        '''
        cross = {}
        keys = self.comps.keys()
        for idx, key_1 in enumerate(keys):
            comp_1 = self.comps[key_1]

            for key_2 in keys[idx+1:]:
                comp_2 = self.comps[key_2]

                cross_name = '{:s}-{:s}'.format(comp_1.name, comp_2.name)
                cross[cross_name] = Power._cross_power(comp_1, comp_2)

        return cross

    @cached_property('cross_P')
    def cross_delta(self):
        '''
        Compute dimensionless power spectrum
        '''
        k_range = self.comps[self.comps.keys()[0]].k_range
        cross_d = {}
        for key, item in self.cross_P.iteritems():
            cross_d[key] = 1./(2*np.pi**2) * k_range**3 * item

        return cross_d

    @cached_property('cross_P', 'comps')
    def P_tot(self):
        '''
        Return total power including correlations
        '''
        rho = p.prms.rho_m
        P_tot = 0.
        for key, comp in self.comps.iteritems():
            k_range = comp.k_range
            rho_c = comp.rho_comp

            P = rho_c**2/rho**2 * comp.P_tot
            P_tot += P

        for key, cross_comp in self.cross_P.iteritems():
            c1, c2 = key.split('-')

            k_range = self.comps[c1].k_range
            rho_c1 = self.comps[c1].rho_comp
            rho_c2 = self.comps[c2].rho_comp

            P_cross = 2 * (rho_c1 * rho_c2) / rho**2 * cross_comp
            P_tot += P_cross

        return P_tot

    @cached_property('cross_delta', 'comps')
    def delta_tot(self):
        '''
        Return total dimensionless power including correlations
        '''
        rho = p.prms.rho_m
        D_tot = 0.
        for key, comp in self.comps.iteritems():
            k_range = comp.k_range
            rho_c = comp.rho_comp

            D = rho_c**2/rho**2 * comp.Delta_tot
            D_tot += D

        for key, cross_comp in self.cross_delta.iteritems():
            c1, c2 = key.split('-')

            k_range = self.comps[c1].k_range
            rho_c1 = self.comps[c1].rho_comp
            rho_c2 = self.comps[c2].rho_comp

            D_cross = 2 * (rho_c1 * rho_c2) / rho**2 * cross_comp
            D_tot += D_cross

        return D_tot
