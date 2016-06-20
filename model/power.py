import numpy as np
import matplotlib.pyplot as plt

import halo.parameters as p
from halo.tools import Integrate
import halo.model.component as comp
from halo.model._cache import Cache, cached_property, parameter

import pdb

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
    def __init__(self, components, p_lin):
        super(Power, self).__init__()
        self.comps = {}
        for comp in components:
            self.comps[comp.name] = comp

        self.p_lin = p_lin

    #===========================================================================
    # Parameters
    #===========================================================================
    @parameter
    def comps(self, val):
        return val

    @parameter
    def p_lin(self, val):
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

        # dndlnm = comp_1.m_fn.dndlnm.reshape(m,1)
        nu = comp_1.nu.reshape(m,1)
        fnu = comp_1.fnu.reshape(m,1)
        r_x = comp_1.r_range[:,-1].reshape(m,1)
        m_range = comp_1.m_range.reshape(m,1)
        f_comp_1 = comp_1.f_comp.reshape(m,1)
        f_comp_2 = comp_2.f_comp.reshape(m,1)

        prefactor = 4./3 * np.pi * 200.
        # prefactor = 1./(p.prms.rho)
        result = Integrate(y=fnu * f_comp_1 * comp_1.rho_k *
                           f_comp_2 * comp_2.rho_k * r_x**3,
                           x=nu,
                           axis=0)
        # # print (f_comp_1.reshape(-1) * comp_1.rho_k[:,0] *
        # #        f_comp_2.reshape(-1) *
        # #        comp_2.rho_k[:,0] *
        # #        (r_x**3).reshape(-1))
        # try:
        #     print Integrate(fnu.reshape(-1) * (f_comp_1.reshape(-1) *
        #                                        comp_1.rho_k[:,0] *
        #                                        f_comp_2.reshape(-1) *
        #                                        comp_2.rho_k[:,0] *
        #                                        (r_x**3).reshape(-1)),
        #                     nu,
        #                     axis=0)

        #     plt.plot(nu.reshape(-1), fnu.reshape(-1) * (f_comp_1.reshape(-1) *
        #                               comp_1.rho_k[:,0] *
        #                               f_comp_2.reshape(-1) *
        #                               comp_2.rho_k[:,0] *
        #                               (r_x**3).reshape(-1)))
        #     plt.xscale('log')
        #     plt.yscale('log')
        #     plt.show()
        # except:
        #     pass

        result *= prefactor

        return result

    # @staticmethod
    # def _cross_2halo(comp_1, comp_2):
    #     '''
    #     Compute the 2-halo cross-correlation between components comp_1 and
    #     comp_2
    #     '''
    #     # define shapes for readability
    #     m = comp_1.m_range.shape[0]
    #     k = comp_1.k_range.shape[0]

    #     # dndlnm = comp_1.m_fn.dndlnm.reshape(m,1)
    #     nu = comp_1.nu.reshape(m,1)
    #     fnu = comp_1.fnu.reshape(m,1)

    #     f_comp_1 = comp_1.f_comp.reshape(m,1)
    #     f_comp_2 = comp_2.f_comp.reshape(m,1)
    #     bias_1 = comp_1.bias.reshape(m,1)
    #     bias_2 = comp_2.bias.reshape(m,1)

    #     prefactor = comp_1.p_lin
    #     result = (Integrate(y=fnu * f_comp_1 * comp_1.rho_k * bias_1,
    #                         x=nu,
    #                         axis=0) *
    #               Integrate(y=fnu * f_comp_2 * comp_2.rho_k * bias_2,
    #                         x=nu,
    #                         axis=0))
    #     result *= prefactor

    #     return result

    @staticmethod
    def _cross_power(comp_1, comp_2):
        '''
        Compute the cross-correlation between comp_1 and comp_2.
        '''
        # P_2h = Power._cross_2halo(comp_1, comp_2)
        P_1h = Power._cross_1halo(comp_1, comp_2)
        P_tot = P_1h #+ P_2h

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

    @cached_property('cross_P', 'comps', 'p_lin')
    def P_tot(self):
        '''
        Return total power including correlations
        '''
        rho = p.prms.rho_m
        P_tot = self.p_lin
        for key, comp in self.comps.iteritems():
            k_range = comp.k_range

            P_tot += comp.P_1h

        for key, cross_comp in self.cross_P.iteritems():
            c1, c2 = key.split('-')

            k_range = self.comps[c1].k_range

            P_cross = 2 * cross_comp
            P_tot += P_cross

        return P_tot

    @cached_property('cross_delta', 'comps', 'p_lin')
    def delta_tot(self):
        '''
        Return total dimensionless power including correlations
        '''
        D_tot = 0.
        for key, comp in self.comps.iteritems():
            k_range = comp.k_range
            D_tot += comp.Delta_1h

        for key, cross_comp in self.cross_delta.iteritems():
            c1, c2 = key.split('-')

            D_cross = 2 * cross_comp
            D_tot += D_cross

        D_lin = 0.5 / np.pi**2 * k_range**3 * self.p_lin
        D_tot += D_lin

        return D_tot
