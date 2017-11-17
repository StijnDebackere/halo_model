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
    name : str
      name for this Power instance

    Methods
    -------

    '''
    def __init__(self, components, name):
        super(Power, self).__init__()
        self.name = name
        self.comps = {}
        self.r_range = components[0].r_range
        self.r200m = components[0].r200m
        self.m200m = components[0].m200m
        self.k_range = components[0].k_range
        self.dndm = components[0].dndm
        self.p_lin = components[0].p_lin
        for comp in components:
            self.comps[comp.name] = comp
            r_range = comp.r_range
            r200m = comp.r200m
            m200m = comp.m200m
            k_range = comp.k_range
            dndm = comp.dndm
            p_lin = comp.p_lin
            if not (np.allclose(r_range, self.r_range) or
                    np.allclose(r200m, self.r200m) or
                    np.allclose(m200m, self.m200m) or
                    np.allclose(k_range, self.k_range) or
                    np.allclose(dndm, self.dndm) or
                    np.allclose(p_lin, self.p_lin)):
                raise AttributeError('m200m/r200m/r_range/k_range/p_lin/dndm need to be equal')


    #===========================================================================
    # Parameters
    #===========================================================================
    @parameter
    def name(self, val):
        return val

    @parameter
    def comps(self, val):
        return val

    @parameter
    def r_range(self, val):
        return val

    @parameter
    def r200m(self, val):
        return val

    @parameter
    def m200m(self, val):
        return val

    @parameter
    def k_range(self, val):
        return val

    @parameter
    def dndm(self, val):
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
        m = comp_1.m200m.shape[0]
        k = comp_1.k_range.shape[0]

        r200m = comp_1.r200m.reshape(m,1)
        m200m = comp_1.m200m.reshape(m,1)
        dndm = comp_1.dndm.reshape(m,1)

        f_comp_1 = comp_1.f_comp.reshape(m,1)
        f_comp_2 = comp_2.f_comp.reshape(m,1)

        prefactor = (4./3 * np.pi * 200)**2
        result = Integrate(y=dndm * f_comp_1 * comp_1.rho_k *
                           f_comp_2 * comp_2.rho_k * r200m**6,
                           x=m200m,
                           axis=0)

        result *= prefactor

        return result

    # @staticmethod
    # def _cross_2halo(comp_1, comp_2):
    #     '''
    #     Compute the 2-halo cross-correlation between components comp_1 and
    #     comp_2
    #     '''
    #     # define shapes for readability
    #     m = comp_1.m200m.shape[0]
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
        # p_2h = Power._cross_2halo(comp_1, comp_2)
        p_1h = Power._cross_1halo(comp_1, comp_2)
        p_tot = p_1h #+ p_2h

        return p_tot

    @cached_property('comps')
    def cross_p(self):
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

    @cached_property('cross_p')
    def cross_delta(self):
        '''
        Compute dimensionless power spectrum
        '''
        k_range = self.comps[self.comps.keys()[0]].k_range
        cross_d = {}
        for key, item in self.cross_p.iteritems():
            cross_d[key] = 1./(2*np.pi**2) * k_range**3 * item

        return cross_d

    @cached_property('p_lin', 'comps')
    def delta_lin(self):
        '''
        Return linear delta
        '''
        k_range = self.comps.values()[0].k_range

        return 0.5 / np.pi**2 * k_range**3 * self.p_lin

    @cached_property('cross_p', 'comps', 'p_lin')
    def p_tot(self):
        '''
        Return total power including correlations
        '''
        rho = p.prms.rho_m
        p_tot = self.p_lin
        for key, comp in self.comps.iteritems():
            k_range = comp.k_range

            p_tot += comp.p_1h

        for key, cross_comp in self.cross_p.iteritems():
            c1, c2 = key.split('-')

            k_range = self.comps[c1].k_range

            p_cross = 2 * cross_comp
            p_tot += p_cross

        return p_tot

    @cached_property('cross_delta', 'comps', 'p_lin')
    def delta_tot(self):
        '''
        Return total dimensionless power including correlations
        '''
        d_tot = 0.
        for key, comp in self.comps.iteritems():
            k_range = comp.k_range
            d_tot += comp.delta_1h

        for key, cross_comp in self.cross_delta.iteritems():
            c1, c2 = key.split('-')

            d_cross = 2 * cross_comp
            d_tot += d_cross

        d_lin = 0.5 / np.pi**2 * k_range**3 * self.p_lin
        d_tot += d_lin

        return d_tot
