import numpy as np
import matplotlib.pyplot as plt

import halo.hmf as hmf
from astropy.cosmology import WMAP9
import halo.tools as tools
from halo.tools import Integrate
import halo.model.density as density
import pdb

class Power(object):
    '''
    Object containing info on the total matter power spectrum of all matter
    components. Components should have same range properties.

    Parameters
    ----------
    prof : density.Profile()
      a density profile to compute the power spectrum for
    hmf_prms : dict
      a dictionary with the parameters for the hmf.Massfunction class

    Methods
    -------

    '''
    def __init__(self, m200m_dmo, m200m_obs, prof,
                 hmf_prms={'transfer_model': hmf.transfer_models.FromFile,
                           'transfer_params': {'fname': 'camb/wmap9_transfer_out.dat'},
                           'cosmo_model': WMAP9,
                           'hmf_model': hmf.fitting_functions.Tinker08},
                 bar2dmo=True):
        super(Power, self).__init__()
        self.m200m_dmo = m200m_dmo
        self.m200m_obs = m200m_obs
        self.profile = prof
        self.m_h = prof.m_h
        try:
            self.k_nan = np.min(np.where(np.isnan(prof.rho_k))[1])
        except ValueError:
            self.k_nan = prof.k_range.shape[0]
        self.k_range = prof.k_range[:self.k_nan]
        self.rho_k = prof.rho_k[:,:self.k_nan]
        self.z = prof.z
        self.cosmo = prof.cosmo
        self.bar2dmo = bar2dmo

        self.hmf_prms = hmf_prms
        self.hmf_prms['cosmo_params'] = prof.cosmo.astropy_dict
        self.hmf_prms['n'] = prof.cosmo.n
        self.hmf_prms['sigma_8'] = prof.cosmo.sigma_8
        self.hmf_prms['lnk_min'] = np.log(np.min(self.k_range))
        self.hmf_prms['lnk_max'] = np.log(np.max(self.k_range))
        self.hmf_prms['dlnk'] = np.log(self.k_range[1]) - np.log(self.k_range[0])
        self.hmf_prms['z'] = prof.z
        # m_range needs to be m200m_dmo
        # self.hmf_prms['m_range'] = prof.m200m

    @property
    def rho_m(self):
        return self.cosmo.rho_m

    @property
    def m_fn(self):
        hmf_prms = self.hmf_prms
        if self.bar2dmo:
            hmf_prms['m'] = self.m200m_dmo
        else:
            hmf_prms['m'] = self.m200m_obs

        m_fn = hmf.MassFunction(**hmf_prms)
        return m_fn

    @property
    def p_lin(self):
        '''
        Return linear power spectrum
        '''
        return self.m_fn.power

    @property
    def delta_lin(self):
        '''
        Return linear dimensionless power spectrum
        '''
        return 0.5 / np.pi**2 * self.k_range**3 * self.p_lin

    # @property
    # def dndm(self):
    #     return self.m_fn.dndm

    @property
    def dndm(self):
        return self.m_fn.dndm

    @property
    def p_1h(self):
        '''
        Return the 1h power spectrum P_1h(k)

        P_1h = int_m_h (m_h/rho_m)^2 n(m_dmo(m_h)) |rho(k|m)|^2 dm_h
        '''
        # define shapes for readability
        m_s = self.m_h.shape[0]

        m_h = self.m_h.reshape(m_s,1)
        dndm = self.dndm.reshape(m_s,1)

        prefactor = (1. / self.rho_m)**2
        result = tools.Integrate(y=dndm * np.abs(self.rho_k)**2,
                                 x=m_h, axis=0)

        result *= prefactor

        return result

    @property
    def delta_1h(self):
        '''
        Return 1h dimensionless power
        '''
        return 0.5 / np.pi**2 * self.k_range**3 * self.p_1h


    @property
    def p_tot(self):
        '''
        Return total power spectrum
        '''
        return self.p_lin + self.p_1h

    @property
    def delta_tot(self):
        '''
        Return total dimensionless power spectrum
        '''
        return 0.5 / np.pi**2 * self.k_range**3 * self.p_tot
