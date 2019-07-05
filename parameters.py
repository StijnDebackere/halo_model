'''Main definitions for ease of use of the package

'''
import numpy as np

import halo.tools as tools
import halo.cosmo as cosmo
import halo.input.interpolators as inp_interp

import pdb

class Parameters(object):
    '''
    An object containing all parameters for the halo model.

    Parameters
    ----------
    m500c : (m,) array
      Halo mass range [units Msun h^-1]
    r_min : float
      Minimum log radius [units h^-1 Mpc]
    r_bins : int
      Number of r bins
    logk_min : float
      Minimum log wavenumber [units h Mpc^-1]
    logk_max : float
      Maximum log wavenumber [units h Mpc^-1]
    k_bins : int
      Number of k bins to compute tranfer function for
    cosmo : hmf.Cosmology object
      cosmological parameters to compute for
    z_range : float or array
      Redshift to compute for
    fgas500c_prms : dict
      "log10mt" : turnover mass for the fgas500c-m500c relation
      "a" : sharpness of turnover
      "fgas_500c_max" : interpolator for the maximum gas fraction at 
                        r500c as function of z and m500c
    f_c : float
      ratio between satellite concentration and DM concentration
    sigma_lnc : float
      logarithmic offset to take the c(m) relation at

    Methods
    -------
    r500c : (m,) array
      Radius range corresponding to m500c and cosmo
    f_bar : float
      cosmic baryon fraction corresponding to cosmo
    fgas500c : (z,m) array
      gas fractions corresponding to fgas500c_prms
    fcen500c : (z,m) array
      central fractions corresponding to fgas500c_prms
    fsat500c : (z,m) array
      satellite fractions corresponding to fgas500c_prms
    m200m_dmo : (z,m) array
      DMO equivalent halo masses corresponding to m500c and fgas500c
    c200m_dmo : (z,m) array
      concentrations for the DMO equivalent haloes
    r200m_dmo : (z,m) array
      virial radii for the DMO equivalent haloes
    dlnk : float
      lnk k_range interval for mass function
    k_range : (k_bins,) array
      wavevector range corresponding to given k_min, k_max and k_bins
    cosmo_prms : dict
      cosmological parameters from cosmo
    rho_crit : (z_range,) [units Msun h^2 / Mpc^3]
      critical density of the universe for z_range
    rho_m : (z_range,) [units Msun h^2 / Mpc^3]
      mean matter density of the universe for z_range

    '''
    def __init__(self,
                 m500c,
                 # r_min needs to be low in order to get correct masses from
                 # integration
                 r_min=-4, r_bins=100,
                 logk_min=-1.8, logk_max=2., k_bins=200,
                 cosmo=cosmo.Cosmology(),
                 z_range=0.,
                 fgas_500c_prms={"log10mt": 13.94,
                                 "a": 1.35,
                                 "norm": None,
                                 "fgas_500c_max": inp_interp.fgas_500c_max_interp},
                 f_c=0.86,
                 sigma_lnc=0.0,
                 # gamma=np.linspace(0., 3., 9)
                 **kwargs):

        super(Parameters, self).__init__()
        self.m500c = m500c
        # self.m200m_dmo = m200m_dmo
        self.r_min = r_min
        self.r_bins = r_bins
        self.logk_min = logk_min
        self.logk_max = logk_max
        self.k_bins = k_bins
        self.cosmo = cosmo
        if np.size(z_range) > 1:
            raise ValueError("redshift dependence not yet implemented")
        self.z_range = z_range
        self.fgas_500c_prms = fgas_500c_prms
        self.f_c = f_c
        self.sigma_lnc = sigma_lnc
        # self.gamma = gamma

    @property
    def r500c(self):
        return tools.mass_to_radius(self.m500c, 500 * self.cosmo.rho_crit)

    @property
    def f_bar(self):
        return (self.cosmo.omegab/self.cosmo.omegam)

    @property
    def fgas_500c(self):
        '''
        Return the f_gas(m500c) relation for m. The relation cannot exceed f_b

        This function assumes h=0.7 for everything!

        Parameters
        ----------
        m : array [M_sun / h_70]
        values of m500c to compute f_gas for
        log10mc : float
        the turnover mass for the relation in log10([M_sun/h_70])
        a : float
        the strength of the transition
        fstar_500c : interpolator or function of z and m
        asymptotic stellar fraction at r500c for each m500c
        cosmo : hmf.cosmo.Cosmology object
        relevant cosmological parameters

        Returns
        -------
        f_gas : array [h_70^(-3/2)]
        gas fraction at r500c for m
        '''
        log10mt = self.fgas_500c_prms["log10mt"]
        a = self.fgas_500c_prms["a"]
        fgas_500c_max_intrp = self.fgas_500c_prms["fgas_500c_max"](f_c=self.f_c,
                                                                   sigma_lnc=self.sigma_lnc)
        # allow for non-standard normalisations of the baryon fraction
        norm = self.fgas_500c_prms.get("norm", None)

        f_bar = self.f_bar

        if norm is None:
            norm = f_bar

        x = np.log10(self.m500c / 0.7) - log10mt

        # gas fractions without adjustment for stellar fraction
        fgas_fit = norm * (0.5 * (1 + np.tanh(x / a)))

        # interpolate stellar fractions
        coords = inp_interp.arrays_to_coords(self.z_range, np.log10(self.m500c))
        fgas_500c_max = fgas_500c_max_intrp(coords).reshape(np.shape(self.z_range) +
                                                            np.shape(self.m500c))
    
        # gas fractions that will cause halo to exceed cosmic baryon fraction
        cb_exceeded = (fgas_fit >= fgas_500c_max)

        # if fgas is simply an (m,) array, we need to tile it along the z-axis
        if not fgas_fit.shape == cb_exceeded.shape:
            fgas_fit = np.tile(fgas_fit.reshape(1,-1), (cb_exceeded.shape[0], 1))

        fgas_fit[cb_exceeded] = fgas_500c_max[cb_exceeded]

        return fgas_fit

    @property
    def fsat_500c(self):
        fsat_500c_interp = inp_interp.fsat500c_interp(f_c=self.f_c,
                                                      sigma_lnc=self.sigma_lnc)
        
        # reshape z and m500c
        z = np.tile(np.reshape(self.z_range, (-1,1)),
                    (1, np.shape(self.m500c)[0]))
        m500c = np.tile(np.reshape(self.m500c, (1,-1)),
                        (np.shape(z)[0], 1))
        fgas_500c = self.fgas_500c

        coords = np.vstack([z.flatten(), np.log10(m500c).flatten(),
                            fgas_500c.flatten()]).T
        fsat_500c = fsat_500c_interp(coords).reshape(fgas_500c.shape)
        return fsat_500c
        
    @property
    def fcen_500c(self):
        fcen_500c_interp = inp_interp.fcen500c_interp(f_c=self.f_c,
                                                      sigma_lnc=self.sigma_lnc)
        
        # reshape z and m500c
        z = np.tile(np.reshape(self.z_range, (-1,1)),
                    (1, np.shape(self.m500c)[0]))
        m500c = np.tile(np.reshape(self.m500c, (1,-1)),
                        (np.shape(z)[0], 1))
        fgas_500c = self.fgas_500c

        coords = np.vstack([z.flatten(), np.log10(m500c).flatten(),
                            fgas_500c.flatten()]).T
        fcen_500c = fcen_500c_interp(coords).reshape(fgas_500c.shape)
        return fcen_500c

    @property
    def m200m_dmo(self):
        m200m_dmo_interp = inp_interp.m200m_dmo_interp(f_c=self.f_c,
                                                       sigma_lnc=self.sigma_lnc)

        # reshape z and m500c
        z = np.tile(np.reshape(self.z_range, (-1,1)), (1, np.shape(self.m500c)[0]))
        m500c = np.tile(np.reshape(self.m500c, (1,-1)), (np.shape(z)[0], 1))
        fgas_500c = self.fgas_500c
        
        coords = np.vstack([z.flatten(), np.log10(m500c).flatten(),
                            fgas_500c.flatten()]).T
        m200m_dmo = m200m_dmo_interp(coords).reshape(fgas_500c.shape)
        return m200m_dmo

    @property
    def c200m_dmo(self):
        c200m_interp = inp_interp.c200m_interp()

        # reshape z and m500c
        z = np.tile(np.reshape(self.z_range, (-1,1)), (1, np.shape(self.m500c)[0]))
        m200m_dmo = self.m200m_dmo
        
        coords = np.vstack([z.flatten(), np.log10(m200m_dmo).flatten()]).T
        c200m_dmo = (c200m_interp(coords).reshape(m200m_dmo.shape) *
                     np.e**self.sigma_lnc)
        return c200m_dmo

    @property
    def r200m_dmo(self):
        rho_mz = 200 * np.reshape(self.rho_m, (np.shape(self.z_range) + (1,)))
        return tools.mass_to_radius(self.m200m_dmo, rho_mz)

    @property
    def dlnk(self):
        return np.log(10) * (self.logk_max - self.logk_min)/np.float(self.k_bins)

    @property
    def k_range(self):
        return np.logspace(self.logk_min, self.logk_max, self.k_bins)

    @property
    def cosmo_prms(self):
        return {
            "sigma_8": self.sigma_8,
            "H0": self.H0,
            "omegab": self.omegab,
            "omegac": self.omegac,
            "omegav": self.omegav,
            "n": self.n}

    @property
    def rho_crit(self):
        return self.cosmo.rho_crit * (1 + self.z_range)**3

    @property
    def rho_m(self):
        rho_critz = self.cosmo.rho_crit * (1 + self.z_range)**3
        return self.cosmo.omegam * rho_critz

# ------------------------------------------------------------------------------
# End of Parameters()
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Typical parameters for our simulations
# ------------------------------------------------------------------------------
# fiducial parameters, logk_min for comparisons with simulations
prms = Parameters(m500c=np.logspace(10,15,101), logk_min=-1.)

# fiducial parameters, but with f_c = 1
prms_fc1 = Parameters(m500c=np.logspace(10,15,101), logk_min=-1., f_c=1)

# get different mass ranges
prms_m1 = Parameters(m500c=np.logspace(10,11,101), logk_min=-1.)
prms_m2 = Parameters(m500c=np.logspace(11,12,101), logk_min=-1.)
prms_m3 = Parameters(m500c=np.logspace(12,13,101), logk_min=-1.)
prms_m4 = Parameters(m500c=np.logspace(13,14,101), logk_min=-1.)
prms_m5 = Parameters(m500c=np.logspace(14,15,101), logk_min=-1.)
prms_m = [prms_m1,
          prms_m2,
          prms_m3,
          prms_m4,
          prms_m5]

prms_mmin = Parameters(m500c=np.logspace(6,15,101), logk_min=-1.)
prms_mmax = Parameters(m500c=np.logspace(10,16,101), logk_min=-1.)
prms_mbinsm = Parameters(m500c=np.logspace(10,15,50), logk_min=-1.)
prms_mbinsp = Parameters(m500c=np.logspace(10,15,202), logk_min=-1.)

# get different gas fractions
prms_l10mt1 = Parameters(m500c=np.logspace(10,15,101), logk_min=-1.,
                         fgas_500c_prms={"log10mt": 13,
                                         "a": 1.35,
                                         "fgas_500c_max": inp_interp.fgas_500c_max_interp})
prms_l10mt2 = Parameters(m500c=np.logspace(10,15,101), logk_min=-1.,
                         fgas_500c_prms={"log10mt": 13.5,
                                         "a": 1.35,
                                         "fgas_500c_max": inp_interp.fgas_500c_max_interp})
prms_l10mt3 = Parameters(m500c=np.logspace(10,15,101), logk_min=-1.,
                         fgas_500c_prms={"log10mt": 14,
                                         "a": 1.35,
                                         "fgas_500c_max": inp_interp.fgas_500c_max_interp})
prms_l10mt4 = Parameters(m500c=np.logspace(10,15,101), logk_min=-1.,
                         fgas_500c_prms={"log10mt": 14.5,
                                         "a": 1.35,
                                         "fgas_500c_max": inp_interp.fgas_500c_max_interp})
prms_l10mt5 = Parameters(m500c=np.logspace(10,15,101), logk_min=-1.,
                         fgas_500c_prms={"log10mt": 15,
                                         "a": 1.35,
                                         "fgas_500c_max": inp_interp.fgas_500c_max_interp})
prms_l10mt = [prms_l10mt1,
              prms_l10mt2,
              prms_l10mt3,
              prms_l10mt4,
              prms_l10mt5]

# get 15 & 85 percentiles fgas_500c
prms_f15 = Parameters(m500c=np.logspace(10,15,101), logk_min=-1.,
                      fgas_500c_prms={"log10mt": 14.18,
                                      "a": 1.22,
                                      "fgas_500c_max": inp_interp.fgas_500c_max_interp})
prms_f85 = Parameters(m500c=np.logspace(10,15,101), logk_min=-1.,
                      fgas_500c_prms={"log10mt": 13.69,
                                      "a": 1.39,
                                      "fgas_500c_max": inp_interp.fgas_500c_max_interp})
prms_bias70 = Parameters(m500c=np.logspace(10,15,101), logk_min=-1.,
                         fgas_500c_prms={"log10mt": 14.44,
                                         "a": 1.99,
                                         "fgas_500c_max": inp_interp.fgas_500c_max_interp})
prms_bias80 = Parameters(m500c=np.logspace(10,15,101), logk_min=-1.,
                         fgas_500c_prms={"log10mt": 14.22,
                                         "a": 1.80,
                                         "fgas_500c_max": inp_interp.fgas_500c_max_interp})
prms_bias84 = Parameters(m500c=np.logspace(10,15,101), logk_min=-1.,
                         fgas_500c_prms={"log10mt": 14.15,
                                         "a": 1.70,
                                         "fgas_500c_max": inp_interp.fgas_500c_max_interp})

# get sigma_lnc variations
prms_slnc_p = Parameters(m500c=np.logspace(10,15,101), logk_min=-1., sigma_lnc=0.25)
prms_slnc_m = Parameters(m500c=np.logspace(10,15,101), logk_min=-1., sigma_lnc=-0.25)

# get constant fgas_500c at f_bar
prms_fconst = Parameters(m500c=np.logspace(10,15,101), logk_min=-1.,
                         fgas_500c_prms={"log10mt": 0,
                                         "a": np.inf,
                                         "norm": 2*prms.f_bar,
                                         "fgas_500c_max": inp_interp.fgas_500c_max_interp})

