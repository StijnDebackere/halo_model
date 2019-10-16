'''
TO-DO -> add z_range support to profile as new axis
'''
import multiprocessing

import numpy as np
from scipy.special import factorial

import halo.tools as tools
import halo.parameters as p

import pdb

class Profile(object):
    '''
    An object containing all relevant information for a density
    profile rho(r|m,z)

    Parameters
    ----------
    cosmo : hmf.cosmo.Cosmology object
      object with cosmological parameters in hmf.hmf readable format
    r_min : float [log10(Mpc/h)]
      the logarithmic minimum radius in physical units
    r_h : (m,) array [Mpc/h]
      the halo radius in physical units
    r_bins : int
      number of bins for the radial range of the profile
    k_range : (k,) array
      array containing wavevectors to compute profile for
    profile : function or array
      function to compute the density profile (should have first and second
      args for r_range and m_range) or array containing profile with M along
      axis 0 and r along axis 1
    profile_args : dict (only needed if profile is function)
      dictionary containing the additional arguments to the profile function
    profile_f : function/array or None
      function to compute the density profile Fourier transform (should have
      first and second args for k_range and m_range) or array containing profile
      with M along axis 0 and k along axis 1.
      Choose None if you want to use the Taylor expanded Fourier transform.
    profile_f_args : dict (only needed if profile_f is function)
      dictionary containing the additional arguments to the profile_f function
    n : int
      number of last coefficient in Taylor expansion of density profile
    taylor_err : float
      this value determines what value is accepted as zero for the last Taylor
      coefficient, since we want the alternating series to converge
    cpus : int
      number of cores to use

    Methods
    -------
    r_range : (m,r) array
      array containing radial range in physical units
    rho_r : (m,r) array
      density profile
    rho_k : (m,k) array
      Fourier transform of density profile
    F_n : (m,n+1) array
      array containing Taylor coefficients of Fourier transform
    rho_k_T : (m,k) array
      Fourier transform of density profile using Taylor expansion
    '''
    defaults = {}
    # this will indicate indices where taylor_err is exceeded for each m_h
    def __init__(self, cosmo, r_h, profile, profile_mass,
                 r_min=p.prms.r_min,
                 # r_h=p.prms.r200m_dmo,
                 r_bins=p.prms.r_bins,
                 k_range=p.prms.k_range,
                 z=p.prms.z_range,
                 # profile=profs.profile_NFW,
                 profile_args=None,
                 # profile_mass=tools.m_NFW,
                 profile_f=None,
                 profile_f_args=None,
                 n=84,
                 taylor_err=1e-50,
                 extrap=True,
                 cpus=multiprocessing.cpu_count()):
        '''
        Initializes some parameters
        '''
        # This gets the Cache system working
        super(Profile, self).__init__()
        # Set all given parameters.
        self.cosmo = cosmo
        self.r_min = r_min
        self.r_h = r_h
        self.r_bins = r_bins
        if ((np.roll(np.log10(k_range), -1)[:-1] - np.log10(k_range)[:-1]) !=
            (np.log10(k_range)[1] - np.log10(k_range)[0])).all():
            raise ValueError('k_range needs to be log spaced')
        self.k_range = k_range
        self.z = z
        self.n = n
        self.extrap = extrap
        self.cpus = cpus
        self.taylor_err = taylor_err
        self.profile = profile
        self.profile_args = profile_args
        # WATCH OUT WITH THIS ONE! It already assumes profile_args in the
        # function call!! Need to update profile_mass if updating profile_args!!
        if not hasattr(profile_mass, '__call__'):
            raise ValueError('profile_mass needs to be callable function')
        self.profile_mass = profile_mass
        self.profile_f = profile_f
        self.profile_f_args = profile_f_args

    def __add__(self, other):
        if not np.allclose(self.r_min, other.r_min):
            raise AttributeError('Profiles need same r_min')
        if not np.allclose(self.r_h, other.r_h):
            raise AttributeError('Profiles need same r_h')
        if not np.allclose(self.r_bins, other.r_bins):
            raise AttributeError('Profiles need same r_bins')
        if not np.allclose(self.k_range, other.k_range):
            raise AttributeError('Profiles need same k_range')
        if not np.allclose(self.z, other.z):
            raise AttributeError('Profiles need same z')

        profile_args = tools.merge_dicts(self.profile_args, other.profile_args)
        profile_mass = (lambda r, **kwargs: self.profile_mass(r, **self.profile_args) +
                        other.profile_mass(r, **other.profile_args))
        profile = self.rho_r + other.rho_r
        profile_f = self.rho_k + other.rho_k

        return Profile(cosmo=self.cosmo,
                       r_min=self.r_min,
                       r_h=self.r_h,
                       r_bins=self.r_bins,
                       k_range=self.k_range,
                       z=self.z,
                       profile=profile,
                       profile_args=profile_args,
                       profile_mass=profile_mass,
                       profile_f=profile_f,
                       profile_f_args=None,
                       n=self.n,
                       taylor_err=self.taylor_err,
                       extrap=self.extrap,
                       cpus=self.cpus)

    #===========================================================================
    # Methods
    #===========================================================================
    def m_r(self, r):
        '''
        Return the total mass inside r
        '''
        return self.profile_mass(r, **self.profile_args)

    @property
    def r_range(self):
        '''
        Transform the radial range to physical units

        Returns
        -------
        r_range : (m,r) array
          Radial range in physical units
        '''
        return np.array([np.logspace(self.r_min, np.log10(rm), self.r_bins)
                         for rm in self.r_h])

    @property
    def m_h(self):
        '''
        Return total mass in the halo
        '''
        return self.m_r(self.r_h)

    @property
    def rho_r(self):
        '''
        Computes the density profile either by calling the function with its args,
        or by just returning the given array.

        Returns
        -------
        dens_profile : (m,r) array
          Density profile rho_r
        '''
        if hasattr(self.profile,'__call__'):
            dens_profile = self.profile(self.r_range, **self.profile_args)
        else:
            dens_profile = self.profile

        # # we do this roundabout inequality because there is a different roundoff
        # # in the logspace for r_range to r_max
        # m_in_prof = np.array([tools.m_h(dens_profile[idx][tools.lte(r, self.r_h[idx])],
        #                                 r[tools.lte(r, self.r_h[idx])])
        #                       for idx, r in enumerate(self.r_range)])

        # frac_diff = np.max(np.abs(m_in_prof / self.m_h - 1))

        # # only raise a warning in this case, since otherwise you have to keep
        # # tweaking r_min and r_bins. Warning is sufficient, user decides whether
        # # the level is acceptable...
        # if frac_diff > 5e-3:
        #     warnings.warn('the mass in rho_r and m_h differ at 5x10^-3 level ({:.3e})'.format(frac_diff))

        if len(dens_profile.shape) != 2:
            raise ValueError('profile should be an (m,r) array. ')

        else:
            return dens_profile

    @property
    def rho_k(self):
        '''
        Computes the Fourier profile either by calling the
        function with its args, by just returning the given array, or
        by computing the transform.

        Returns
        -------
        dens_profile_f : (m,k) array
          Fourier transform of rho_r

        '''
        if hasattr(self.profile_f,'__call__'):
            dens_profile_f = self.profile_f(self.k_range,
                                            **self.profile_f_args)

        elif isinstance(self.profile_f, np.ndarray):
            dens_profile_f = self.profile_f

        elif self.profile_f == None:
            dens_profile_f = self.rho_k_T

        else:
            raise ValueError('profile_f type invalid.')

        if len(dens_profile_f.shape) != 2:
            raise ValueError('profile_f should be an (m,k) array. ')

        else:
            return dens_profile_f

    @staticmethod
    def _taylor_expansion(procn,n_range,r,profile,out_q):
        '''
        Computes the Taylor coefficients for the profile expansion for n_range.

            F_n = 1 / (2n+1)! int_r r^(2n+2) * profile[M,r]

        Parameters
        ----------
        procn : int
          process id
        n_range : array
          array containing the index of the Taylor coefficients
        r : array
          radius range to integrate over
        profile : array
          density profile with M along axis 0 and r along axis 1
        out_q : queue
          queue to output results

        '''
        # (m,n) array
        F_n = np.empty((profile.shape[0],) + n_range.shape,dtype=np.longdouble)
        r = np.longdouble(r)

        for idx,n in enumerate(n_range):
            prefactor = 1./factorial(2*n+1, exact=True)
            result = prefactor * tools.Integrate(y=np.power(r, (2.0*n+2)) *
                                                 profile,
                                                 x=r,
                                                 axis=1)

            F_n[:,idx] = result

        results = [procn,F_n]
        out_q.put(results)

        return

    @staticmethod
    def _taylor_expansion_multi(n,r_range,profile,cpus):
        '''
        Computes Taylor expansion of the density profile in parallel.

        Parameters
        ----------
        n : int
          number of Taylor coefficients to compute
        r_range : (m,r) array
          radius range to integrate over
        profile : array
          density profile with M along axis 0 and r along axis 1
        cpus : int
          number of cpus to use

        Returns
        -------
        taylor_coefs : (m,k,n) array
          array containing Taylor coefficients of Fourier expansion
        '''
        manager = multiprocessing.Manager()
        out_q = manager.Queue()

        taylor = np.arange(0,n+1)
        # Split array in number of CPUs
        taylor_split = np.array_split(taylor,cpus)

        # Start the different processes
        procs = []

        for i in range(cpus):
            process = multiprocessing.Process(target=Profile._taylor_expansion,
                                              args=(i, taylor_split[i],
                                                    r_range,
                                                    profile,
                                                    out_q))
            procs.append(process)
            process.start()

        # Collect all results
        result = []
        for i in range(cpus):
            result.append(out_q.get())

        result.sort()
        taylor_coefs = np.concatenate([item[1] for item in result],
                                      axis=-1)

        # Wait for all worker processes to finish
        for p in procs:
            p.join()

        return taylor_coefs

    @property
    def F_n(self):
        '''
        Computes the Taylor coefficients in the Fourier expansion:

            F_n[M] = 4 * pi * 1 / (2n+1)! int_r r^(2n+2) * profile[M,r] dr

        Returns
        -------
        F_n : (m,n+1) array
          Taylor coefficients of Fourier expansion
        '''
        # define shapes for readability
        m_s = self.r_h.shape[0]
        # Prefactor only changes along axis 0 (Mass)
        prefactor = (4.0 * np.pi)

        # F_n is (m,n+1) array
        F_n = Profile._taylor_expansion_multi(n=self.n, r_range=self.r_range,
                                              profile=self.rho_r,
                                              cpus=self.cpus)
        F_n *= prefactor

        return F_n

    @property
    def rho_k_T(self):
        '''
        Computes the Fourier transform of the density profile, using a Taylor
        expansion of the sin(kr)/(kr) term. We have

            u[M,k] = sum_n (-1)^n F_n[M] k^(2n)

        Returns
        -------
        u : (m,k) array
          Fourier transform of density profile
        '''
        # define shapes for readability
        n_s = self.n
        m_s = self.r_h.shape[0]
        k_s = self.k_range.shape[0]

        Fn = self.F_n
        # need (1,n+1) array to match F_n
        n_arr = np.arange(0,n_s+1,dtype=np.longdouble).reshape(1,n_s+1)
        # -> (m,n) array
        c_n = np.power(-1,n_arr) * Fn

        # need (k,n+1) array for exponent
        k_n = np.power(np.tile(np.longdouble(self.k_range).reshape(k_s,1),
                               (1,n_s+1)),
                       (2 * n_arr))

        # need to match n terms and sum over them
        # result is (k,m) array -> transpose
        T_n = c_n.reshape(1,m_s,n_s+1) * k_n.reshape(k_s,1,n_s+1)
        u = np.sum(T_n,axis=-1).T

        # k-values which do not converge anymore will have coefficients
        # that do not converge to zero. Convergence to zero is determined
        # by taylor_err.
        indices = np.argmax((T_n[:,:,-1] > self.taylor_err), axis=0)
        indices[indices == 0] = k_s
        self.taylor_nan = indices
        for idx, idx_max in enumerate(indices):
            u[idx,idx_max:] = np.nan
            # this extrapolation is not really very good...
            if (idx_max != k_s) and self.extrap:
                u[idx] = tools.extrapolate_plaw(self.k_range, u[idx])

        # # normalize spectrum so that u[k=0] = 1, otherwise we get a small
        # # systematic offset, while we know that theoretically u[k=0] = 1
        # if (np.abs(u[:,0]) - 1. > 1.e-2).any():
        #     print('-------------------------------------------------',
        #           '! Density profile mass does not match halo mass !',
        #           '-------------------------------------------------',
        #           sep='\n')

        # nonnil = (u[:,0] != 0)
        # u[nonnil] = u[nonnil] / u[nonnil,0].reshape(-1,1)

        return u
