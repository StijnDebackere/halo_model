'''
TO-DO -> add z_range support to profile as new axis
'''
import multiprocessing

import numpy as np
import mpmath as mp
from scipy.special import factorial

import halo.tools as tools
from halo.model._cache import Cache, cached_property, parameter

import pdb

class Profile(Cache):
    '''
    An object containing all relevant information for a density profile.

    Parameters
    ----------
    r_range : (m,r) array
      array containing r_range for each M (with r_range[:,-1] = r_vir)
    m_range : (m,) array
      array containing masses to compute profile for
    k_range : (k,) array
      array containing wavevectors to compute profile for
    profile : function or array
      function to compute the density profile (should have first and second
      args for r_range and m_range) or array containing profile with M along 
      axis 0 and r along axis 1
    profile_f : function/array or None
      function to compute the density profile Fourier transform (should have 
      first and second args for k_range and m_range) or array containing profile 
      with M along axis 0 and k along axis 1.
      Choose None if you want to use the Taylor expanded Fourier transform.
    n : int
      number of last coefficient in Taylor expansion of density profile
    taylor_err : float
      this value determines what value is accepted as zero for the last Taylor
      coefficient, since we want the alternating series to converge
    profile_args : dict (only needed if profile is function)
      dictionary containing the additional arguments to the profile function
    profile_f_args : dict (only needed if profile_f is function)
      dictionary containing the additional arguments to the profile_f function
    cpus : int
      number of cores to use

    Methods
    -------
    rho_r : (m,r) array
      density profile
    rho_k : (m,k) array
      Fourier transform of density profile
    F_n : (m,n+1) array
      array containing Taylor coefficients of Fourier transform
    rho_k_T : (m,k) array
      Fourier transform of density profile using Taylor expansion
    m_h : (m,) array
      array containing the halo mass for each halo, obtained by integration
    '''
    def __init__(self, r_range, m_range, k_range, profile, profile_f=None, n=84,
                 taylor_err=1e-50, profile_args=None, profile_f_args=None,
                 cpus=multiprocessing.cpu_count()):
        '''
        Initializes some parameters
        '''
        # This gets the Cache system working
        super(Profile, self).__init__()
        # Set all given parameters.
        self.r_range = r_range
        self.m_range = m_range
        self.k_range = k_range
        self.n = n
        self.cpus = cpus
        self.taylor_err = taylor_err
        self.profile = profile
        self.profile_args = profile_args
        self.profile_f = profile_f
        self.profile_f_args = profile_f_args
        
    #===========================================================================
    # Parameters
    #===========================================================================    
    @parameter
    def r_range(self, val):
        return val

    @parameter
    def m_range(self, val):
        return val

    @parameter
    def k_range(self, val):
        return val

    @parameter
    def n(self, val):
        if val <= (170-1)/2.:
            return val
        # elif val == None:
        else:
            raise ValueError('n will result in (2n+1)! > 170! -> Overflow')

    @parameter
    def taylor_err(self, val):
        return val

    @parameter
    def cpus(self, val):
        return val

    @parameter
    def profile(self, val):
        return val

    @parameter
    def profile_args(self, val):
        return val

    @parameter
    def profile_f(self, val):
        return val

    @parameter
    def profile_f_args(self, val):
        return val

    #===========================================================================
    # Methods
    #===========================================================================    
    @cached_property('r_range', 'm_range','profile', 'profile_args')
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
            dens_profile = self.profile(self.r_range,
                                        self.m_range,
                                        **self.profile_args)
        else:
            dens_profile = self.profile

        if len(dens_profile.shape) != 2:
            raise ValueError('profile should be an (m,r) array. ')
                
        else:
            return dens_profile

    @cached_property('rho_r', 'n', 'F_n', 'r_range', 'k_range', 'm_range',
                     'taylor_err', 'profile_f', 'profile_f_args', 'rho_k_T')
    def rho_k(self):
        '''
        Computes the Fourier profile either by calling the function with its args,
        by just returning the given array, or by computing the transform.

        Returns
        -------
        dens_profile_f : (m,k) array
          Fourier transform of rho_r
        '''
        if hasattr(self.profile_f,'__call__'):
            dens_profile_f = self.profile_f(self.k_range,
                                            self.m_range,
                                            **self.profile_f_args)
        elif self.profile_f != None:
            dens_profile_f = self.profile_f

        else:
            dens_profile_f = self.rho_k_T

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

    @cached_property('rho_r', 'n', 'r_range', 'k_range', 'm_range')
    def F_n(self):
        '''
        Computes the Taylor coefficients in the Fourier expansion:

            F_n[M] = 4 * pi / M * 1 / (2n+1)! int_r r^(2n+2) * profile[M,r] dr

        Returns
        -------
        F_n : (m,n+1) array
          Taylor coefficients of Fourier expansion
        '''
        # define shapes for readability
        n = self.n
        m = self.m_range.shape[0]
        k = self.k_range.shape[0]
        # Prefactor only changes along axis 0 (Mass)
        prefactor = (4.0 * np.pi) / self.m_range.reshape(m,1)

        # F_n is (m,n+1) array
        F_n = Profile._taylor_expansion_multi(n=self.n, r_range=self.r_range,
                                              profile=self.rho_r,
                                              cpus=self.cpus)
        F_n *= prefactor

        return F_n
        
    @cached_property('rho_r', 'n', 'F_n', 'r_range', 'k_range', 'm_range',
                     'taylor_err')
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
        n = self.n
        m = self.m_range.shape[0]
        k = self.k_range.shape[0]

        Fn = self.F_n
        # need (1,n+1) array to match F_n
        n_arr = np.arange(0,n+1,dtype=np.longdouble).reshape(1,n+1)
        # -> (m,n) array
        c_n = np.power(-1,n_arr) * Fn
        
        # need (k,n+1) array for exponent
        k_range = np.longdouble(self.k_range).reshape(self.k_range.shape[0],1)
        k_n = np.power(np.tile(k_range, (1,n+1)), (2 * n_arr))

        # need to match n terms and sum over them
        # result is (k,m) array -> transpose
        T_n = c_n.reshape(1,m,n+1) * k_n.reshape(k,1,n+1)
        u = np.sum(T_n,axis=-1).T

        # k-values which do not converge anymore will have coefficients
        # that do not converge to zero. Convergence to zero is determined
        # by taylor_err.
        indeces = np.argmax((T_n[:,:,-1] > self.taylor_err), axis=0)
        indeces[indeces == 0] = k
        for idx, idx_max in enumerate(indeces):
            u[idx,idx_max:] = np.nan
            if idx_max != k:
                u[idx] = tools.extrapolate_plaw(self.k_range, u[idx])

        # normalize spectrum so that u[k=0] = 1, otherwise we get a small
        # systematic offset, while we know that theoretically u[k=0] = 1
        # if (np.abs(u[:,0]) - 1. > 1.e-2).any():
        #     print('-------------------------------------------------',
        #           '! Density profile mass does not match halo mass !',
        #           '-------------------------------------------------')

        # u = u / u[:,0].reshape(m,1)

        return u
