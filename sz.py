import numpy as np
from scipy.interpolate import interp1d

import halo.tools as tools
import halo.parameters as p

def compton_y(s, r_range, rho, T):
    '''
    Compute the Compton y parameter at projected distance s from the center
    for density profile rho

    Parameters
    ----------
    s : float
      Projected distance in Mpc

    r_range : array
      Radial range for rho in Mpc

    rho : array
      Halo density profile in M_sun/Mpc^3

    T : float
      Temperature in Kelvin assumed for the halo
    '''
    rho_int = interp1d(r_range, rho)
    r_int = np.logspace(np.log10(s + 1e-5), np.log10(r_range.max()), 150)

    integrand = rho_int(r_int) * r_int / np.sqrt(r_int**2 - s**2)

    integral = tools.Integrate(integrand, r_int)

    prefactor = 2.379e-18 * (T / 1e8) # in Mpc^2 / M_sun

    return prefactor * integral

# ------------------------------------------------------------------------------
# End of compton_y()
# ------------------------------------------------------------------------------
