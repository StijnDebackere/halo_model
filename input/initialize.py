'''IDEAS
   -----

Want to keep everything as general and user-modifiable as possible.
Will have to assume some things to be fixed, or some contraints on the
input.

Input should be analytic functions or tables for an interpolator as a
function of:
- mass
- redshift
- cosmology (if this is the case, mostly for DM)

Parameters for these functions should be provided.

Make Profile class take these inputs and provide the corresponding
variables for the halo model

'''
import numpy as np
import yaml

import halo.input.interpolators as interp
import halo.cosmo as cosmo
from halo.parameters import Parameters

default = {
    # m500c grid
    "m500c": np.logspace(10, 15, 101),
    # r grid
    "r_min": -4,
    "r_bins": 100,
    # k grid
    "logk_min": -1.8,
    "logk_max": 2.,
    "logk_bins": 101,
    # z grid
    # "z_range": np.linspace(0, 2, 8),
    "z_range": 0.,
    # stellar parameters
    # => matters for initialization of the model,
    # if this is changed, need to rerun
    # table_m500c_to_m200m_dmo() to get correct halo masses
    "f_c": 0.86,
    "sigma_lnc": 0.0,
    # hot gas parameters
    # => only matter for halo model
    "fgas500c_prms": {"log10mt": 13.94,
                      "a": 1.35,
                      "norm": None,
                      "fgas_500c_max": interp.fgas_500c_max_interp},
    # cosmology
    "cosmo": cosmo.Cosmology(**{"sigma_8": 0.821,
                                "omegam": 0.2793,
                                "omegac": 0.233,
                                "omegab": 0.0463,
                                "omegav": 0.7207,
                                "n": 0.972,
                                "h": 0.7})
}


def init_model(fname=None):
    '''
    Initialize the model from an input YAML file.
    '''
    # load either the default parameters or fname
    if fname is None:
        prms = default
    else:
        with open(fname, 'r') as f:
            try:
                prms = yaml.load(f)
            except yaml.YAMLError as exc:
                print(exc)

    return Parameters(**prms)
