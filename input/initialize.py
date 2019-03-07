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

import halo.density_profiles as dp

default = {
    "parameters": {
        # m500c grid
        "logm_min": 11,
        "logm_max": 15,
        "logm_bins": 101,
        # k grid
        "logk_min": -1.8,
        "logk_max": 2.,
        "logk_bins": 101,
        "z_min": 0,
        "z_max": 4,
        "z_bins": 16,
        # cosmology
        "cosmology": {
            "omegam": 0.2973,
            "omegac": 0.233,
            "omegab": 0.0463,
            "omegav": 0.7207,
            "n": 0.972,
            "h": 0.7
        }
    },
    "components": {
        "gas": {
            "profile": dp.profile_beta_plaw_uni,
            "parameters": {
                "delta": 500,
                "delta_ref": "crit",
                "f_gas": None,
                "beta": None,
                "rc": None
            },
        },
        "dm": {
            "profile": dp.profile_NFW,
            "parameters":{
                "delta": 500,
                "delta_ref": "crit",
                "f_dm": None,
                "c_dm": None
            },
        },
        "stars": {
            "cen" : {
                "profile": dp.profile_delta,
                "parameters":{
                    "delta": 200,
                    "delta_ref": "mean",
                    "f_cen": None
                },
            },
            "sat" : {
                "profile": dp.profile_NFW,
                "parameters":{
                    "delta": 200,
                    "delta_ref": "mean",
                    "f_sat": None,
                    "c_sat": None
                },
            }
        }
    },
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

    
    






