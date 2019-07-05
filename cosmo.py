import pdb

class Cosmology(object):
    """
    A class that nicely deals with cosmological parameters.

    Most cosmological parameters are merely input and exposed as
    attributes in the class. However, more complicated relations such as
    the interrelation of omegab, omegac, omegam, omegav for example are dealt
    with in a more robust manner.

    The secondary purpose of this class is to provide simple mappings of the
    parameters to common python cosmology libraries (for now just `cosmolopy`
    and `pycamb`). It has been found by the authors that using more than one
    library becomes confusing in terms of naming all the parameters, so this
    class helps deal with that.

    .. note :: There are an incredible number of possible combinations of
            input parameters, many of which could easily be inconsistent. To
            this end, this class raises an exception if an inconsistent
            parameter combination is input, eg. h = 1.0, omegab = 0.05,
            omegab_h2 = 0.06.

    .. note :: `force_flat` is provided for convenience to ensure a flat cosmology.
            In nearly all cases (except where it would be quite perverse to do
            so) this will modify omegav if it is otherwise inconsistent. Eg. if
            ``omegam = 0.3, omegav = 0.8, force_flat = True`` is passed, the
            omegav will modified to 0.7.

    Parameters
    ----------

    default : str, {``"planck1_base"``}
        Defines a set of default parameters, based on a published set from WMAP
        or Planck. These defaults are applied in a smart way, so as not to
        override any user-set parameters.

        Current options are

        1. ``"planck1_base"``: The cosmology of first-year PLANCK mission (with no lensing or WP)

    force_flat : bool, default ``False``
        If ``True``, enforces a flat cosmology :math:`(\Omega_m+\Omega_\lambda=1)`.
        This will modify ``omegav`` only, never ``omegam``.

    \*\*kwargs :
        The list of available keyword arguments is as follows:

        1. ``sigma_8``: The normalisation. Mass variance in top-hat spheres with :math:`R=8Mpc h^{-1}`
        #. ``n``: The spectral index
        #. ``w``: The dark-energy equation of state
        #. ``cs2_lam``: The constant comoving sound speed of dark energy
        #. ``t_cmb``: Temperature of the CMB
        #. ``y_he``: Helium fraction
        #. ``N_nu``: Number of massless neutrino species
        #. ``N_nu_massive``:Number of massive neutrino species
        #. ``z_reion``: Redshift of reionization
        #. ``tau``: Optical depth at reionization
        #. ``delta_c``: The critical overdensity for collapse
        #. ``h``: The hubble parameter
        #. ``H0``: The hubble constant
        #. ``omegan``: The normalised density of neutrinos
        #. ``omegam``: The normalised density of matter
        #. ``omegav``: The normalised density of dark energy
        #. ``omegab``: The normalised baryon density
        #. ``omegac``: The normalised CDM density
        #. ``omegab_h2``: The normalised baryon density by ``h**2``
        #. ``omegac_h2``: The normalised CDM density by ``h**2``

        .. note :: The reason these are implemented as `kwargs` rather than the
                usual arguments, is because the code can't tell *a priori* which
                combination of density parameters the user will input.

    """
    rho_crit = 2.7763458 * (10.0**11.0)
    def __init__(self, default="wmap9_nobao", force_flat=True, **kwargs):
        # This gets the Cache system working
        super(Cosmology, self).__init__()

        # Map the 'default' cosmology to its dictionary
        if default == "wmap9_nobao":
            self.__base = dict(wmap9_nobao, **extras)

        # Set some simple parameters
        self.force_flat = force_flat

        # if no kwargs are given, assume the default cosmology
        if not any(kwargs):
            if default == "wmap9_nobao":
                self.cosmo_update(**dict(wmap9_nobao, **extras))
        else:
            self.cosmo_update(**kwargs)

    def __update_base(self):
        """
        Transfer current cosmology into base, and if it doesn't exist, just pass
        """
        try:
            for k in self.__base:
                self.__base[k] = getattr(self, k)
        except:
            pass

    def cosmo_update(self, **kwargs):
        # First update the base
        self.__update_base()

        #=======================================================================
        # Set the "easy" values (no dependence on anything else)
        #=======================================================================
        easy_params = ["sigma_8", "n", 'w', 'cs2_lam', 't_cmb', 'y_he', "N_nu",
                       "z_reion", "tau", "omegan", 'delta_c', "N_nu_massive"]
        for p in easy_params:
            if p in kwargs:
                setattr(self, p, kwargs.pop(p))
            elif not hasattr(self, p):
                setattr(self, p, self.__base[p])

        #=======================================================================
        # Now the hard parameters (multi-dependent)
        #=======================================================================
        ################### h/H0 ###############################################
        if "h" in kwargs and "H0" in kwargs:
            if kwargs['h'] != kwargs["H0"] / 100.0:
                raise ValueError("Inconsistent arguments: h and H0")
            else:
                del kwargs["H0"]

        if "H0" in kwargs:
            self.H0 = kwargs.pop("H0")
            self.h = self.H0 / 100.0

        elif "h" in kwargs:
            self.h = kwargs.pop("h")
            self.H0 = 100 * self.h

        else:
            self.H0 = self.__base["H0"]
            self.h = self.H0 / 100.0

        ############ DENSITY PARAMETERS ########################################
        # First check for any inconsistent combinations
        if "omegab" in kwargs and "omegab_h2" in kwargs:
            if kwargs["omegab"] != kwargs["omegab_h2"] * self.h ** 2:
                raise ValueError("Inconsistent arguments: omegab and omegab_h2")

        if "omegac" in kwargs and "omegac_h2" in kwargs:
            if kwargs["omegac"] != kwargs["omegac_h2"] * self.h ** 2:
                raise ValueError("Inconsistent arguments: omegac and omegac_h2")

        if "omegab" in kwargs and "omegac_h2" in kwargs:
            raise ValueError("Inconsistent arguments: omegab and omegac_h2")

        if "omegac" in kwargs and "omegab_h2" in kwargs:
            raise ValueError("Inconsistent arguments: omegab and omegac_h2")

        if "omegab" in kwargs and "omegac" in kwargs and "omegam" in kwargs:
            if kwargs["omegam"] != kwargs["omegab"] + kwargs["omegac"]:
                raise ValueError("Inconsistent arguments: omegam, omegac, omegab")

        if "omegab_h2" in kwargs and "omegac_h2" in kwargs and "omegam" in kwargs:
            if kwargs["omegam"] != (kwargs["omegab"] + kwargs["omegac"]) / self.h ** 2:
                raise ValueError("Inconsistent arguments: omegam_h2, omegac_h2, omegab")

        # # NOW SET THE VALUES
        for k, val in kwargs.items():
            if k in ["omegab", "omegac", "omegab_h2", "omegac_h2",
                     "omegam", "omegav"]:
                setattr(self, k, val)
            else:
                raise ValueError("%s is not a valid parameter for Cosmology" % k)

        if len(kwargs) <= 1:
            if "omegab" in kwargs:
                self.omegab_h2 = self.omegab * self.h ** 2
                

            elif "omegac" in kwargs:
                self.omegac_h2 = self.omegac * self.h ** 2

            elif "omegac_h2" in kwargs or "omegab_h2" in kwargs:
                pass

            elif "omegam" in kwargs:
                self.omegab_h2 = self.__base["omegab_h2"]
                self.omegac_h2 = self.omegam * self.h ** 2 - self.omegab_h2

            elif "omegav" in kwargs:
                if self.force_flat:
                    self.omegam = 1 - self.omegav
                    self.omegab_h2 = self.__base["omegab_h2"]
                    self.omegac_h2 = self.omegam * self.h ** 2 - self.omegab_h2

        elif len(kwargs) == 2:
            if "omegab_h2" in kwargs and "omegac_h2" in kwargs:
                pass

            elif "omegab_h2" in kwargs and "omegam" in kwargs:
                self.omegac_h2 = self.omegam * self.h ** 2 - self.omegab_h2
                self.omegam = self.omegam

            elif "omegab_h2" in kwargs and "omegav" in kwargs:
                if self.force_flat:
                    self.omegam = 1 - self.omegav
                    self.omegav = omegav
                    self.omegac_h2 = self.omegam * self.h ** 2 - self.omegab_h2

            elif "omegac_h2" in kwargs and "omegam" in kwargs:
                self.omegab_h2 = self.omegam * self.h ** 2 - self.omegac_h2
                self.omegam = self.omegam

            elif "omegac_h2" in kwargs and "omegav" in kwargs:
                if self.force_flat:
                    self.omegam = 1 - self.omegav
                    self.omegab_h2 = self.omegam * self.h ** 2 - self.omegac_h2

            elif "omegab" in kwargs and "omegac" in kwargs:
                self.omegab_h2 = self.omegab * self.h ** 2
                self.omegac_h2 = self.omegac * self.h ** 2

            elif "omegab" in kwargs and "omegam" in kwargs:
                self.omegac = self.omegam - self.omegab
                self.omegam = self.omegam
                self.omegab_h2 = self.omegab * self.h ** 2
                self.omegac_h2 = self.omegac * self.h ** 2

            elif "omegab" in kwargs and "omegav" in kwargs:
                self.omegab_h2 = self.omegab * self.h ** 2
                if self.force_flat:
                    self.omegam = 1 - self.omegav
                    self.omegac = self.omegam - self.omegab
                    self.omegac_h2 = self.omegac ** self.h ** 2

            elif "omegac" in kwargs and "omegam" in kwargs:
                self.omegab = self.omegam - self.omegac
                self.omegam = self.omegam
                self.omegab_h2 = self.omegab * self.h ** 2
                self.omegac_h2 = self.omegac * self.h ** 2

            elif "omegac" in kwargs and "omegav" in kwargs:
                self.omegac_h2 = self.omegac * self.h ** 2
                if self.force_flat:
                    self.omegam = 1 - self.omegav
                    self.omegab = self.omegam - self.omegac
                    self.omegab_h2 = self.omegab ** self.h ** 2

            elif "omegam" in kwargs and "omegav" in kwargs:
                if self.force_flat:
                    self.omegav = 1 - self.omegam
                self.omegab_h2 = self.__base["omegab_h2"]
                self.omegab = self.omegab_h2 / self.h ** 2
                self.omegac = self.omegam - self.omegab
                self.omegam = self.omegam
                self.omegac_h2 = self.omegac * self.h ** 2


        elif len(kwargs) == 3:
            if "omegab" in kwargs and "omegac" in kwargs and "omegav" in kwargs:
                if self.force_flat:
                    self.omegav = 1 - self.omegab - self.omegac
                self.omegab_h2 = self.omegab * self.h ** 2
                self.omegac_h2 = self.omegac * self.h ** 2

            elif "omegab_h2" in kwargs and "omegac_h2" in kwargs and "omegav" in kwargs:
                if self.force_flat:
                    self.omegav = 1 - (self.omegab_h2 + self.omegac_h2) / self.h ** 2

            elif "omegab" in kwargs and "omegam" in kwargs and "omegav" in kwargs:
                if self.force_flat:
                    self.omegav = 1 - self.omegam
                self.omegab_h2 = self.omegab * self.h ** 2
                self.omegac = self.omegam - self.omegab
                self.omegac_h2 = self.omegac * self.h ** 2

            elif "omegac" in kwargs and "omegam" in kwargs and "omegav" in kwargs:
                if self.force_flat:
                    self.omegav = 1 - self.omegam
                self.omegac_h2 = self.omegac * self.h ** 2
                self.omegab = self.omegam - self.omegac
                self.omegab_h2 = self.omegab * self.h ** 2

            elif "omegab_h2" in kwargs and "omegam" in kwargs and "omegav" in kwargs:
                if self.force_flat:
                    self.omegav = 1 - self.omegam
                self.omegac_h2 = self.omegam * self.h ** 2 - self.omegab_h2

            elif "omegac_h2" in kwargs and "omegam" in kwargs and "omegav" in kwargs:
                if self.force_flat:
                    self.omegav = 1 - self.omegam
                self.omegab_h2 = self.omegam * self.h ** 2 - self.omegac_h2

    @property
    def rho_m(self):
        return self.omegam * self.rho_crit

    @property
    def cosmo_dict(self):
        """
        Collect parameters into a dictionary suitable for pycamb.

        Returns
        -------
        dict
            Dictionary of values appropriate for pycamb
        """
        amap = {"sigma_8":"sigma_8",
                "omegab":"omegab",
                "omegac":"omegac",
                "omegav":"omegav",
                "omegam":"omegam",
                "n":"n",
                "H0":"H0"}

        return_dict = {}
        for k, v in amap.items():
            return_dict.update({v:getattr(self, k)})

        return return_dict

    @property
    def astropy_dict(self):
        """
        Collect parameters into a dictionary suitable for pycamb.

        Returns
        -------
        dict
            Dictionary of values appropriate for pycamb
        """
        amap = {"omegam":"Om0",
                "H0":"H0"}

        return_dict = {}
        for k, v in amap.items():
            return_dict.update({v:getattr(self, k)})

        return return_dict

    @property
    def pycamb_dict(self):
        """
        Collect parameters into a dictionary suitable for pycamb.

        Returns
        -------
        dict
            Dictionary of values appropriate for pycamb
        """
        amap = {"w":"w_lam",
               "t_cmb":"TCMB",
               "y_he":"yhe",
               "z_reion":"reion__redshift",
               "N_nu":"Num_Nu_massless",
               "omegab":"omegab",
               "omegac":"omegac",
               "H0":"H0",
               "omegav":"omegav",
               "omegak":"omegak",
               "omegan":"omegan",
               "cs2_lam":"cs2_lam",
               "n":"scalar_index",
               "N_nu_massive":"Num_Nu_massive"
               }

        return_dict = {}
        for k, v in amap.items():
            return_dict.update({v:getattr(self, k)})

        return return_dict

    @property
    def cosmolopy_dict(self):
        """
        Collect parameters into a dictionary suitable for cosmolopy.

        Returns
        -------
        dict
            Dictionary of values appropriate for cosmolopy
        """
        amap = {"tau":"tau",
               "z_reion":"z_reion",
               "omegab":"omega_b_0",
               "h":"h",
               "omegav":"omega_lambda_0",
               "omegak":"omega_k_0",
               "sigma_8":"sigma_8",
               "omegam":"omega_M_0",
               "n":"n",
               "omegan":"omega_n_0",
               "N_nu_massive":"N_nu",
               "w":"w"}

        return_dict = {}
        for k, v in amap.items():
            return_dict.update({v:getattr(self, k)})

        return return_dict

#===============================================================================
# SOME BASE COSMOLOGIES
#===============================================================================
# The extras dict has common parameter defaults between all bases
extras = {"w"   :-1,
          "omegan"  : 0.0,
          'cs2_lam' : 1,
          't_cmb'    : 2.725,
          'y_he'     : 0.24,
          'N_nu'    : 3.04,
          "delta_c" : 1.686,
          "N_nu_massive":0.0,
          }

# # Base Planck (no extra things like lensing and WP)
planck1_base = {"omegab_h2"   : 0.022068,
                "omegac_h2"   : 0.12029,
                "omegav"   : 0.6825,
                "H0"       : 67.11,
                'z_reion': 11.35,
                'tau': 0.0925,
                "sigma_8":0.8344,
                "n":0.9624,
                }

wmap9_nobao = {'sigma_8': 0.821,
               'H0': 70.0,
               'z_reion': 10.6,
               'tau': 0.089,
               'omegab': 0.0463,
               'omegac': 0.233,
               'omegam': 0.2793,
               'omegav': 0.7207,
               'n': 0.972}
