import types

from .headers import *
from .castorina import castorinaPn
import FishLSS.obuljen as obuljen
from .util import Interpolator2d


class experiment(object):
    """
    An object that contains all the information related to the experiment
    """

    def __init__(
        self,
        zmin=0.8,  # Minimum redshift of survey
        zmax=1.2,  # Maximum redshift of survey
        nbins=1,  # Number of redshift bins
        zedges=None,  # Optional: Edges of redshift bins. Default is evenly-spaced bins.
        fsky=0.5,  # Fraction of sky observed
        sigma_z=0.0,  # Redshift error sz/(1+z)
        # Galaxy number density, float (constant n) or function of z
        # Must be specified if none of the survey-specific flags below are used
        n=None,
        # Galaxy bias, float (constant b) or function of z
        # Must be specified if none of the survey-specific flags below are used
        b=None,
        b2=None,
        bs=None,
        alpha0=None,  #
        LBG=False,  #
        HI=False,  #
        Halpha=False,  #
        ELG=False,  #
        Euclid=False,  #
        MSE=False,  #
        Roman=False,  #
        pessimistic=False,  # HI survey: specifies the k-wedge
        Ndetectors=256**2.0,  # HI survey: number of detectors
        fill_factor=0.5,  # HI survey: the array's fill factor
        tint=5,  # HI survey: oberving time [years]
        sigv=100,  # comoving velocity dispersion for FoG contribution [km/s]
        D=6,
        hex_pack=True,
        aperture_efficiency=0.7,
        sky_coupling=0.9,
        omt_coupling=0.9,
        T_ground=300.0,
        T_ampl=50.0,
        HI_stoch_model="castorina",
        HI_stoch_multiplier=1.0,
        HI_sampling_model="castorina",
        knl_z0=None,
        dknl_dz=None,
    ):

        # Redshift parameters
        self.zmin = zmin
        self.zmax = zmax
        self.nbins = nbins
        self.zedges = np.linspace(zmin, zmax, nbins + 1)
        if zedges is not None:
            self.zedges = zedges
            self.nbins = len(zedges) - 1
        self.zcenters = (self.zedges[1:] + self.zedges[:-1]) / 2.0
        self.sigma_z = sigma_z

        # Sky fraction
        self.fsky = fsky

        # If the number density is not a float, assumed to be a function of z
        if n is not None:
            if not isinstance(n, float):
                self.n = n
            else:
                self.n = lambda z: n
        else:
            self.n = None

        # Bias/stochastic parameters.
        # If the bias is not a float, assumed to be a function of z
        if b is not None:
            if not isinstance(b, float):
                self.b = b
            else:
                self.b = lambda z: b
        else:
            self.b = None

        if b2 is not None:
            if not isinstance(b2, float):
                self.b2 = b2
            else:
                self.b2 = lambda z: b2
        else:
            self.b2 = None

        if bs is not None:
            if not isinstance(bs, float):
                self.bs = bs
            else:
                self.bs = lambda z: bs
        else:
            self.bs = None

        self.alpha0 = alpha0

        # Assumption for k_nl(z) (function itself is defined in fisherForecast).
        # In default cosmology, inverse of rms Zeldovich displacement is roughly
        # described by knl_z0 = 0.16 h/Mpc, dknl_dz = 0.13 h/Mpc
        if (knl_z0 is None and dknl_dz is not None) or (
            knl_z0 is not None and dknl_dz is None
        ):
            raise InputError("Must specify both knl_z0 and dknl_dz if one is specified")
        self.knl_z0 = knl_z0
        self.dknl_dz = dknl_dz

        # Flags for specific surveys
        self.LBG = LBG
        self.HI = HI
        self.Halpha = Halpha
        self.ELG = ELG
        self.Euclid = Euclid
        self.MSE = MSE
        self.Roman = Roman

        # HI stochasticity model
        self.HI_stoch_model = HI_stoch_model

        if HI_stoch_model in ["obuljen_TNG", "obuljen_TNG_white"]:
            # If using Obuljen models, read simulation measurements from disk, and
            # define interpolating functions
            obuljen_data = obuljen.get_TNG_Pstoch_models(
                os.path.join(os.path.dirname(obuljen.__file__), "../input")
            )
            if HI_stoch_model == "obuljen_TNG":
                self.HI_stoch_interpolator = Interpolator2d(
                    obuljen_data["z"],
                    obuljen_data["k"],
                    obuljen_data["Pstoch_values"],
                    kx=1,
                    ky=1,
                )
                # HI_stoch_function returns a 1d array, evaluated at the single input z
                # and array of input k values
                self.HI_stoch_function = self.HI_stoch_interpolator.evaluate
            else:
                self.HI_stoch_function = interp1d(
                    obuljen_data["z"],
                    obuljen_data["Pstoch_values_white"],
                    bounds_error=False,
                    fill_value=(
                        obuljen_data["Pstoch_values_white"][0],
                        obuljen_data["Pstoch_values_white"][-1],
                    ),
                )
        self.HI_stoch_multiplier = HI_stoch_multiplier

        # HI sampling noise model
        self.HI_sampling_model = HI_sampling_model
        if HI_sampling_model == "obuljen_TNG_white":
            # If using Obuljen TNG sampling noise model, read simulation measurements
            # from disk and define interpolating function
            obuljen_data = obuljen.get_TNG_Pstoch_models(
                os.path.join(os.path.dirname(obuljen.__file__), "../input")
            )
            self.HI_sampling_function = interp1d(
                obuljen_data["z"], obuljen_data["Pstoch_values_white"]
            )

        # HI survey parameters
        self.Ndetectors = Ndetectors
        self.fill_factor = fill_factor
        self.tint = tint
        self.sigv = sigv
        self.D = D
        self.hex_pack = hex_pack
        self.aperture_efficiency = aperture_efficiency
        self.sky_coupling = sky_coupling
        self.omt_coupling = omt_coupling
        self.T_ground = T_ground
        self.T_ampl = T_ampl
        self.pessimistic = pessimistic
        if pessimistic:
            self.N_w = 3.0
            self.kparallel_min = 0.1
        else:
            self.N_w = 1.0
            self.kparallel_min = 0.01

    def Pstoch_HI(self, z, k=None):
        """HI stochastic noise power spectrum.

        Parameters
        ----------
        z : float
            Redshift.
        k : array_like, optional
            Wavenumber(s). Can either be float or array of values. Must be specified
            if stochastic noise model is k-dependent; ignored otherwise. Default: None.

        Returns
        -------
        result : float or array_like
            Stochastic noise evaluated at z and k.
        """
        if self.HI_stoch_model == "castorina":
            result = self.HI_stoch_multiplier * castorinaPn(z)
        elif self.HI_stoch_model == "obuljen_TNG":
            result = self.HI_stoch_multiplier * self.HI_stoch_function(z, k)
        elif self.HI_stoch_model == "obuljen_TNG_white":
            result = self.HI_stoch_multiplier * self.HI_stoch_function(z)
        else:
            raise NotImplementedError("Unrecognized HI stochastic noise model!")

        return result

    def Psampling_HI(self, z):
        """HI sampling noise power spectrum.

        Sampling noise is assumed white, so the power spectrum is only a function of z.

        Parameters
        ----------
        z : float
            Redshift.

        Returns
        -------
        result : float
            Sampling noise power spectrum at z.
        """
        if self.HI_sampling_model == "castorina":
            return castorinaPn(z)
        elif self.HI_sampling_model == "obuljen_TNG_white":
            return self.HI_sampling_function(z)
        else:
            raise NotImplementedError("Unrecognized HI sampling noise model!")
