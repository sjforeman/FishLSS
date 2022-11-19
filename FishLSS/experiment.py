import types

from .headers import *
from .castorina import castorinaPn


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
        n=1e-3,  # Galaxy number density, float (constant n) or function of z
        b=1.5,  # Galaxy bias, float (constant b) or function of z
        b2=None,  #
        alpha0=None,  #
        LBG=False,  #
        HI=False,  #
        Halpha=False,  #
        ELG=False,  #
        Euclid=False,  #
        MSE=False,  #
        Roman=False,  #
        custom_n=False,  #
        custom_b=False,  #
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
        HI_shot_model="castorina",
        HI_shot_multiplier=1.0,
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
        if not isinstance(n, float):
            self.n = n
        else:
            self.n = lambda z: n + 0.0 * z

        # If the bias is not a float, assumed to be a function of z
        if not isinstance(b, float):
            self.b = b
        else:
            self.b = lambda z: b + 0.0 * z

        # Bias/stochastic parameters
        self.b2 = b2
        self.alpha0 = alpha0

        # Flags for specific surveys
        self.LBG = LBG
        self.HI = HI
        self.Halpha = Halpha
        self.ELG = ELG
        self.Euclid = Euclid
        self.MSE = MSE
        self.Roman = Roman
        self.custom_n = custom_n
        self.custom_b = custom_b

        # HI survey parameters
        self.HI_shot_model = HI_shot_model
        self.HI_shot_multiplier = HI_shot_multiplier
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

    def Pshot_HI(self, z):
        if self.HI_shot_model == "castorina":
            return self.HI_shot_multiplier * castorinaPn(z)
        else:
            raise NotImplementedError("Unrecognized HI shot noise model!")
