from .headers import *
from .twoPoint import *
from .castorina import castorinaBias
import scipy


"""
Values and defintions from Table 3 of Wilson and White 2019.
"""
zs = np.array([2.0, 3.0, 3.8, 4.9, 5.9])
Muvstar = np.array([-20.60, -20.86, -20.63, -20.96, -20.91])
Muvstar = interp1d(
    zs, Muvstar, kind="linear", bounds_error=False, fill_value="extrapolate"
)
muv = np.array([24.2, 24.7, 25.4, 25.5, 25.8])
muv = interp1d(zs, muv, kind="linear", bounds_error=False, fill_value="extrapolate")
phi = np.array([9.70, 5.04, 9.25, 3.22, 1.64]) * 0.001
phi = interp1d(zs, phi, kind="linear", bounds_error=False, fill_value="extrapolate")
alpha = np.array([-1.6, -1.78, -1.57, -1.60, -1.87])
alpha = interp1d(zs, alpha, kind="linear", bounds_error=False, fill_value="extrapolate")


def compute_covariance_matrix(fishcast, zbin_index, nratio=1):
    """
    Covariance is diagonal. Returns an array of length Nk*Nmu.
    """

    # Compute prefactor that multiplies P_{gg,i}(z)^2 in covariance (see Eq. 3.6 in
    # Sailer+2021)
    z = fishcast.experiment.zcenters[zbin_index]
    prefactor = (4.0 * np.pi**2.0) / (
        fishcast.dk * fishcast.dmu * fishcast.Vsurvey[zbin_index] * fishcast.k**2.0
    )

    # Compute number density (incorporates thermal noise for HI survey)
    number_density = compute_n(fishcast, z)
    number_density_with_sigmaz = number_density

    # Get fiducial power spectrum (including noise)
    if fishcast.recon:
        P_fid = fishcast.P_recon_fid_for_cov[zbin_index]
    else:
        P_fid = fishcast.P_fid_for_cov[zbin_index]

    if fishcast.experiment.sigma_z > 1e-10:
        # The number density (including 21cm noise) is effectively reduced if there are
        # redshift uncertainties
        Hz = fishcast.cosmo_fid.Hubble(z) * (299792.458) / fishcast.params["h"]
        sigma_parallel = (3.0e5) * (1.0 + z) * fishcast.experiment.sigma_z / Hz
        number_density_with_sigmaz = number_density * np.maximum(
            np.exp(-fishcast.k**2.0 * fishcast.mu**2.0 * sigma_parallel**2.0),
            1.0e-20,
        )

    if (fishcast.experiment.sigma_z > 1e-10) or (nratio != 1):
        # Adjust noise in P_fid to account for redshift uncertainties and/or manual
        # rescaling, assuming that fiducial experiment doesn't have any redshift
        # uncertainties
        if fishcast.experiment.HI:
            raise NotImplmementedError(
                "Can't have sigma_z or rescaled noise with HI survey!"
            )

        C = (
            prefactor
            * (P_fid - 1 / number_density + 1 / nratio / number_density_with_sigmaz)
            ** 2.0
        )
    else:
        C = prefactor * P_fid**2

    # Ensure covariance doesn't contain any zeros
    return np.maximum(C, 1e-50)


def covariance_Cls(
    fishcast, kmax_knl=1.0, CMB="SO", only_kk=False, fsky_CMB=0.4, fsky_intersect=None
):
    """
    Returns a covariance matrix Cov[X,Y] as a function of l. X (and Y) is in the basis

         X \in {k-k, k-g1, ..., k-gn, g1-g1, ..., gn-gn}   (the basis has dimension 2*n+1)

    where g1 is the galaxies in the first redshift bin, k-gi is the cross-correlation of
    the CMB kappa map and the galaxies in the i'th bin, and so on.

    if only_kk, output C has shape (1, 1, n_ell), otherwise (n_z, n_z, n_ell)
    """
    n = fishcast.experiment.nbins
    zs = fishcast.experiment.zcenters
    zes = fishcast.experiment.zedges
    # Lensing noise
    input_dir = os.path.join(os.path.dirname(FishLSS.__file__), "../input/")
    if CMB == "SO":
        data = np.genfromtxt(
            os.path.join(
                input_dir, "nlkk_v3_1_0deproj0_SENS2_fsky0p4_it_lT30-3000_lP30-5000.dat"
            )
        )
        l, N = data[:, 0], data[:, 7]
    elif CMB == "Planck":
        data = np.genfromtxt(os.path.join(input_dir, "nlkk_planck.dat"))
        l, N = data[:, 0], data[:, 1]
    elif CMB == "Perfect":
        l, N = fishcast.ell, fishcast.ell * 0
    else:
        data = np.genfromtxt(
            os.path.join(
                input_dir, "S4_kappa_deproj0_sens0_16000_lT30-3000_lP30-5000.dat"
            )
        )
        l, N = data[:, 0], data[:, 7]

    Nkk_interp = interp1d(l, N, kind="linear", bounds_error=False, fill_value=1)
    l = fishcast.ell
    Nkk = Nkk_interp(l)
    # Cuttoff high ell by blowing up the covariance for ell > ellmax
    chi = (
        lambda z: (1.0 + z)
        * fishcast.cosmo_fid.angular_distance(z)
        * fishcast.params["h"]
    )
    ellmaxs = np.array([kmax_knl * chi(z) * fishcast.knl_z(z) for z in zs])
    constraint = np.ones((n, len(l)))
    idx = np.array([np.where(l)[0] >= ellmax for ellmax in ellmaxs])
    for i in range(n):
        constraint[i][idx[i]] *= 1e10
    # relevant fsky values
    fsky_LSS = fishcast.experiment.fsky
    if fsky_intersect is None:
        fsky_intersect = min(fsky_LSS, fsky_CMB)  # full-overlap by default
    # build covariance matrix
    if only_kk:
        C = np.zeros((1, 1, len(l)))
    else:
        C = np.zeros((2 * n + 1, 2 * n + 1, len(l)))
    #
    Ckk = fishcast.Ckk_fid_for_cov
    # kk, kk
    C[0, 0] = 2 * (Ckk + Nkk) ** 2 / (2 * l + 1) / fsky_CMB
    if not only_kk:
        for i in range(n):
            Ckgi = fishcast.Ckg_fid_for_cov[i]
            # kk, kg
            C[i + 1, 0] = (
                2 * (Ckk + Nkk) * Ckgi / (2 * l + 1) * constraint[i] / fsky_CMB
            )
            C[0, i + 1] = C[i + 1, 0]
            # kk, gg
            C[i + 1 + n, 0] = (
                2
                * Ckgi**2
                / (2 * l + 1)
                * constraint[i]
                * fsky_intersect
                / fsky_LSS
                / fsky_CMB
            )
            C[0, i + 1 + n] = C[i + 1 + n, 0]
            for j in range(n):
                Ckgj = fishcast.Ckg_fid_for_cov[j]
                Cgigi = fishcast.Cgg_fid_for_cov[i]
                Cgjgj = fishcast.Cgg_fid_for_cov[j]
                # kgi, kgj
                C[i + 1, j + 1] = Ckgi * Ckgj * constraint[i] * constraint[j]
                if i == j:
                    C[i + 1, j + 1] += (Ckk + Nkk) * Cgigi * constraint[i]
                C[i + 1, j + 1] /= 2 * l + 1 * fsky_intersect
                # gigi, gjgj
                if i == j:
                    C[i + 1 + n, j + 1 + n] = (
                        2 * Cgigi**2 / (2 * l + 1) * constraint[i] / fsky_LSS
                    )
                # kgi, gjgj
                if i == j:
                    C[i + 1, i + 1 + n] = (
                        2 * Cgigi * Ckgi / (2 * l + 1) * constraint[i] / fsky_LSS
                    )
                    C[i + 1 + n, i + 1] = C[i + 1, i + 1 + n]
    return C


def compute_n(fishcast, z):
    """Effective 3d survey number density, in h^3/Mpc^3.

    For HI surveys, returns an array of length Nk*Nmu, which includes stochastic noise
    and thermal noise. For all other surveys, return a float corresponding to the
    comoving number density of tracers.

    Parameters
    ----------
    z : float
        Redshift.

    Returns
    -------
    n : array_like
        Number density (single number), or HI effective number density (packed as
        [k*mu]).
    """
    custom_n = fishcast.experiment.n is not None
    if fishcast.experiment.LBG and not custom_n:
        return LBGn(fishcast, z)
    if fishcast.experiment.Halpha and not custom_n:
        return hAlphaN(fishcast, z)
    if fishcast.experiment.ELG and not custom_n:
        return ELGn(fishcast, z)
    if fishcast.experiment.HI and not custom_n:
        return HIneff(fishcast, z)
    if fishcast.experiment.Euclid and not custom_n:
        return Euclidn(z)
    if fishcast.experiment.MSE and not custom_n:
        return MSEn(fishcast, z)
    if fishcast.experiment.Roman and not custom_n:
        return Romann(fishcast, z)
    return fishcast.experiment.n(z)


def Muv(fishcast, z, m=24.5):
    """
    Equation 2.6 of Wilson and White 2019. Assumes a m=24.5 limit
    """
    result = m - 5.0 * np.log10(fishcast.cosmo_fid.luminosity_distance(z) * 1.0e5)
    result += 2.5 * np.log10(1.0 + z)
    return result


def muv_from_Muv(fishcast, z, M):
    """
    Equation 2.6 of Wilson and White 2019. Assumes a m=24.5 limit
    """
    result = M + 5.0 * np.log10(fishcast.cosmo_fid.luminosity_distance(z) * 1.0e5)
    result -= 2.5 * np.log10(1.0 + z)
    return result


def LBGn(fishcast, z, m=24.5):
    """
    Equation 2.5 of Wilson and White 2019. Return number
    density of LBGs at redshift z in units of Mpc^3/h^3.
    """
    upper_limit = Muv(fishcast, z, m=m)
    integrand = (
        lambda M: (np.log(10.0) / 2.5)
        * phi(z)
        * 10.0 ** (-0.4 * (1.0 + alpha(z)) * (M - Muvstar(z)))
        * np.exp(-(10.0 ** (-0.4 * (M - Muvstar(z)))))
    )

    return scipy.integrate.quad(integrand, -200, upper_limit)[0]


def ELGn(fishcast, z):
    zs = np.array([0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65])
    dNdz = np.array([309, 2269, 1923, 2094, 1441, 1353, 1337, 523, 466, 329, 126])
    N = 41252.96125 * dNdz * 0.1  # number of emitters in dz=0.1 across the whole sky
    volume = np.array(
        [
            ((1.0 + z + 0.05) * fishcast.cosmo_fid.angular_distance(z + 0.05)) ** 3.0
            for z in zs
        ]
    )
    volume -= np.array(
        [
            ((1.0 + z - 0.05) * fishcast.cosmo_fid.angular_distance(z - 0.05)) ** 3.0
            for z in zs
        ]
    )
    volume *= 4.0 * np.pi * fishcast.params_fid["h"] ** 3.0 / 3.0  # volume in Mpc^3/h^3
    n = list(N / volume)
    zs = np.array(
        [0.6, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.7]
    )
    n = [n[0]] + n
    n = n + [n[-1]]
    n = np.array(n)
    n_interp = interp1d(zs, n, kind="linear", bounds_error=False, fill_value=0.0)
    return float(n_interp(z))


def Romann(fishcast, z):
    zs = np.linspace(1.05, 2.95, 20)
    dNdz = np.array(
        [
            6160,
            5907,
            4797,
            5727,
            5147,
            4530,
            4792,
            3870,
            2857,
            2277,
            1725,
            1215,
            1642,
            1615,
            1305,
            1087,
            850,
            795,
            847,
            522,
        ]
    )
    N = 41252.96125 * dNdz * 0.1  # number of emitters in dz=0.1 across the whole sky
    volume = np.array(
        [
            ((1.0 + z + 0.05) * fishcast.cosmo_fid.angular_distance(z + 0.05)) ** 3.0
            for z in zs
        ]
    )
    volume -= np.array(
        [
            ((1.0 + z - 0.05) * fishcast.cosmo_fid.angular_distance(z - 0.05)) ** 3.0
            for z in zs
        ]
    )
    volume *= 4.0 * np.pi * fishcast.params_fid["h"] ** 3.0 / 3.0  # volume in Mpc^3/h^3
    n = list(N / volume)
    zs = np.array([zs[0]] + list(zs) + [zs[-1]])
    n = np.array([n[0]] + n + [n[-1]])
    n_interp = interp1d(zs, n, kind="linear", bounds_error=False, fill_value=0.0)
    return float(n_interp(z))


def Euclidn(z):
    #'''
    # From Table 3 of https://arxiv.org/pdf/1606.00180.pdf.
    #'''
    # zs = np.array([0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0])
    # n = np.array([1.25,1.92,1.83,1.68,1.51,1.35,1.20,1.00,0.80,0.58,0.38,0.35,0.21,0.11])*1e-3
    # return interp1d(zs, n, kind='linear', bounds_error=False, fill_value=0.)(z)
    """
    From Table 3 of https://arxiv.org/pdf/1910.09273.pdf
    """
    zs = np.array([0.9, 1.0, 1.2, 1.4, 1.65, 1.8])
    n = np.array([6.86, 6.86, 5.58, 4.21, 2.61, 2.61]) * 1e-4
    n_interp = interp1d(zs, n, kind="linear", bounds_error=False, fill_value=0.0)
    return n_interp(z)


def hAlphaN(fishcast, z):
    """
    Table 2 from Merson+17. Valid for 0.9<z<1.9.
    """
    zs = np.array([0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85])
    dNdz = np.array(
        [
            10535.0,
            8014.0,
            4998.0,
            3931.0,
            3455.0,
            2446.0,
            2078.0,
            1747.0,
            1524.0,
            1329.0,
        ]
    )
    N = 41252.96125 * dNdz * 0.1  # number of emitters in dz=0.1 across the whole sky
    volume = np.array(
        [
            ((1.0 + z + 0.05) * fishcast.cosmo_fid.angular_distance(z + 0.05)) ** 3.0
            for z in zs
        ]
    )
    volume -= np.array(
        [
            ((1.0 + z - 0.05) * fishcast.cosmo_fid.angular_distance(z - 0.05)) ** 3.0
            for z in zs
        ]
    )
    volume *= 4.0 * np.pi * fishcast.params_fid["h"] ** 3.0 / 3.0  # volume in Mpc^3/h^3
    n = list(N / volume)
    zs = np.array(
        [0.9, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.9]
    )
    n = [n[0]] + n
    n = n + [n[-1]]
    n = np.array(n)
    n_interp = interp1d(zs, n, kind="linear", bounds_error=False, fill_value=0.0)
    return n_interp(z)


def MSEn(fishcast, z, m=24.5):
    # return ELG number density for z<2.4
    if z <= 2.4:
        return 1.8e-4
    # interpolate figure 2 of https://arxiv.org/pdf/1903.03158.pdf to get efficiency
    mags = np.array([22.75, 23.25, 23.75, 24.25])
    zs = np.array([2.6, 3.0, 3.4, 3.8])
    blue = np.array(
        [
            [0.619, 0.846, 0.994],
            [0.452, 0.745, 0.962],
            [0.269, 0.495, 0.919],
            [0.102, 0.327, 0.908],
        ]
    )
    orange = np.array(
        [
            [0.582, 0.780, 0.981],
            [0.443, 0.663, 0.929],
            [0.256, 0.481, 0.849],
            [0.119, 0.314, 0.854],
        ]
    )
    green = np.array(
        [
            [0.606, 0.805, 0.919],
            [0.486, 0.708, 0.815],
            [0.289, 0.559, 0.746],
            [0.146, 0.363, 0.754],
        ]
    )
    red = np.array(
        [
            [0.624, 0.752, 0.934],
            [0.501, 0.671, 0.843],
            [0.334, 0.552, 0.689],
            [0.199, 0.371, 0.699],
        ]
    )
    weight = np.array([0.4, 0.3, 0.3])
    b, o = np.sum(blue * weight, axis=1), np.sum(orange * weight, axis=1)
    g, r = np.sum(green * weight, axis=1), np.sum(red * weight, axis=1)
    eff = np.array([b, o, r, g])
    #
    efficiency = interp2d(zs, mags, eff, kind="linear", bounds_error=False)
    #
    def integrand(M):
        result = (
            (np.log(10.0) / 2.5)
            * phi(z)
            * 10.0 ** (-0.4 * (1.0 + alpha(z)) * (M - Muvstar(z)))
        )
        result *= np.exp(-(10.0 ** (-0.4 * (M - Muvstar(z)))))
        m = muv_from_Muv(fishcast, z, M)
        result *= efficiency(z, m)
        return result

    #
    n = lambda m: scipy.integrate.quad(integrand, -200, Muv(fishcast, z, m=m))[0]
    return n(m)


def nofl(x, hexpack=True, Nside=256, D=6):
    """
    Adapted from https://github.com/slosar/PUMANoise.
    Helper function for puma_therm. Returns baseline
    density.
    """
    # quadratic packing
    if hexpack:
        # hexagonal packing
        a, b, c, d, e = 0.56981864, -0.52741196, 0.8358006, 1.66354748, 7.31776875
    else:
        # square packing
        a, b, c, d, e = 0.4847, -0.330, 1.3157, 1.5975, 6.8390
    xn = x / (Nside * D)
    n0 = (Nside / D) ** 2
    res = n0 * (a + b * xn) / (1 + c * xn**d) * np.exp(-((xn) ** e))
    return res


def get_Tb(fishcast, z):
    """
    Returns the mean 21cm brightness temp in K.
    If z < 6 use fitting formula from Eq. B1 of
    https://arxiv.org/pdf/1810.09572.
    """
    if z <= 6:
        Ohi = 4e-4 * (1 + z) ** 0.6
        h = fishcast.params_fid["h"]
        Ez = fishcast.cosmo_fid.Hubble(z) / fishcast.cosmo_fid.Hubble(0)
        Tb = 188e-3 * h / Ez * Ohi * (1 + z) ** 2
        return Tb
    else:
        raise NotImplementedError("T_b at z>6 is not implemented! (Requires X_HI(z))")
    omb = fishcast.params_fid["omega_b"]
    omm = fishcast.params_fid["omega_cdm"] + omb
    result = 28e-3 * ((1 + z) * 0.14 / 10 / omm) ** 0.5
    result *= omb / 0.022
    return result


def HI_therm(
    fishcast,
    z,
    old=True,
    sailer_nside=False,
):
    """
    Adapted from https://github.com/slosar/PUMANoise.
    Thermal noise power in Mpc^3/h^3. Thermal noise is
    given by equation D4 in https://arxiv.org/pdf/1810.09572.
    I divide by Tb (see get_Tb) to convert to Mpc^3/h^3.
    Returns a function of k [h/Mpc] and mu.
    """
    exp = fishcast.experiment

    D = exp.D
    effic = exp.aperture_efficiency

    ttotal = exp.tint * 365 * 24 * 3600.0 * exp.fill_factor**2
    if sailer_nside:
        # Noah's code divides Ndetectors by the fill factor, but the PUMANoise code
        # (and the forecasts in the Cosmic Visions white paper) only accounts for the
        # fill factor by rescaling the observing time, so our default option is *not*
        # to rescale Nside
        Nside = np.sqrt(exp.Ndetectors / exp.fill_factor)
    else:
        Nside = np.sqrt(exp.Ndetectors)
    Hz = (
        fishcast.cosmo_fid.Hubble(z) * (299792.458) / fishcast.params_fid["h"]
    )  # in h km/s/Mpc
    Ez = fishcast.cosmo_fid.Hubble(z) / fishcast.cosmo_fid.Hubble(0)
    lam = 0.211 * (1 + z)
    r = (
        (1.0 + z) * fishcast.cosmo_fid.angular_distance(z) * fishcast.params_fid["h"]
    )  # in Mpc/h
    Deff = D * np.sqrt(effic)
    FOV = (lam / Deff) ** 2
    y = 3e5 * (1 + z) ** 2 / (1420e6 * Hz)
    Sarea = 4 * np.pi * exp.fsky
    Ae = np.pi / 4 * D**2 * effic
    # k dependent terms
    kperp = lambda k, mu: k * np.sqrt(1.0 - mu**2.0)
    l = lambda k, mu: kperp(k, mu) * r * lam / (2 * np.pi)

    def Nu(k, mu):
        if old:
            return nofl(l(k, mu), hexpack=exp.hex_pack, Nside=Nside, D=D) * lam**2
        #
        ll, pi2lnb = np.genfromtxt("input/baseline_bs_44_D_14.txt").T
        nofl_new = interp1d(
            ll, pi2lnb / 2 / np.pi / ll, bounds_error=False, fill_value=0
        )
        result = nofl_new(l(k, mu)) * lam**2
        result = np.maximum(result, 1e-20)
        I = np.where(l(k, mu) < D)
        result[I] = 1e-20
        return result

    # temperatures
    Tb = get_Tb(fishcast, z)
    Tsky = lambda f: 25.0 * (f / 400.0) ** (-2.75) + 2.7
    Tscope = (
        exp.T_ampl / exp.omt_coupling / exp.sky_coupling
        + exp.T_ground * (1 - exp.sky_coupling) / exp.sky_coupling
    )
    Tsys = Tsky(1420.0 / (1 + z)) + Tscope
    Pn = (
        lambda k, mu: (Tsys / Tb) ** 2
        * r**2
        * y
        * (lam**4 / Ae**2)
        * 1
        / (2 * Nu(k, mu) * ttotal)
        * (Sarea / FOV)
    )
    return Pn


def HIneff(fishcast, z):
    """Effective number density for 21cm survey, including stochastic and thermal noise.

    Parameters
    ----------
    z : float
        Redshift.

    Returns
    -------
    neff : array_like
        Effective number density at z and k, packed as [k*mu].
    """
    therm = HI_therm(fishcast, z)(fishcast.k, fishcast.mu)
    stoch = fishcast.experiment.Pstoch_HI(z, fishcast.k)
    return 1.0 / (therm + stoch)
