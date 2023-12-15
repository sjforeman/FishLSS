"""Functions for forecasts for specific parameters.
"""

from .headers import *

# Basis of parameters corresponding to pre-computed CMB Fisher matrices
_CMB_BASIS = [
    "h",
    "log(A_s)",
    "n_s",
    "omega_cdm",
    "omega_b",
    "tau_reio",
    "m_ncdm",
    "N_ur",
    "alpha_s",
    "Omega_k",
]

# List of pre-computed CMB Fisher matrices, and directory containing them
_CMB_OPTIONS = [None, "Planck", "Planck_SO", "Planck_S4", "LiteBIRD_S0", "LiteBIRD_S4"]
_CMB_DIR = os.path.join(os.path.dirname(FishLSS.__file__), "../input/")

# List of options for CMB lensing to include in forecasts
_LENSING_OPTIONS = [None, "Planck", "SO", "S4", "Perfect"]


def load_forecast(name, verbose=False, out_root_dir="output", use_mpi=True):
    """Load forecast outputs into convenient dict.

    Parameters
    ----------
    name : str
        Identifier for forecast to load.
    verbose : bool, optional
        Print status messages. Default: False.
    out_root_dir : str, optional
        Path to root directory of forecast outputs. Default: "output".
    use_mpi : bool, optional
        Whether to use MPI when computing certain forecast quantities. Default: True.

    Returns
    -------
    output: dict
        Forecast outputs. Keys are "exp" for experiment object, "forecast" for
        fisherForecast object, and all keys in JSON summary.
    """

    # Load forecast summary from JSON file
    with open(f"{out_root_dir}/{name}/summary.json") as f:
        summary = json.load(f)

    # Initialize CLASS
    class_params = summary["CLASS default parameters"]
    cosmo = Class()
    cosmo.set(class_params)
    cosmo.compute()

    # Load b, n from summary
    b_z = summary["Linear Eulerian bias in each bin"]
    n_z = summary["Number density in each bin"]
    zcenters = summary["Centers of redshift bins"]

    # Define linear bias interpolating function
    if len(zcenters) > 1:
        b_interp = interp1d(zcenters, b_z)
    else:

        def b_interp(z):
            return np.array(b_z).item()

    # Define number density interpolating function
    n_z = np.array(n_z)
    if n_z.ndim == 1:
        if len(zcenters) > 1:
            n_interp = interp1d(zcenters, n_z)
        else:

            def n_interp(z):
                return np.array(n_z).item()

    else:
        interp_func = interp2d(zcenters, k, n_z.T)

        def n_interp(z_, k_):
            return interp_func(z_, k_).T

    # Define b_2 interpolating function
    if summary["custom_b2"]:
        if len(zcenters) > 1:
            b2_interp = interp1d(zcenters, summary["b2"])
        else:

            def b2_interp(z):
                return np.array(summary["b2"]).item()

    # Define b_s interpolating function
    if summary["custom_bs"]:
        if len(zcenters) > 1:
            bs_interp = interp1d(zcenters, summary["bs"])
        else:

            def bs_interp(z):
                return np.array(summary["bs"]).item()

    # Look for other forecast specifications
    if "HI_stoch_file" not in summary.keys():
        summary["HI_stoch_file"] = None
    if "HI_sampling_file" not in summary.keys():
        summary["HI_sampling_file"] = None
    if "HI_stoch_multiplier" not in summary.keys():
        summary["HI_stoch_multiplier"] = 1.0

    if "sigv" not in summary.keys():
        summary["sigv"] = 100
    if "dknl_dDinv" not in summary.keys():
        summary["dknl_dDinv"] = 0

    for key in ["deriv_dir", "deriv_Cl_dir", "deriv_recon_dir"]:
        if key not in summary.keys():
            summary[key] = None

    # Initialize experiment
    exp = experiment(
        zmin=summary["Edges of redshift bins"][0],
        zmax=summary["Edges of redshift bins"][-1],
        nbins=len(summary["Centers of redshift bins"]),
        fsky=summary["fsky"],
        b=b_interp if summary["custom_b"] else None,
        n=n_interp if summary["custom_n"] else None,
        b2=b2_interp if summary["custom_b2"] else None,
        bs=bs_interp if summary["custom_bs"] else None,
        sigv=summary["sigv"],
        HI=summary["HI"],
        pessimistic=summary["pessimistic"],
        HI_stoch_file=summary["HI_stoch_file"],
        HI_stoch_multiplier=summary["HI_stoch_multiplier"],
        HI_sampling_file=summary["HI_sampling_file"],
        Ndetectors=summary["Ndetectors"],
        fill_factor=summary["fill_factor"],
        tint=summary["t_int"],
        D=summary["dish_diameter"],
        hex_pack=summary["hex_pack"],
        aperture_efficiency=summary["aperture_efficiency"],
        sky_coupling=summary["sky_coupling"],
        omt_coupling=summary["omt_coupling"],
        T_ground=summary["T_ground"],
        T_ampl=summary["T_ampl"],
        knl_z0=summary["knl_z0"],
        dknl_dz=summary["dknl_dz"],
        dknl_dDinv=summary["dknl_dDinv"],
    )

    # Initialize forecast
    forecast = fisherForecast(
        experiment=exp,
        cosmo=cosmo,
        name=name,
        verbose=verbose,
        ell=np.array(summary["ell"]),
        kmax=summary["k"][-1],
        remove_lowk_delta2_powspec=summary["remove_lowk_delta2_powspec"],
        remove_lowk_delta2_cov=summary["remove_lowk_delta2_cov"],
        output_root_directory=out_root_dir,
        write_json_summary=False,
        use_mpi=use_mpi,
        deriv_directory=summary["deriv_dir"],
        deriv_Cl_directory=summary["deriv_Cl_dir"],
        deriv_recon_directory=summary["deriv_recon_dir"],
    )

    # Make dict of all the stuff we've loaded
    output = {"exp": exp, "forecast": forecast}
    output.update(summary)

    return output


def compute_forecast_recon_BAO(fc):
    """Compute single-parameter forecasts for alpha_perp and alpha_par.

    Parameters
    ----------
    fc : dict
        Dict from `load_forecast`.

    Returns
    -------
    output : dict
        Dict containing Fishers and inverse-Fishers at each z, as well as computed
        one-sigma uncertainties on alpha_perp and alpha_par at each z.
    """
    # Parameter basis to forecast for: 2 AP parameters and linear bias b.
    # When using reconstruction, Fisher also automatically includes 15 polynomial
    # coefficients describing broadband power spectrum shape.
    basis = np.array(["alpha_perp", "alpha_parallel", "b"])

    # Store reconstruction flag (so that we can restore it later), and set to True
    original_recon_flag = fc["forecast"].recon
    fc["forecast"].recon = True

    # Set parameters to marginalize over, and load precomputed derivatives to use
    fc["forecast"].marg_params = basis
    deriv = fc["forecast"].load_derivatives(basis)

    # Define routine that computes Fisher matrix in redshift bin i
    def F_func(i):
        return fc["forecast"].gen_fisher(
            basis, 100, derivatives=deriv, zbins=np.array([i])
        )

    # Get the Fisher matrices in each z bin, looping over z bins and then inverting
    #  in each bin.
    # Each Fisher matrix will be 18 x 18 with basis
    #  [alpha_perp, alpha_parallel, b, c_00, ...]
    F_z = np.array([F_func(i) for i in range(fc["exp"].nbins)])
    Finv_z = np.array([np.linalg.inv(F_z[i]) for i in range(fc["exp"].nbins)])

    # Compute one-sigma uncertainties on AP parameters. These are interpreted as the
    # uncertainties on D_A(z)/r_d and H(z)*r_d.
    onesigma_aperp = np.array(
        [np.sqrt(Finv_z[i][0, 0]) for i in range(fc["exp"].nbins)]
    )
    onesigma_apar = np.array([np.sqrt(Finv_z[i][1, 1]) for i in range(fc["exp"].nbins)])

    # Make dict of Fishers, inverse Fishers, and AP uncertainties
    output = {
        "F_z": F_z,
        "Finv_z": Finv_z,
        "onesigma_aperp": onesigma_aperp,
        "onesigma_apar": onesigma_apar,
    }

    # Set reconstruction flag to original value
    fc["forecast"].recon = original_recon_flag

    return output


def compute_forecast_sigma8_fixed_shape(fc, marginalize_over_Tb=True):
    """Compute single-parameter forecasts for sigma_8 with fixed linear P(k) shape.

    Parameters
    ----------
    fc : dict
        Dict from `load_forecast`.
    marginalize_over_Tb : bool, optional
        Marginalize over mean brightness temperature. Default: True.

    Returns
    -------
    output : dict
        Dict containing Fishers and inverse-Fishers at each z, as well as computed
        one-sigma uncertainties on sigma_8 at each z.
    """
    # Parameter basis to forecast for: power spectrum amplitude and bias/stoch
    # parameters. If desired, also include mean HI brightness temperature.
    basis = ["log(A_s)", "N", "alpha0", "b", "b2", "bs", "N2", "N4", "alpha2", "alpha4"]
    if marginalize_over_Tb:
        basis.append("Tb")
    basis = np.array(basis)

    # Store reconstruction flag (so that we can restore it later), and set to False
    original_recon_flag = fc["forecast"].recon
    fc["forecast"].recon = False

    # Set parameters to marginalize over, and load precomputed derivatives to use
    fc["forecast"].marg_params = basis
    deriv = fc["forecast"].load_derivatives(basis)

    # Get the Fisher matrices in each z bin: first define function that gets Fisher
    # matrix in z bin i, and then loop over z bins.
    def F_func(i):
        return fc["forecast"].gen_fisher(
            basis, 100, derivatives=deriv, zbins=np.array([i])
        )

    F_z = np.array([F_func(i) for i in range(fc["exp"].nbins)])

    # Invert Fisher matrix in each z bin
    Finv_z = np.array([np.linalg.inv(F_z[i]) for i in range(fc["exp"].nbins)])

    # Compute one-sigma uncertainty on sigma_8.
    # At the level of derivatives, d/dlog(sigma8) = 2*d/dlog(A_s), so the relative
    # error on sigma_8 is half the relative error on A_s.
    onesigma_sigma8 = np.array(
        [np.sqrt(Finv_z[i][0, 0]) / 2 for i in range(fc["exp"].nbins)]
    )

    # Make dict of Fishers, inverse Fishers, and AP uncertainties
    output = {
        "F_z": F_z,
        "Finv_z": Finv_z,
        "onesigma_sigma8": onesigma_sigma8,
    }

    # Set reconstruction flag to original value
    fc["forecast"].recon = original_recon_flag

    return output


def compute_forecast_sigma8_full_cosmology(
    fc, marginalize_over_Tb=True, omb_bbn_prior=0.0005
):
    """Compute single-parameter forecasts for sigma_8, varying all LCDM parameters.

    Parameters
    ----------
    fc : dict
        Dict from `load_forecast`.
    marginalize_over_Tb : bool, optional
        Marginalize over mean brightness temperature. Default: True.
    omb_bbn_prior : float, optional
        One-sigma prior on omega_b, from BBN. Default: 0.0005.

    Returns
    -------
    output : dict
        Dict containing Fishers and inverse-Fishers at each z, as well as computed
        one-sigma uncertainties on sigma_8 at each z.
    """
    # Parameter basis to forecast for: cosmological params along with bias/stoch
    # parameters. If desired, also include mean HI brightness temperature.
    # Before converting to numpy array, save index of omega_b for later.
    basis = [
        # Shared with CMB Fisher matrix
        "log(A_s)",
        "h",
        "n_s",
        "omega_cdm",
        "omega_b",
        "tau_reio",
        # LSS-only
        "N",
        "alpha0",
        "b",
        "b2",
        "bs",
        "N2",
        "N4",
        "alpha2",
        "alpha4",
    ]
    if marginalize_over_Tb:
        basis.append("Tb")
    omb_idx = basis.index("omega_b")
    basis = np.array(basis)

    # Store reconstruction flag (so that we can restore it later), and set to False
    original_recon_flag = fc["forecast"].recon
    fc["forecast"].recon = False

    # Set parameters to marginalize over, and load precomputed derivatives to use
    fc["forecast"].marg_params = basis
    deriv = fc["forecast"].load_derivatives(basis)

    # Routine to get R matrix that transforms Fisher matrix from log(A_s) basis to
    # sigma_8 basis
    def fisher_R_to_sig8(params, z, verbose=False):
        Rinv = np.zeros((len(params), len(params)), dtype=np.float64)

        for pari, par in enumerate(params):
            if verbose:
                print(pari, par)
            Rinv[0, pari] = fc["forecast"].dsigma8_dp(par, z)

        for i in range(1, len(params)):
            Rinv[i, i] = 1.0

        R = np.linalg.inv(Rinv)

        return R

    # Get the Fisher matrices in each z bin: first define function that gets Fisher
    # matrix in z bin i, and then loop over z bins
    def F_func(i):
        # Get Fisher
        out = fc["forecast"].gen_fisher(
            basis, 100, derivatives=deriv, zbins=np.array([i])
        )
        # Get R matrix
        R = fisher_R_to_sig8(basis, fc["Centers of redshift bins"][i])
        # Transform Fisher matrix from log(A_s) basis to sigma_8 basis
        out = np.dot(R.T, np.dot(out, R))
        # Add BBN prior on omega_b
        out[omb_idx, omb_idx] += omb_bbn_prior**-2
        return out

    F_z = np.array([F_func(i) for i in range(fc["exp"].nbins)])

    # Invert Fisher matrix in each z bin
    Finv_z = np.array([np.linalg.inv(F_z[i]) for i in range(fc["exp"].nbins)])

    # Compute absolute one-sigma uncertainty on sigma_8
    onesigma_sigma8_abs = np.array(
        [np.sqrt(Finv_z[i][0, 0]) for i in range(fc["exp"].nbins)]
    )

    # Convert to relative one-sigma uncertainty by dividing by fiducial value
    cosmo = fc["forecast"].cosmo
    s8_fid = np.array(
        [
            cosmo.sigma8() * cosmo.scale_independent_growth_factor(z)
            for z in fc["Centers of redshift bins"]
        ]
    )
    onesigma_sigma8_rel = onesigma_sigma8_abs / s8_fid

    # Make dict of Fishers, inverse Fishers, and relative sigma_8 uncertainties
    output = {
        "F_z": F_z,
        "Finv_z": Finv_z,
        "onesigma_sigma8": onesigma_sigma8_rel,
    }

    # Set reconstruction flag to original value
    fc["forecast"].recon = original_recon_flag

    return output


def _compute_forecast_parameters(
    cos_basis,
    LSS_basis,
    fc,
    cmb_fisher_basis_idx=None,
    marginalize_over_Tb=True,
    omb_bbn_prior=0.0005,
    cmb=None,
    lensing=None,
    lensing_ell_min=30,
    lensing_ell_max=500,
    exclude_LSS=False,
):
    """Base function for computing cosmological parameter forecast.

    If including priors from CMB, indices corresponding to each parameter in cos_basis
    must be passed as a list in cmb_fisher_basis_idx.

    Parameters
    ----------
    cos_basis : list
        List of cosmological parameters to vary in forecast.
    LSS_basis
        List of LSS-specific parameters to vary in forecast.
    fc : dict
        Dict from `load_forecast`.
    cmb_fisher_basis_idx : list, optional
        List of indices in CMB Fisher matrix that correspond to parameters in cos_basis.
        Must be specified if using CMB primary information. Default: None.
    marginalize_over_Tb : bool, optional
        Marginalize over mean brightness temperature. Default: True.
    omb_bbn_prior : float, optional
        One-sigma prior on omega_b, from BBN. Default: 0.0005.
    cmb : str, optional
        CMB experiment to include primary CMB info from. Default: None.
    lensing : str, optional
        CMB experiment to include lensing from. Default: None.
    lensing_ell_min, lensing_ell_max : int, optional
        Min/max multipoles to use from lensing. Defaults: 30, 500.
    exclude_LSS : bool, optional
        Exclude LSS from forecasts. Will throw error if True and not using either CMB
        or lensing in forecast. Default: False

    Returns
    -------
    output : dict
        Dict containing Fishers and inverse-Fishers, as well as computed
        one-sigma uncertainties on specified parameters.
    """

    if cmb not in _CMB_OPTIONS:
        raise InputError(f"Invalid choice for CMB priors: {cmb}")

    if lensing not in _LENSING_OPTIONS:
        raise InputError(f"Invalid choice for CMB lensing noise: {lensing}")

    if exclude_LSS and lensing is None and cmb is None:
        raise InputError("Can't exclude LSS unless using CMB or lensing in forecast!")

    n_cos_pars = len(cos_basis)

    # Load CMB Fisher matrix if specified, and select parameters we're varying
    if cmb is not None:
        if cmb_fisher_basis_idx is None:
            raise InputError("If using CMB priors, must specify cmb_fisher_basis_idx")

        # Load the full CMB Fisher matrix, and subselect the rows and columns
        # corresponding to cos_basis, based on the mapping in cmb_fisher_basis_idx.
        # TODO: find a more pythonic way to to this
        full_cmb_fisher = np.loadtxt(os.path.join(_CMB_DIR, f"{cmb}.txt"))
        cmb_fisher = np.zeros((n_cos_pars, n_cos_pars), dtype=full_cmb_fisher.dtype)
        for i in range(n_cos_pars):
            for j in range(n_cos_pars):
                cmb_fisher[i, j] = full_cmb_fisher[
                    cmb_fisher_basis_idx[i], cmb_fisher_basis_idx[j]
                ]

    # Combine cosmological parameter basis and LSS basis into one list
    basis = cos_basis + LSS_basis

    # If desired, also include mean HI brightness temperature
    if marginalize_over_Tb:
        basis.append("Tb")
    basis = np.array(basis)

    # Store reconstruction flag (so that we can restore it later), and set to False
    original_recon_flag = fc["forecast"].recon
    fc["forecast"].recon = False

    # Set parameters to marginalize over, and load precomputed derivatives to use.
    # Lensing derivatives will be loaded by gen_lensing_fisher() if needed
    fc["forecast"].marg_params = basis
    deriv = fc["forecast"].load_derivatives(basis)

    # Get Fisher matrix, containing "global" parameters and then z-specific parameters
    # for each z bin
    if not exclude_LSS:
        F_total = fc["forecast"].gen_fisher(basis, n_cos_pars, derivatives=deriv)

    # If using lensing information, compute kappa-kappa Fisher matrix and combine with
    # 21cm Fisher matrix. (For 21cm, we don't use Cl_gg and Cl_kg)
    if lensing is not None:
        Flens_total = fc["forecast"].gen_lensing_fisher(
            cos_basis,
            n_cos_pars,
            ell_min=lensing_ell_min,
            ell_max=lensing_ell_max,
            CMB=lensing,
            kk=True,
            only_kk=True,
        )
        if exclude_LSS:
            F_total = Flens_total
        else:
            F_total = fc["forecast"].combine_fishers([F_total, Flens_total], n_cos_pars)

    # Add Omega_b prior from BBN
    if omb_bbn_prior > 0:
        omb_idx = cos_basis.index("omega_b")
        BBN_prior = np.zeros(n_cos_pars, dtype=np.float64)
        BBN_prior[omb_idx] = omb_bbn_prior**-2
        BBN_prior = np.diag(BBN_prior)
        F_total = fc["forecast"].combine_fishers([F_total, BBN_prior], n_cos_pars)

    # Add CMB priors
    if cmb is not None:
        if exclude_LSS and lensing is None:
            F_total = cmb_fisher
        else:
            F_total = fc["forecast"].combine_fishers([F_total, cmb_fisher], n_cos_pars)

    # Invert Fisher matrix in each z bin
    Finv_total = np.linalg.inv(F_total)

    # Save Fisher and inverse Fisher matrix to output dict
    output = {
        "F_total": F_total,
        "Finv_total": Finv_total,
    }

    # Save estimated one-sigma uncertainties on each cosmo parameter
    for pi, p in enumerate(cos_basis):
        output[f"onesigma_{p}"] = np.sqrt(Finv_total[pi, pi])

    # Set reconstruction flag to original value
    fc["forecast"].recon = original_recon_flag

    return output


def compute_forecast_6parLCDM(
    fc,
    marginalize_over_Tb=True,
    omb_bbn_prior=0.0005,
    cmb=None,
    lensing=None,
    lensing_ell_min=30,
    lensing_ell_max=500,
    exclude_LSS=False,
):
    """Compute 6-parameter LCDM forecast.

    See docstring of `_compute_forecast_parameters` for parameter descriptions.
    """

    cos_basis = ["h", "log(A_s)", "n_s", "omega_cdm", "omega_b", "tau_reio"]
    cmb_fisher_basis_idx = [0, 1, 2, 3, 4, 5]

    LSS_basis = ["N", "alpha0", "b", "b2", "bs", "N2", "N4", "alpha2", "alpha4"]

    return _compute_forecast_parameters(
        cos_basis,
        LSS_basis,
        fc,
        cmb_fisher_basis_idx=cmb_fisher_basis_idx,
        marginalize_over_Tb=marginalize_over_Tb,
        omb_bbn_prior=omb_bbn_prior,
        cmb=cmb,
        lensing=lensing,
        lensing_ell_min=lensing_ell_min,
        lensing_ell_max=lensing_ell_max,
        exclude_LSS=exclude_LSS,
    )


def compute_forecast_Mnu(
    fc,
    marginalize_over_Tb=True,
    omb_bbn_prior=0.0005,
    cmb=None,
    lensing=None,
    lensing_ell_min=30,
    lensing_ell_max=500,
    exclude_LSS=False,
):
    """Compute forecast for one-parameter LCDM extension with free Mnu.

    See docstring of `_compute_forecast_parameters` for parameter descriptions.
    """
    cos_basis = ["h", "log(A_s)", "n_s", "omega_cdm", "omega_b", "tau_reio", "m_ncdm"]
    cmb_fisher_basis_idx = [0, 1, 2, 3, 4, 5, 6]

    LSS_basis = ["N", "alpha0", "b", "b2", "bs", "N2", "N4", "alpha2", "alpha4"]

    return _compute_forecast_parameters(
        cos_basis,
        LSS_basis,
        fc,
        cmb_fisher_basis_idx=cmb_fisher_basis_idx,
        marginalize_over_Tb=marginalize_over_Tb,
        omb_bbn_prior=omb_bbn_prior,
        cmb=cmb,
        lensing=lensing,
        lensing_ell_min=lensing_ell_min,
        lensing_ell_max=lensing_ell_max,
        exclude_LSS=exclude_LSS,
    )


def compute_forecast_Neff(
    fc,
    marginalize_over_Tb=True,
    omb_bbn_prior=0.0005,
    cmb=None,
    lensing=None,
    lensing_ell_min=30,
    lensing_ell_max=500,
    exclude_LSS=False,
):
    """Compute forecast for one-parameter LCDM extension with free Neff.

    See docstring of `_compute_forecast_parameters` for parameter descriptions.
    """
    cos_basis = ["h", "log(A_s)", "n_s", "omega_cdm", "omega_b", "tau_reio", "N_ur"]
    cmb_fisher_basis_idx = [0, 1, 2, 3, 4, 5, 7]

    LSS_basis = ["N", "alpha0", "b", "b2", "bs", "N2", "N4", "alpha2", "alpha4"]

    return _compute_forecast_parameters(
        cos_basis,
        LSS_basis,
        fc,
        cmb_fisher_basis_idx=cmb_fisher_basis_idx,
        marginalize_over_Tb=marginalize_over_Tb,
        omb_bbn_prior=omb_bbn_prior,
        cmb=cmb,
        lensing=lensing,
        lensing_ell_min=lensing_ell_min,
        lensing_ell_max=lensing_ell_max,
        exclude_LSS=exclude_LSS,
    )


def compute_forecast_OmegaK(
    fc,
    marginalize_over_Tb=True,
    omb_bbn_prior=0.0005,
    cmb=None,
    lensing=None,
    lensing_ell_min=30,
    lensing_ell_max=500,
    exclude_LSS=False,
):
    """Compute forecast for one-parameter LCDM extension with free OmegaK.

    See docstring of `_compute_forecast_parameters` for parameter descriptions.
    """
    cos_basis = ["h", "log(A_s)", "n_s", "omega_cdm", "omega_b", "tau_reio", "Omega_k"]
    cmb_fisher_basis_idx = [0, 1, 2, 3, 4, 5, 9]

    LSS_basis = ["N", "alpha0", "b", "b2", "bs", "N2", "N4", "alpha2", "alpha4"]

    return _compute_forecast_parameters(
        cos_basis,
        LSS_basis,
        fc,
        cmb_fisher_basis_idx=cmb_fisher_basis_idx,
        marginalize_over_Tb=marginalize_over_Tb,
        omb_bbn_prior=omb_bbn_prior,
        cmb=cmb,
        lensing=lensing,
        lensing_ell_min=lensing_ell_min,
        lensing_ell_max=lensing_ell_max,
        exclude_LSS=exclude_LSS,
    )
