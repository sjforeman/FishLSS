"""HI stochastic noise models from simulations.

Measured by Andrej Obuljen, imported into Python by Simon Foreman
"""
import os
import numpy as np
from scipy.interpolate import interp1d, interp2d


def get_TNG_Pstoch_models(dir_):
    """Read IllustrisTNG HI noise models from disk.

    Parameters
    ----------
    dir_ : string
        Directory containing `obuljen` directory, which itself contains simulation
        measurements.

    Returns
    -------
    output : dict
        dict containing redshifts, wavenumbers, Pstoch values (packed as [k,z]),
        and Psampling values (packed as [z]).
    """

    Psampling_data = np.genfromtxt(os.path.join(dir_, "obuljen/TNG_z_Psampling.txt"))
    z = Psampling_data[:, 0]
    Pstoch_values_white = Psampling_data[:, 1]

    Pstoch_data = np.genfromtxt(os.path.join(dir_, "obuljen/TNG_k_HIstoch_z.txt"))
    k = Pstoch_data[:, 0]
    Pstoch_values = Pstoch_data[:, 1:]

    return {
        "z": z,
        "k": k,
        "Pstoch_values": Pstoch_values,
        "Pstoch_values_white": Pstoch_values_white,
    }
