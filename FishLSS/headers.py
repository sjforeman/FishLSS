"""Useful imports for FishLSS and jupyter notebooks.
"""

import os
import sys
from time import time
from timeit import timeit
from copy import copy

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

import numpy as np
import scipy
from scipy import special, optimize, integrate, stats
from scipy.special import hyp2f1, hyperu, gamma
from scipy.interpolate import (
    UnivariateSpline,
    RectBivariateSpline,
    interp1d,
    interp2d,
    BarycentricInterpolator,
)

import pyfftw
from classy import Class
from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD

from .experiment import *
from .fisherForecast import *
from .bao_recon.loginterp import loginterp
from .bao_recon.spherical_bessel_transform_fftw import SphericalBesselTransform
from .bao_recon.qfuncfft_recon import QFuncFFT
from .bao_recon.zeldovich_rsd_recon_fftw import Zeldovich_Recon
