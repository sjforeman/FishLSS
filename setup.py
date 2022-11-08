import sys

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

import numpy as np

# Enable OpenMP support if available
if sys.platform == "darwin":
    compile_args = []
    link_args = []
else:
    compile_args = ["-fopenmp"]
    link_args = ["-fopenmp"]

setup(
    name="FishLSS",
    version=0.1,
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.6",
    author="Noah Sailer",
    description="Forecasting code for LSS surveys",
)