"""Utility functions for FishLSS.
"""
import os
import numpy as np
from scipy.interpolate import interp2d, RectBivariateSpline


class Interpolator2d(object):
    """Class for 2d interpolation.

    Interpolates values z along axes (x,y). The interpolator is designed to accept
    a single x value and an array of y values and return a 1d array. If x or y are
    outside the bounds of the input arrays, the edge value(s) of z are used in an
    intelligent way.
    """

    def __init__(self, x, y, z, bbox=[None, None, None, None], kx=3, ky=3, s=0):
        """Initialize interpolator.

        Parameters
        ----------
        x, y, z : array_like
            Input 1d arrays of x and y coordinates.
        z : array_like
            Input 2d array of z values, packed as [y, x].
        bbox, kx, ky, s
            Parameters for scipy.interpolate.RectBivariateSpline (see that docstring
            for details).
        """
        self.x_in = x
        self.y_in = y
        self.z_in = z

        self.x_min = self.x_in[0]
        self.x_max = self.x_in[-1]
        self.y_min = self.y_in[0]
        self.y_max = self.y_in[-1]

        self.spline = RectBivariateSpline(
            self.x_in, self.y_in, self.z_in.T, bbox=bbox, kx=kx, ky=ky, s=s
        )

    def evaluate(self, x, y):
        """Evaluate interpolator at single x value and array of y values.

        If x or y are outside the bounds used to define the interpolator, the values
        at the edges are used.

        Parameters
        ----------
        x : float
            x coordinate to evaluate at.
        y : array_like
            1d array of y coordinates to evaluate at.

        Returns
        -------
        result : np.ndarray
            1d array of interpolated values.
        """

        # Check that x is a single number and y is a 1d array
        if np.array(x).ndim != 0:
            raise InputError("x cannot be an array!")
        if np.array(y).ndim != 1:
            raise InputError("y must be a 1d array!")

        # Make array for results, filled with NaNs so we know if something has gone
        # wrong
        result = np.empty(y.shape)
        result[:] = np.nan

        if x < self.x_min:
            # If x < x_min, compute results at x_min
            result[y < self.y_min] = self.z_in[0, 0]
            result[y > self.y_max] = self.z_in[-1, 0]
            result[(y >= self.y_min) & (y <= self.y_max)] = self.spline.ev(
                self.x_min, y[(y >= self.y_min) & (y <= self.y_max)]
            )
        elif x > self.x_max:
            # If x > x_min, compute results at x_max
            result[y < self.y_min] = self.z_in[0, -1]
            result[y > self.y_max] = self.z_in[-1, -1]
            result[(y >= self.y_min) & (y <= self.y_max)] = self.spline.ev(
                self.x_max, y[(y >= self.y_min) & (y <= self.y_max)]
            )
        else:
            # If x is within bounds, still handle out-of-bounds y values carefully
            result[y < self.y_min] = self.spline.ev(x, self.y_min)
            result[y > self.y_max] = self.spline.ev(x, self.y_max)
            result[(y >= self.y_min) & (y <= self.y_max)] = self.spline.ev(
                x, y[(y >= self.y_min) & (y <= self.y_max)]
            )

        return result
