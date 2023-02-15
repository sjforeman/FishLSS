"""Utility functions for FishLSS.
"""
import os
import numpy as np

from scipy.interpolate import LinearNDInterpolator


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

        x_mesh, y_mesh = np.meshgrid(self.x_in, self.y_in)
        x_mesh = x_mesh.flatten()
        y_mesh = y_mesh.flatten()

        self.interp = LinearNDInterpolator(
            list(zip(x_mesh, y_mesh)), self.z_in.flatten(), rescale=True
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
        y = np.array(y)
        if y.ndim != 1:
            raise InputError("y must be a 1d array!")

        # Make array for results, filled with NaNs so we know if something has gone
        # wrong
        result = np.empty(y.shape)
        result[:] = np.nan

        # Make mask that selects y values that are within interpolator domain
        y_mask = (y >= self.y_min) & (y <= self.y_max)

        if x < self.x_min:
            # If x < x_min, compute results at x_min
            result[y < self.y_min] = self.z_in[0, 0]
            result[y > self.y_max] = self.z_in[-1, 0]
            result[y_mask] = self.interp(
                self.x_min * np.ones_like(y[y_mask]), y[y_mask]
            )
        elif x > self.x_max:
            # If x > x_min, compute results at x_max
            result[y < self.y_min] = self.z_in[0, -1]
            result[y > self.y_max] = self.z_in[-1, -1]
            result[y_mask] = self.interp(
                self.x_max * np.ones_like(y[y_mask]), y[y_mask]
            )
        else:
            # If x is within bounds, still handle out-of-bounds y values carefully
            # print(y, self.y_min, self.y_max, x, y[(y >= self.y_min) & (y <= self.y_max)])
            result[y < self.y_min] = self.interp(x, self.y_min)
            result[y > self.y_max] = self.interp(x, self.y_max)
            result[y_mask] = self.interp(x * np.ones_like(y[y_mask]), y[y_mask])

        return result
