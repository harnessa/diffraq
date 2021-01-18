"""
get_grid_points.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-15-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: build uniformly spaced target grid from number of points and size.

"""

import numpy as np

def get_grid_points(ngrid, width):
    """
    grid_pts = get_grid_points(ngrid, width)

    build uniformly spaced target grid from number of points and size

    Inputs:
        ngrid = length of one size of target grid
        width = physical width of target [m]

    Outputs:
        grid_pts = target grid points (1D)
    """

    #Build target grid
    grid_pts = width*(np.arange(ngrid)/ngrid - 0.5)
    #Grid spacing
    dx = width/ngrid
    #Handle the odd case
    if ngrid % 2 == 1:
        grid_pts += dx/2

    return grid_pts
