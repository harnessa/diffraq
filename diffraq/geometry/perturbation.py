"""
perturbation.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-09-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Base class of a perturbation to generate quadrature in addition
    to the occulter.

"""

import numpy as np

class Perturbation(object):

    def __init__(self, shape_func, **kwargs):
        #Point to parent occulter's shape function
        self.shape_func = shape_func

        #Set perturbation-specific keywords
        for k,v in kwargs.items():
            setattr(self, k, v)


    def build_quadrature(self, edge):

        #Get edge point closest to center
        self.shape_func.closest_point(self.center)

        import matplotlib.pyplot as plt;plt.ion()
        plt.plot(*edge.T)
        breakpoint()
