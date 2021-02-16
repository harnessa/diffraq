"""
shape.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-15-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Base class of an occulter shape, used to generate quadrature points.

"""

import numpy as np

class Shape(object):

    def __init__(self, parent, **kwargs):
        """
        Keyword arguments (all not used by every shape):
            - kind:         kind of shape object
            - edge_func:    lambda function describing the shape's edge; f(theta)
            - edge_diff:    lambda function describing the derivative of the shapes edge; f'(theta)
            - loci_file:    filename that holds the (x,y) coordinates describing the occulter edge
            - edge_file:    filename that holds numerical apodization function as a function of radius. Supercedes apod_func
            - is_opaque:    described shape is opaque?
            - num_petals:   number of petals
            - min_radius:   minimum radius
            - max_radius:   maximum radius
            - radial_nodes: number of radial quadrature nodes
            - theta_nodes:  number of azimuthal quadrature nodes
            - perturbations: List of dictionaries describing perturbations to be added to the shape
        """

        #Point to parent [occulter]
        self.parent = parent

        #Default parameters
        def_params = {'kind':'polar', 'edge_func':None, 'edge_diff':None, \
            'loci_file':None, 'edge_file':None, 'is_opaque':False, \
            'num_petals':16, 'min_radius':0, 'max_radius':12, \
            'radial_nodes':self.parent.sim.radial_nodes, \
            'theta_nodes':self.parent.sim.theta_nodes, 'perturbations':[]}

        #Set Default parameters
        for k,v in {**def_params, **kwargs}.items():
            setattr(self, k, v)

        #Set outline function
        self.set_outline()
