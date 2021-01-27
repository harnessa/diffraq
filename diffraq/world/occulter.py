"""
occulter.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-15-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class to hold occulter shape: loci and area quadrature points.

"""

import numpy as np
from diffraq.quadrature import polar_quad, polar_edge, starshade_quad, starshade_edge

class Occulter(object):

    def __init__(self, sim):
        self.sim = sim
        self.approved_shapes = ['polar', 'circle', 'analytic_starshade', \
            'numeric_starshade']

############################################
#####  Main Functions #####
############################################

    def build_quadrature(self):
        if self.sim.occulter_shape in self.approved_shapes:
            getattr(self, f'build_quad_{self.sim.occulter_shape}')()
        else:
            self.sim.logger.error('Invalid Occulter Shape')

    def get_edge_points(self):
        if self.sim.occulter_shape in self.approved_shapes:
            return getattr(self, f'build_edge_{self.sim.occulter_shape}')()
        else:
            self.sim.logger.error('Invalid Occulter Shape')

############################################
############################################

############################################
#####  Polar Occulters #####
############################################

    #### Polar ####

    def build_quad_polar(self, apod_func=None):
        #Get apod function
        if apod_func is None:
            apod_func = self.sim.apod_func

        #Calculate polar quadrature
        self.xq, self.yq, self.wq = polar_quad(apod_func, \
            self.sim.radial_nodes, self.sim.theta_nodes)

    def build_edge_polar(self, apod_func=None, npts=None):
        #Get apod function
        if apod_func is None:
            apod_func = self.sim.apod_func

        #Theta nodes
        if npts is None:
            npts = self.sim.theta_nodes

        #Get polar edge
        bx, by = polar_edge(apod_func, -1, npts)

        #Stack together
        edge = np.dstack((bx, by)).squeeze()

        #Cleanup
        del bx, by

        return edge

    #### Circle ####

    def build_quad_circle(self):
        #Set apod function to constant radius
        apod_func = lambda t: self.sim.circle_rad*np.ones_like(t)

        #Build polar shape
        self.build_quad_polar(apod_func=apod_func)

    def build_edge_circle(self, npts=None):
        #Set apod function to constant radius
        apod_func = lambda t: self.sim.circle_rad*np.ones_like(t)

        #Build polar shape
        self.build_edge_polar(apod_func=apod_func, npts=npts)

############################################
############################################

############################################
#####  Starshade Occulters #####
############################################

    def build_quad_analytic_starshade(self):
        #Normalize radius

        breakpoint()

    def build_quad_numeric_starshade(self):

        breakpoint()

    def build_edge_starshade(self, apod):

        breakpoint()

############################################
############################################
