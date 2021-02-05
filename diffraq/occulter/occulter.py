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
import diffraq.quadrature as quad
from diffraq.occulter import defects

class Occulter(object):

    def __init__(self, sim):
        self.sim = sim
        self.approved_shapes = ['polar', 'circle', 'starshade', 'cartesian']
        #Check shape name is in approved lost
        if self.sim.occulter_shape not in self.approved_shapes:
            self.sim.logger.error('Invalid Occulter Shape')

############################################
#####  Main Wrappers #####
############################################

    def build_quadrature(self):
        return getattr(self, f'build_quad_{self.sim.occulter_shape}')()

    def get_edge_points(self):
        return getattr(self, f'build_edge_{self.sim.occulter_shape}')()

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

        #Add etching error if applicable
        if self.sim.etching_error != 0:
            apod_func = defects.add_polar_etching(apod_func, self.sim.etching_error)

        #Calculate polar quadrature
        self.xq, self.yq, self.wq = quad.polar_quad(apod_func, \
            self.sim.radial_nodes, self.sim.theta_nodes)

    def build_edge_polar(self, apod_func=None, npts=None):
        #Get apod function
        if apod_func is None:
            apod_func = self.sim.apod_func

        #Theta nodes
        if npts is None:
            npts = self.sim.theta_nodes

        #Get polar edge
        edge = quad.polar_edge(apod_func, npts)

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
        return self.build_edge_polar(apod_func=apod_func, npts=npts)

############################################
############################################

############################################
#####  Cartesian Occulter #####
############################################

    def build_quad_cartesian(self, apod_func=None, apod_deriv=None):
        #Get apod function and derivative
        if apod_func is None:
            apod_func = self.sim.apod_func
        if apod_deriv is None:
            apod_deriv = self.sim.apod_deriv

        #Add etching error if applicable
        if self.sim.etching_error != 0:
            apod_func = defects.add_cartesian_etching(*apod_func, *apod_deriv, \
                self.sim.etching_error)

        #Calculate polar quadrature
        self.xq, self.yq, self.wq = quad.cartesian_quad(*apod_func, *apod_deriv, \
            self.sim.radial_nodes, self.sim.theta_nodes)

    def build_edge_cartesian(self, apod_func=None, npts=None):
        #Get apod function
        if apod_func is None:
            apod_func = self.sim.apod_func

        #Theta nodes
        if npts is None:
            npts = self.sim.theta_nodes

        #Get polar edge
        edge = quad.cartesian_edge(*apod_func, npts)

        return edge

############################################
############################################

############################################
#####  Starshade Occulters #####
############################################

    def build_quad_starshade(self):
        #Get apodization function
        apod_func = self.get_starshade_apod()

        #Calculate starshade quadrature
        self.xq, self.yq, self.wq = quad.starshade_quad(apod_func, self.sim.num_petals, \
            self.sim.ss_rmin, self.sim.ss_rmax, self.sim.radial_nodes, self.sim.theta_nodes, \
            is_babinet=self.sim.is_babinet)

    def build_edge_starshade(self, npts=None):
        #Get apodization function
        apod_func = self.get_starshade_apod()

        #Radial nodes
        if npts is None:
            npts = self.sim.radial_nodes

        #Calculate starshade edge
        edge = quad.starshade_edge(apod_func, self.sim.num_petals, \
            self.sim.ss_rmin, self.sim.ss_rmax, npts)

        return edge

    def get_starshade_apod(self):
        #Get starshade apodization function
        if self.sim.apod_file is not None:
            #Load data from file and get interpolation function
            apod_func = self.interp_apod_file(self.sim.apod_file)

        elif self.sim.apod_func is not None:
            #Use user-supplied apodization function
            apod_func = self.sim.apod_func

        else:
            #Raise error
            self.sim.logger.error('Starshade apodization not supplied')

        #Add etching error if applicable
        if self.sim.etching_error != 0:
            apod_func = defects.add_starshade_etching(apod_func, self.sim.etching_error)

        return apod_func

############################################
############################################

############################################
#####  Helper Functions #####
############################################

    def interp_apod_file(self, apod_file):
        #Load file
        data = np.genfromtxt(apod_file, delimiter=',')

        #Build interpolation function
        apod_func = lambda r: np.interp(r, data[:,0], data[:,1], left=1., right=0.)

        #Replace min/max radius
        self.sim.ss_rmin = data[:,0].min()
        self.sim.ss_rmax = data[:,0].max()

        return apod_func

############################################
############################################
