"""
starshade_occulter.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Derived class of starshade occulter.

"""

import numpy as np
import diffraq.quadrature as quad
from diffraq.geometry import Occulter, Radial_Shape_Func

class Starshade_Occulter(Occulter):

    name = 'starshade'

############################################
#####  Startup #####
############################################

    def set_shape_function(self):
        #Get starshade apodization function (file takes precedence)
        if self.sim.apod_file is not None:
            #Load data from file and get interpolation function
            apod_func = self.load_apod_file(self.sim.apod_file)

        else:
            #Use user-supplied apodization function
            apod_func = self.sim.apod_func

        self.shape_func = Radial_Shape_Func(apod_func, self.sim.apod_diff)

############################################
############################################

############################################
#####  Main Shape #####
############################################

    def build_shape_quadrature(self):
        #Calculate starshade quadrature
        xq, yq, wq = quad.starshade_quad(self.shape_func.func, self.sim.num_petals, \
            self.sim.ss_rmin, self.sim.ss_rmax, self.sim.radial_nodes, self.sim.theta_nodes, \
            is_babinet=self.sim.is_babinet)

        return xq, yq, wq

    def build_edge(self, npts=None):
        #Radial nodes
        if npts is None:
            npts = self.sim.radial_nodes

        #Calculate starshade edge
        edge = quad.starshade_edge(self.shape_func.func, self.sim.num_petals, \
            self.sim.ss_rmin, self.sim.ss_rmax, npts)

        return edge

############################################
############################################

############################################
#####  Helper Functions #####
############################################

    def load_apod_file(self, apod_file):
        #Load file
        data = np.genfromtxt(apod_file, delimiter=',')

        #Replace min/max radius
        self.sim.ss_rmin = data[:,0].min()
        self.sim.ss_rmax = data[:,0].max()

        return data

############################################
############################################
