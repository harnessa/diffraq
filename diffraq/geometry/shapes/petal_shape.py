"""
petal_shape.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Derived class of starshade occulter.

"""

import numpy as np
import diffraq.quadrature as quad
from diffraq.geometry import Shape, PetalOutline

class PetalShape(Shape):

############################################
#####  Startup #####
############################################

    def set_outline(self):
        #Get starshade apodization (edge) function (file takes precedence)
        if self.edge_file is not None:
            #Load data from file and get interpolation function
            edge_func = self.load_edge_file(self.edge_file)

        else:
            #Use user-supplied apodization function
            edge_func = self.edge_func

        self.outline = PetalOutline(edge_func, self.edge_diff, \
            num_petals=self.num_petals)

############################################
############################################

############################################
#####  Main Shape #####
############################################

    def build_local_shape_quad(self):
        #Calculate starshade quadrature
        xq, yq, wq = quad.starshade_quad(self.outline.func, self.num_petals, \
            self.min_radius, self.max_radius, self.radial_nodes, self.theta_nodes, \
            is_opaque=self.is_opaque)

        return xq, yq, wq

    def build_local_shape_edge(self, npts=None):
        #Radial nodes
        if npts is None:
            npts = self.radial_nodes

        #Calculate starshade edge
        edge = quad.starshade_edge(self.outline.func, self.num_petals, \
            self.min_radius, self.max_radius, npts)

        return edge

############################################
############################################

############################################
#####  Helper Functions #####
############################################

    def load_edge_file(self, edge_file):
        #Load file
        data = np.genfromtxt(edge_file, delimiter=',')

        #Replace min/max radius
        self.min_radius = data[:,0].min()
        self.max_radius = data[:,0].max()

        return data

############################################
############################################

class StarshadeShape(PetalShape):

    pass
