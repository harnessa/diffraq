"""
occulter.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-15-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Base class of an occulter shape and used to generate quadrature points.

"""

import numpy as np
import diffraq.quadrature as quad
import diffraq.geometry as geometry

class Occulter(object):

    def __init__(self, sim):
        self.sim = sim
        #Etching error sign is flipped for Babinet
        self.bab_sign = [1, -1][self.sim.is_babinet]
        #Set shape function
        self.set_shape_function()

############################################
#####  Quadrature #####
############################################

    def build_quadrature(self):

        #Build shape quadrature
        self.xq, self.yq, self.wq = self.build_shape_quadrature()

        #Loop through perturbation list and add quadratures
        for kind, pms in self.sim.perturbations:

            #Build perturbation
            pert = getattr(geometry, kind)(self, **pms)

            #Build perturbations's quadrature (which adds to current [xq,yq,wq])
            pert.build_quadrature()

############################################
############################################

############################################
#####  Edge Points #####
############################################

    def build_edge(self):

        #Build edge points
        self.edge = self.build_shape_edge()

        #Loop through perturbation list and add edge points
        for kind, pms in self.sim.perturbations:

            #Build perturbation
            pert = getattr(geometry, kind)(self, **pms)

            #Build perturbations's quadrature (which adds to current [xq,yq,wq])
            pert.build_edge_points()

        #Sort by angle
        #TODO: fix sorting for multiple, unconnected apertures
        self.edge = self.edge[np.argsort(np.arctan2(self.edge[:,1], self.edge[:,0]))]

############################################
############################################
