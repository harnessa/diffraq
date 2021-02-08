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
        #Etching error sign is flipped for Babinet
        self.bab_etch = [1, -1][self.sim.is_babinet]
        #Set shape function
        self.set_shape_function()

############################################
#####  Main Wrappers #####
############################################

    def build_quadrature(self):
        #Build shape quadrature
        self.xq, self.yq, self.wq = self.build_shape_quadrature()

    def get_edge_points(self):
        return self.build_edge()

############################################
############################################
