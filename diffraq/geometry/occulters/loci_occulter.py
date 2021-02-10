"""
loci_occulter.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Derived class of occulter defined by loci of edge points.

"""

import numpy as np
import diffraq.quadrature as quad
from diffraq.geometry import Occulter, Loci_Shape_Func

class Loci_Occulter(Occulter):

    name = 'loci'

############################################
#####  Startup #####
############################################

    def set_shape_function(self):
        self.shape_func = Loci_Shape_Func(self.get_loci, None)

############################################
############################################

############################################
#####  Main Shape #####
############################################

    def build_shape_quadrature(self):
        #Load loci data
        loci = self.get_loci()

        #Calculate loci quadrature
        xq, yq, wq = quad.loci_quad(loci[:,0], loci[:,1], self.sim.radial_nodes)

        #Cleanup
        del loci

        return xq, yq, wq

    def build_edge(self):
        return self.get_loci()

############################################
############################################

############################################
#####  Helper Functions #####
############################################

    def get_loci(self):
        return self.load_loci_data(self.sim.loci_file)

    def load_loci_data(self, loci_file):
        return np.genfromtxt(loci_file, delimiter=',')

############################################
############################################
