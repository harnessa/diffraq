"""
loci_shape.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Derived class of Shape defined by loci of edge points.

"""

import numpy as np
import diffraq.quadrature as quad
from diffraq.geometry import Shape

class LociShape(Shape):

    kind = 'loci'

############################################
#####  Main Shape #####
############################################

    def build_local_shape_quad(self):
        #Load loci data
        loci = self.get_loci()

        #Calculate loci quadrature
        xq, yq, wq = quad.loci_quad(loci[:,0], loci[:,1], self.radial_nodes)

        #Cleanup
        del loci

        return xq, yq, wq

    def build_local_shape_edge(self):
        return self.get_loci()

############################################
############################################

############################################
#####  Helper Functions #####
############################################

    def get_loci(self):
        return self.load_loci_data(self.loci_file)

    def load_loci_data(self, loci_file):
        return np.genfromtxt(loci_file, delimiter=',')

############################################
############################################
