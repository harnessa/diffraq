"""
shiftedPetal.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-12-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class of the shifted petal perturbation.

"""

import numpy as np
import diffraq.quadrature as quad
from diffraq.geometry import Perturbation

class ShiftedPetal(Perturbation):

    kind = 'shiftedPetal'

############################################
#####  Shared Quad + Edge #####
############################################

    def get_pert_quad(self, t0, tf, m, n, bab_sign):
        """ + direction = more material, - direction = less material"""

        #Get nodes
        xq, yq, wq = getattr(self, \
            f'get_quad_{self.shape_func.kind}')(t0, tf, m, n, bab_sign)

        return xq, yq, wq

    def get_pert_edge(self, t0, tf, m, n, bab_sign):

        import matplotlib.pyplot as plt;plt.ion()
        breakpoint()

############################################
############################################

############################################
#####  Petal Specific Quad #####
############################################

    def get_quad_petal(self, t0, tf, m, n, bab_sign):

        import matplotlib.pyplot as plt;plt.ion()
        breakpoint()
