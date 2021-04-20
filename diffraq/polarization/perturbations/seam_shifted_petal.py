"""
seam_shifted_petal.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 04-19-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class of the vector seam of the shifted petal perturbation.

"""

import numpy as np
from diffraq.geometry import Shifted_Petal
import diffraq.polarization as polar
import diffraq.quadrature as quad

class Seam_Shifted_Petal(Shifted_Petal):

############################################
#####  Main Scripts #####
############################################

    def build_quadrature(self, sxq, syq, swq, sdq, snq, sgw):
        #Change parent's quad points to shift petal
        sxq, syq = self.shift_petal_points(sxq, syq)

        return sxq, syq, swq, sdq, snq, sgw

############################################
############################################

############################################
#####  Shifting Functions #####
############################################

    def shift_petal_points(self, xp, yp):

        #Find points between specified angles
        inds = self.find_between_angles(xp, yp)

        #Don't shift inner circle
        inds = inds & (np.hypot(xp, yp) > self.parent.min_radius)

        #Get mean angle and width
        ang_avg = self.angles.mean()
        ang_wid = np.abs(self.angles[1] - self.angles[0])

        #Shift petal along spine of petal tip
        xp[inds] += np.cos(ang_avg) * self.shift
        yp[inds] += np.sin(ang_avg) * self.shift

        #Cleanup
        del inds

        return xp, yp

############################################
############################################
