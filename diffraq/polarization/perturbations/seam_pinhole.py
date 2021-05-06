"""
seam_pinhole.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 05-06-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class of the vector seam of the pinhole perturbation.

"""

import numpy as np
from diffraq.geometry import Pinhole

class Seam_Pinhole(Pinhole):

############################################
#####  Main Scripts #####
############################################

    def build_quadrature(self, sxq, syq, swq, sdq, snq, sgw):
        #TODO: add pinhole seam
        return sxq, syq, swq, sdq, snq, sgw

############################################
############################################
