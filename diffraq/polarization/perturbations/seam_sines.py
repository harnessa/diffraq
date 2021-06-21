"""
seam_sines.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 05-24-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class of the vector seam of the sine waves perturbation.

"""

import numpy as np
from diffraq.geometry import Sines

class Seam_Sines(Sines):

############################################
#####  Main Scripts #####
############################################

    def build_quadrature(self, sxq, syq, swq, sdq, snq, sgw):
        #TODO: add sines seam
        return sxq, syq, swq, sdq, snq, sgw

############################################
############################################
