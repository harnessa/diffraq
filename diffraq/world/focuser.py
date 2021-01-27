"""
focuser.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-18-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class to propagate the diffracted field to the focal plane of the
    target imaging system.
"""

import numpy as np

class Focuser(object):

    def __init__(self, sim):
        self.sim = sim

############################################
#####  Main Functions #####
############################################

############################################
############################################
