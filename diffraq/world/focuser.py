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

import diffraq
import numpy as np

class Focuser(object):

    def __init__(self, sim):
        self.sim = sim
        self.copy_params()

############################################
#####  Start/Finish #####
############################################

    def copy_params(self):
        pms = []
        for k in pms:
            setattr(self, k, getattr(self.sim, k))

    def start_up(self):
        pass

    def close_up(self):
        pass

############################################
############################################
