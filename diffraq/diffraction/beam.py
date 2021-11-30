"""
beam.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 11-29-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class to hold random phase screens and other functions to
    be applied to the input beam.
"""

import numpy as np
from diffraq.quadrature import polar_quad
from diffraq.diffraction import Phase_Screen
import scipy.fft as fft

class Beam(object):

    def __init__(self, screens, sim):
        self.sim = sim
        self.set_derived_parameters(screens)

    def set_derived_parameters(self, screens):

        #Return if no screens
        if screens is None:
            self.screens = []
            return

        #Turn into list
        if not isinstance(screens, list):
            screens = [screens]

        #Loop through and build screens
        self.screens = []
        for scn in screens:
            self.screens.append(Phase_Screen(scn, self))

        #Build Ang Spec for frequency space
        self.fx, self.fy, self.fw = polar_quad(lambda t: np.ones_like(t), \
            self.sim.angspec_radial_nodes, self.sim.angspec_theta_nodes)

############################################
#####  Main Function #####
############################################

    def apply_function(self, xq, yq, wq, wave, fld_1=None, fld_2=None):

        #Return immediately if no screens or beam function
        if self.sim.beam_function is None and len(self.screens) == 0:
            if fld_2 is None:
                return fld_1
            else:
                return fld_1, fld_2

        #Build beam
        beam = np.ones_like(fld_1) + 0j

        #Apply beam function
        if self.sim.beam_function is not None:
            #Get function
            beam *= self.sim.beam_function(xq, yq, wave)

        #Apply screens
        for scn in self.screens:
            #Get function
            beam *= scn.get_field(xq, yq, wq, wave)

        #Apply to inputs
        if fld_2 is None:
            fld_1 = fld_1 * beam
            del beam
            return fld_1
        else:
            fld_1 = fld_1 * beam
            fld_2 = fld_2 * beam
            del beam
            return fld_1, fld_2

############################################
############################################

############################################
#####  Clean up #####
############################################

    def clear_screen_maps(self):
        for scn in self.screens:
            scn.stored_map = None

    def clean_up(self):
        #Delete trash
        trash_list = ['fx', 'fy', 'fw']

        for att in trash_list:
            if hasattr(self, att):
                delattr(self, att)

############################################
############################################
