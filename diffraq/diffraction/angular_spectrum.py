"""
angular_spectrum.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 11-12-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: class to hold Angular Spectrum

"""

from diffraq.quadrature import frequency_quad

class Angular_Spectrum(object):

    def __init__(self, sim):
        self.sim = sim

############################################
#####   Quadrature #####
############################################

    def build_spectrum_quad(self):
        self.xf, self.yf, self.wf = frequency_quad(wave, zz, m, n, Dmax, grid_pts, over_sample=6)
        breakpoint()

############################################
############################################

############################################
#####   Cleanup #####
############################################

    def clean_up(self):
        #Delete trash
        trash_list = ['xf', 'yf', 'wf']

        for att in trash_list:
            if hasattr(self, att):
                delattr(self, att)

############################################
############################################
