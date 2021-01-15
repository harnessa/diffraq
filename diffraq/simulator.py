"""
simulator.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Main class to control simulations for the DIFFRAQ python package.
    #TODO: ADD MORE

"""

import diffraq
import numpy as np

class Simulator(object):

    def __init__(self, params={}, is_analysis=False):
        self.is_analysis = is_analysis
        self.set_parameters(params)

############################################
####	Initialize  ####
############################################

    def set_parameters(self, params):
        #Set parameters
        diffraq.util.set_default_params(self, params, diffraq.utils.def_params)

############################################
############################################

############################################
####	Main Script  ####
############################################

    def run_sim(self):

        breakpoint()

############################################
############################################
