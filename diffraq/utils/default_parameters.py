"""
default_parameters.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Default parameters for DIFFRAQ simulations.

"""

import diffraq
import numpy as np

""" All units in [m] unless specificed others """

#Default base directory
base_dir = rf"{diffraq.pkg_home_dir}/Results"

############################################
####	Default Parameters ####
############################################

def_params = {
    'tt':           1,      #1
}
