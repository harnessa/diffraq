"""
default_parameters.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Default parameters for DIFFRAQ simulations.

"""

import numpy as np

""" All units in [m] unless specificed others """

############################################
####	Default Parameters ####
############################################

def_params = {

    ### Simulator ###

    ### Saving ###
    'do_save':              False,      #Save data?
    'save_dir_base':        None,       #Base directory to save data
    'session':              '',         #Session name (next level folder)
    'save_ext':             '',         #Save extension
    'verbose':              True,       #Print to STD?

    ### Occulter ###
    'occulter_shape':       'circle',   #Shape of occulter. Options: [circle, polar, hypergaussian]
    'circle_rad':           12,         #Circle occulter radius
    'apod_func':            None,       #String defining apodization lambda function

    ### World ###
    'waves':                0.6e-6,     #Wavelengths
    'z0':                   1e19,       #Source - Occulter distance
    'z1':                   20e6,       #Occulter - Telescope distance
    'tel_diameter':         2.4,        #Telescope diameter
    'num_pts':              256,        #Number of points across telescope

    ### Numerics ###
    'fft_tol':              1e-9,       #Tolerance to feed to NUFFT

}
