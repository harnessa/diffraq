"""
run_sim.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Script to run DIFFRAQ simulation.

"""

import diffraq
import numpy as np

#User-input parameters
params = {

    ### World ###
    'num_pts':              512,
    'zz':                   49.986,
    'z0':                   27.455,
    'tel_diameter':         5e-3,
    'waves':                641e-9,
    'focal_length':         0.496,
    'defocus':              3e-3,
    'image_size':           128,

    ### Numerics ###
    'radial_nodes':         5000,
    'theta_nodes':          300,
    'seam_radial_nodes':    500,
    'seam_theta_nodes':     500,

    ### Vector ###
    'seam_width':           25e-6,
    'do_run_vector':        True,
    'is_sommerfeld':        True,

    ### Saving ###
    'do_save':              True,
    'save_dir_base':        f'{diffraq.results_dir}',
    'session':              'test',
    'save_ext':             'circle_vec',
}

circle_rad = 2e-3

shape = {'kind':'circle', 'is_opaque':False,'max_radius':circle_rad}

#Load simulation class and run sim
sim = diffraq.Simulator(params, shape)
sim.run_sim()
