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
    'waves':                405e-9,
    'focal_length':         0.499,
    'defocus':              3e-3,
    'image_size':           64,

    ### Numerics ###
    'radial_nodes':         400,
    'theta_nodes':          50,

    ### Occulter ###
    'occulter_config':      f'{diffraq.occulter_dir}/bb_2017.cfg',

    ### Saving ###
    'do_save':          True,
    'save_dir_base':    f'{diffraq.results_dir}',
    'session':          'test',
    'save_ext':         '512',

}

#Load simulation class and run sim
sim = diffraq.Simulator(params)
sim.run_sim()
