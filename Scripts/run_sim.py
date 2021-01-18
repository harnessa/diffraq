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

#User-input parameters
params = {
    ### Saving ###
    'do_save':          True,
    'session':          'test',

    ### World ###
    'waves':            0.641e-6,
    'tel_diameter':     3,
    'num_pts':          256,

    ### Occulter ###
    # ''
}

#Load simulation class and run sim
sim = diffraq.Simulator(params)
sim.run_sim()
