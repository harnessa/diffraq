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

    ### World ###
    'waves':            0.6e-6,
    'tel_diameter':     2.4,
    'num_pts':          256,

    ### Saving ###
    'do_save':          True,
    'session':          'test',
    'with_log':         False,

}

#Load simulation class and run sim
sim = diffraq.Simulator(params)
sim.run_sim()
