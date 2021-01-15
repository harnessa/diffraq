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

}

#Load simulation class and run sim
sim = diffraq.Simulator(params)
sim.run_sim()
