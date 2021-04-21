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

    ### Observation ###
    'waves':            0.6e-6,
    'tel_diameter':     2.4,
    'num_pts':          256,

    ### Saving ###
    'do_save':          True,
    'save_dir_base':    f'{diffraq.results_dir}',
    'session':          'test',
    'save_ext':         'x0y2_120',

}

params['beam_function'] = lambda x,y: np.exp(-((x-0)**2 + (y-2)**2)/(2.*120))


#Circle shape
shape = {'kind':'circle', 'is_opaque':True, 'max_radius':12}

#Load simulation class and run sim
sim = diffraq.Simulator(params, shape)
sim.run_sim()
