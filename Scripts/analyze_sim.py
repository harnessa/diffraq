"""
analyze_sim.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-01-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Script to analyze results from DIFFRAQ simulation.

"""

import diffraq

#User-input parameters
params = {

    ### Loading ###
    'load_dir_base':    f'{diffraq.results_dir}/new_OSS_journal',
    'session':          '',
    'load_ext':         'sml_wave',
    'cam_analyzer':     [None, 'p', 'o'][0],
    'wave_ind':         0,
}

#Load analysis class and plot results
alz = diffraq.Analyzer(params)
alz.show_results()
