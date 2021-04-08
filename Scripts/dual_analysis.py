"""
dual_analysis.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 04-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Script to analyze results from two DIFFRAQ simulations.

"""

import diffraq

#User-input parameters
params = {

    ### Dual Parameters ###
    'load_dir_base_1':  f'{diffraq.results_dir}/Milestone_2',
    'session_1':        'M12P2_scalar',
    'load_ext_1':       'mask_1a',
    'load_ext_2':       'mask_1b',

    ### Analyzer Parameters ###
    'cam_analyzer':     'p',

}

#Load analysis class and plot results
duo = diffraq.Dual_Analyzer(params)
duo.show_results()
