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
    'load_dir_base':    f'{diffraq.results_dir}/diffraq_test',
    'session':          'M12P2',
    'load_ext':         'joint_m4000__n200',

}

#Load analysis class and plot results
alz = diffraq.Analyzer(params)
alz.show_results()
