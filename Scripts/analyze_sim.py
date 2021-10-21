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
    'load_dir_base':    f'{diffraq.results_dir}/m2_test',
    # 'load_dir_base':    '/home/aharness/repos/Milestone_2/diffraq_analysis/final_run',
    # 'session':          'Diffraq_Results_final/M12P6_vector',
    'session':          'vec_comps',
    'load_ext':         'bb_2017',
    'cam_analyzer':     ['p', 'o'][0],
    'wave_ind':         2,

    ### Contrast Parameters ###
    'is_contrast':      True,
    'calibration_file': f'{diffraq.results_dir}/Milestone_2/Mcal/image.h5',
    'max_apod':         0.9,
    'freespace_corr':   {641:0.89, 660:0.83, 699:0.96, 725:0.96},

}

#Load analysis class and plot results
alz = diffraq.Analyzer(params)
alz.show_results()
