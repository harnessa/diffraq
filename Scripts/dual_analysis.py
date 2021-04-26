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
import matplotlib.pyplot as plt;plt.ion()

#User-input parameters
params = {

    ### Dual Parameters ###
    'load_dir_base_1':  f'{diffraq.results_dir}/new_OSS_journal',
    'session_1':        '',
    'load_ext_1':       '300nm',
    'load_ext_2':       '512',

    ### Analyzer Parameters ###
    'cam_analyzer':     ['p','o'][0],

    ### Contrast Parameters ###
    'is_contrast':      False,
    'calibration_file': f'{diffraq.results_dir}/Milestone_2/Mcal/image.h5',
    'max_apod':         0.9,
    'freespace_corr':   {641:1.07, 660:0.98, 699:1.07, 725:1.08},
}

#Load analysis class and plot results
duo = diffraq.Dual_Analyzer(params)
duo.show_results()

#Compare difference
dif = duo.alz1.image - duo.alz2.image
maxper = (1 - duo.alz1.image.max()/duo.alz2.image.max())*100
difper = dif.max()/duo.alz1.image.max()*100

plt.figure()
plt.imshow(dif)
print(dif.max(), maxper, difper)
print(duo.alz1.image.max(), duo.alz2.image.max())

breakpoint()
