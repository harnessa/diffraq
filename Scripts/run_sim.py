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
    'tel_diameter':         10e-3,
    'waves':                641e-9,
    'focal_length':         0.499,
    'defocus':              3e-3,
    'image_size':           64*2,

    ### Numerics ###
    'radial_nodes':         1000,
    'theta_nodes':          100,

    'target_center':        [0, 10e-3/(27.455/(77.455))],
    # 'occulter_shift':        [0, 5e-3*.35],

    ### Occulter ###
    # 'occulter_config':      f'{diffraq.occulter_dir}/bb_2017.cfg',

    ### Saving ###
    'do_save':          False,
    'save_dir_base':    f'{diffraq.results_dir}',
    'session':          'test',
    'save_ext':         '512',
    'free_on_end':False,
    'verbose':False,
}

num_pet = 12
max_apod = 0.9
occulter_is_finite = True
apod_dir = '/home/aharness/repos/diffraq/External_Data/Apodization_Profiles'

starshade = {'kind':'starshade', 'is_opaque':True, 'num_petals':num_pet, \
    'edge_file':f'{apod_dir}/bb_2017__inner.h5'}


for x in np.linspace(13e-3, 13e-3/.35*2, 10):
    # params['target_center'] = [0e-3,x]

    #Load simulation class and run sim
    sim = diffraq.Simulator(params, starshade)
    sim.run_sim()


    import matplotlib.pyplot as plt;plt.ion()
    print(x*1e3, sim.image[0].max())
    plt.imshow(sim.image[0])
    # plt.imshow(np.log10(sim.image[0]))
    # plt.imshow(np.log10(np.abs(sim.pupil[0])**2))
    breakpoint()
