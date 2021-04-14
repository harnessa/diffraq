"""
convert_apod_txt2h5.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 03-30-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Script to convert apodization function text files to h5py files
    for faster loading speeds.

"""

import numpy as np
import diffraq
import h5py

apods = ['bb_2017__inner', 'bb_2017__inner', 'bb_2017', 'wfirst_ni2']
apods = ['frick_yk']

is_lotus = False

for aa in apods:

    #Load text data
    data = np.genfromtxt(f'{diffraq.apod_dir}/{aa}.txt', delimiter=',')

    #Convert from microns to meters and scale apod if from lotus
    if is_lotus:
        data[:,0] *= 1e-6
        data[:,1] *= 0.9

    #Save as h5py
    with h5py.File(f'{diffraq.apod_dir}/{aa}.h5', 'w') as f:
        f.create_dataset('data', data=data)
