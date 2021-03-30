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

for aa in apods:

    #Load text data
    data = np.genfromtxt(f'{diffraq.apod_dir}/{aa}.txt', delimiter=',')

    #Save as h5py
    with h5py.File(f'{diffraq.apod_dir}/{aa}.h5', 'w') as f:
        f.create_dataset('data', data=data)
