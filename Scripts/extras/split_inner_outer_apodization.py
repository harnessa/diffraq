"""
split_inner_outer_apodization.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 03-05-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Script to take a joint apodization function with inner and outer
    starshades (e.g., the lab starshade) and split into separate apodization files.

"""

import numpy as np
import diffraq
import h5py

############################
#####   USER INPUT     #####
############################

apod_name = 'bb_2017'
apod_dir = diffraq.apod_dir

############################
############################

#Load joint apodization function
with h5py.File(f'{apod_dir}/{apod_name}.h5', 'r') as f:
    data = f['data'][()]
rads = data[:,0]
apod = data[:,1]

#Get minimum radius and maximum apodization value
rmin = rads.min()
max_apod = apod.max()

#Get index of end of inner starshade (subtract one to go from strut to starshade)
inn_cut = np.where(apod >= max_apod)[0][0] - 1 + 3   #3 matches Stuart's loci for M12P6
data_inn = np.stack((rads, 1 - apod), 1)[:inn_cut]

#Get index of start of outer starshade (add one to go from strut to starshade)
out_cut = np.where(apod >= max_apod)[0][-1] + 1
data_out = data[out_cut:]

#Write out split apod
with h5py.File(f'{apod_dir}/{apod_name}__inner.h5', 'w') as f:
    f.create_dataset('note', data='Inner apodization of bb_2017')
    f.create_dataset('header', data='radius [m], apodization value')
    f.create_dataset('data', data=data_inn)

with h5py.File(f'{apod_dir}/{apod_name}__outer.h5', 'w') as f:
    f.create_dataset('note', data='Outer apodization of bb_2017')
    f.create_dataset('header', data='radius [m], apodization value')
    f.create_dataset('data', data=data_out)
