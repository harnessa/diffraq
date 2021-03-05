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

############################
#####   USER INPUT     #####
############################

apod_name = 'bb_2017'
apod_dir = diffraq.apod_dir

############################
############################

#Load joint apodization function
rads, apod = np.genfromtxt(f'{apod_dir}/{apod_name}.txt', delimiter=',').T

#Get minimum radius and maximum apodization value
rmin = rads.min()
max_apod = apod.max()

#Get index of end of inner starshade (subtract one to go from strut to starshade)
inn_cut = np.where(apod >= max_apod)[0][0] - 1
ainn = 1 - apod[:inn_cut]           #Take inverse of apod
rinn = rads[:inn_cut]

#Get index of start of outer starshade (add one to go from strut to starshade)
out_cut = np.where(apod >= max_apod)[0][-1] + 1
aout = apod[out_cut:]
rout = rads[out_cut:]

#Write out Inner apod
with open(f'{apod_dir}/{apod_name}__inner.txt', 'w') as f:
    f.write(f'#Inner apodization\n')
    #Write out last radius
    f.write(f'#,{rinn[-1]}\n')
    #Write out all data
    for i in range(len(rinn)):
        f.write(f'{rinn[i]},{ainn[i]}\n')

#Write out Outer apod
with open(f'{apod_dir}/{apod_name}__outer.txt', 'w') as f:
    f.write(f'#Outer apodization\n')
    #Write out first radius
    f.write(f'#,{rout[0]}\n')
    #Write out all data
    for i in range(len(rout)):
        f.write(f'{rout[i]},{aout[i]}\n')
