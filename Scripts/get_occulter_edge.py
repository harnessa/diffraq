"""
get_occulter_edge.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-22-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Script to get occulter's edge profile

"""

import diffraq
import numpy as np

#User-input parameters
params = {

    ### Numerics ###
    'radial_nodes':     20292,
    'theta_nodes':      200,

    ### Occulter ###
    'occulter_file':   f'{diffraq.occulter_dir}/M12P2_joint.cfg',
}

#Load simulator + build edge
sim = diffraq.Simulator(params)
sim.occulter.build_edge()

#Get edge
edge = sim.occulter.edge.copy()

#Cleanup
sim.clean_up()

#Plot
import matplotlib.pyplot as plt;plt.ion()

plt.plot(*edge.T, 'x')

breakpoint()
