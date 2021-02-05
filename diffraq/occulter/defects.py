"""
defects.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-04-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Functions to add various defects and perturbations to the occulter shape.

"""

import numpy as np

############################################
#####  Etching Error #####
############################################

def add_polar_etching(apod_func, etch):


    import matplotlib.pyplot as plt;plt.ion()

    the = np.linspace(0,2*np.pi, 1000)
    xx = apod_func(the) * np.cos(the)
    yy = apod_func(the) * np.sin(the)

    plt.plot(xx, yy)

    breakpoint()

############################################
############################################
