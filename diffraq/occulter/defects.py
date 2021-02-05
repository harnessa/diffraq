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

def add_polar_etching(fr, etch):


    import matplotlib.pyplot as plt;plt.ion()

    t = np.linspace(0,2*np.pi, 1000)
    xx = fr(the) * np.cos(t)
    yy = fr(the) * np.sin(t)

    plt.plot(xx, yy)

    breakpoint()

def add_cartesian_etching(fxy, dxy, etch):

    #Build new functions that use normal vector to point to new shape (flipped etch, so positive etch points outward)
    nfxy = lambda t: fxy(t) + etch*np.array([1., -1])*dxy(t)[:,::-1]/np.hypot(*dxy(t).T)[:,None]

    #Build new derivative function
    def ndxy(t):
        fp = dxy(t)
        fpp = (np.roll(fp, -1, 0) - fp)/(t[1]-t[0]) #Second derivative
        norm = np.hypot(*dxy(t).T)
        return fp + etch*np.array([1, -1])*(fpp[:,::-1] - \
            fp[:,::-1]*(np.sum(fp*fpp,1)/norm**2)[:,None])/norm[:,None]

    # import matplotlib.pyplot as plt;plt.ion()
    # t = np.linspace(0,2*np.pi, 1000)[:,None]
    # xy = fxy(t)
    # xy2 = nfxy(t)
    # plt.cla()
    # plt.plot(*xy.T)
    # plt.plot(*xy2.T)
    #
    # breakpoint()


    return nfxy, ndxy

############################################
############################################
