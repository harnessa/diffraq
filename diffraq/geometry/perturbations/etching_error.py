"""
etching_error.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-09-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class of the etching error perturbation.

"""

import numpy as np
from diffraq.geometry import Perturbation
from scipy.misc import derivative

def add_polar_etching(fr, dr, etch):

    #Build new functions that use normal vector to point to new shape (flipped etch, so positive etch points outward)
    # nfr = lambda t: fr(t) +
    import matplotlib.pyplot as plt;plt.ion()

    t = np.linspace(0,2*np.pi, 1000)
    xx = fr(t) * np.cos(t)
    yy = fr(t) * np.sin(t)


    # import matplotlib.pyplot as plt;plt.ion()
    # t = np.linspace(0,2*np.pi, 1000)[:,None]
    # xy = fxy(t)
    # xy2 = nfxy(t)
    # plt.cla()
    # plt.plot(*xy.T)
    # plt.plot(*xy2.T)
    #
    # breakpoint()


    breakpoint()

def add_cartesian_etching(fxy, dxy, etch):

    #Build new functions that use normal vector to point to new shape (flipped etch, so positive etch points outward)
    nfxy = lambda t: \
        fxy(t) + etch*np.array([1., -1])*dxy(t)[:,::-1]/np.hypot(*dxy(t).T)[:,None]

    #Build new derivative function
    def ndxy(t):
        fp = dxy(t)
        fpp = derivative(fxy, t, dx=t[1]-t[0], n=2) #Second derivative
        norm = np.hypot(fp[:,0], fp[:,1])
        return fp + etch*np.array([1, -1])*(fpp[:,::-1] - \
            fp[:,::-1]*(np.sum(fp*fpp,1)/norm**2)[:,None])/norm[:,None]

    return nfxy, ndxy
