"""
loci_outline.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 12-03-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class that holds the outline of a shape given explicitly with loci.

"""

import numpy as np

class LociOutline(object):

    def __init__(self, parent, data, with_2nd=False, etch_error=None):

        self.parent = parent    #Shape
        self._data = data       #(theta, x, y)

        #Store angle
        self._angle = np.arctan2(data[:,1], data[:,0]) % (2*np.pi)

        #TODO: add etch error

    ############################################

    def func(self, all_t, minds=None):
        #Find closest angle and return points there
        if minds is None:
            minds = [np.argmin(np.abs(self._angle - t)) for t in np.atleast_1d(all_t)]
        return self._data[minds]

    def diff(self, all_t, minds=None):
        #TODO: add derivative
        return np.zeros((len(all_t), 2))

    def diff_2nd(self, all_t, minds=None):
        return np.zeros((len(all_t), 2))

############################################
############################################
