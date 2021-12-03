"""
rectangle_shape.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 12-03-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Wrappers for rectangle shape.

"""

import numpy as np
from diffraq.geometry import Shape
import diffraq.geometry as geometry
import diffraq.quadrature as quad

class RectangleShape(Shape):

############################################
#####  Outline #####
############################################

    def set_outline(self):

        #Over sample
        num_pts = max(self.radial_nodes, self.theta_nodes) * 2

        #Get etch error
        if self.etch_error is None:
            etch_err = 0
        else:
            etch_err = self.etch_error
        self.etch_error = None         #Clear etch error

        #Get size of rectangle, with etch error included
        radx = self.width/2 + etch_err
        rady = self.height/2 + etch_err
        xline = np.linspace(-radx, radx, num_pts)
        yline = np.linspace(-rady, rady, num_pts)
        ones = np.ones(num_pts)

        #Build edge data (top, right, bottom, left)
        xdata = np.concatenate((xline, ones*radx, xline[::-1], -ones*radx))
        ydata = np.concatenate((ones*rady, yline[::-1], -ones*rady, yline))
        edge_data = np.stack((xdata, ydata), 1)

        #Flip to go CCW
        edge_data = edge_data[::-1]

        #Cleanup
        del xline, yline, ones, xdata, ydata

        #Replace min/max radius
        self.min_radius = min(radx, rady)
        self.max_radius = np.hypot(radx, rady)

        #Force to use cartesian interp outline
        self.outline = geometry.LociOutline(self, edge_data)

############################################
############################################

############################################
#####  Main Shape #####
############################################

    def build_local_shape_quad(self):
        #Calculate loci quadrature
        xq, yq, wq = quad.loci_quad(self.outline._data[:,0], \
            self.outline._data[:,1], self.radial_nodes)

        return xq, yq, wq

    def build_local_shape_edge(self, npts=None):
        return self.outline._data

############################################
############################################

############################################
#####  Wrappers for Cartesian coordinate systems #####
############################################

    def cart_func(self, t):
        return self.outline.func(t)

    def cart_diff(self, t):
        return self.outline.diff(t)

    def cart_diff_2nd(self, t):
        return self.outline.diff_2nd(t)

    ############################################

    def cart_func_diffs(self, t, func=None, diff=None, diff_2nd=None, with_2nd=False):
        """Same functions as above, just calculate all at once to save time"""

        #Get closest angle
        minds = [np.argmin(np.abs(self.outline._angle - t)) for t in np.atleast_1d(all_t)]

        func_ans = self.outline.func(t, minds=minds)
        diff_ans = self.outline.diff(t, minds=minds)

        if with_2nd:
            diff_2nd_ans = self.outline.diff_2nd(t, minds=minds)
            return func_ans, diff_ans, diff_2nd_ans
        else:
            return func_ans, diff_ans

############################################
############################################
