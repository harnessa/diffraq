"""
perturbation.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-09-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Base class of a perturbation to generate quadrature in addition
    to the occulter.

"""

import numpy as np

class Perturbation(object):

    def __init__(self, shape_func, **kwargs):
        #Point to parent occulter's shape function
        self.shape_func = shape_func

        #Set perturbation-specific keywords
        for k,v in kwargs.items():
            setattr(self, k, v)

############################################
#####  Quadrature + Edge points #####
############################################

    def build_quadrature(self, radial_nodes, theta_nodes, bab_etch):
        #Get location of perturbation
        t0, tf = self.get_param_locs()

        #Determine how many node points to use
        #TODO: what is converged?
        m, n = radial_nodes//2, theta_nodes//2

        #Get perturbation specifc quadrature
        xq, yq, wq = self.get_pert_quad(t0, tf, m, n, bab_etch)

        return xq, yq, wq

    def build_edge_points(self, radial_nodes, theta_nodes, bab_etch):
        #Get location of perturbation
        t0, tf = self.get_param_locs()

        #Determine how many node points to use
        #TODO: what is converged?
        m, n = radial_nodes//2, theta_nodes//2

        #Get perturbation specifc quadrature
        xy, npts = self.get_pert_edge(t0, tf, m, n, bab_etch)

        return xy

    def get_param_locs(self):
        #Get parameter of edge point closest to starting point
        t0 = self.shape_func.find_closest_point(self.xy0)

        #Get parameter to where the cart. distance between is equal to pert. width
        tf = self.shape_func.find_width_point(t0, self.width)

        return t0, tf

    def make_line(self, r1, r2, num_pts):
        xline = np.linspace(r1[0], r2[0], num_pts)
        yline = np.linspace(r1[1], r2[1], num_pts)
        return np.stack((xline,yline),1)[1:-1]

############################################
############################################
