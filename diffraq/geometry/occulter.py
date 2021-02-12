"""
occulter.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-15-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Base class of an occulter shape and used to generate quadrature points.

"""

import numpy as np
import diffraq.quadrature as quad
import diffraq.geometry as geometry

class Occulter(object):

    def __init__(self, sim):
        self.sim = sim
        #Etching error sign is flipped for Babinet
        self.bab_sign = [1, -1][self.sim.is_babinet]
        #Set shape function
        self.set_shape_function()

############################################
#####  Quadrature #####
############################################

    def build_quadrature(self):

        #Build perturbation quadrature (do this first for memory concern)
        xp, yp, wp = self.build_perturbation_quadrature()

        #Build shape quadrature
        self.xq, self.yq, self.wq = self.build_shape_quadrature()

        #Add perturbations to quadrature
        self.xq = np.concatenate((self.xq, xp))
        self.yq = np.concatenate((self.yq, yp))
        self.wq = np.concatenate((self.wq, wp))

        #Cleanup
        del xp, yp, wp

    def build_perturbation_quadrature(self):

        #Initialize
        xp, yp, wp = np.empty(0), np.empty(0), np.empty(0)
        self.perturb_list = []

        #Loop through perturbation list and grab quadratures
        for kind, pms in self.sim.perturbations:

            #Build perturbation
            pert = getattr(geometry, kind.capitalize())(self.shape_func, **pms)

            #Get perturbations's quadrature
            x, y, w = pert.build_quadrature(self.sim.radial_nodes, \
                self.sim.theta_nodes, self.bab_sign)

            #Append
            xp = np.concatenate((xp, x))
            yp = np.concatenate((yp, y))
            wp = np.concatenate((wp, w))
            self.perturb_list.append(pert)

        return xp, yp, wp

############################################
############################################

############################################
#####  Edge Points #####
############################################

    def get_edge_points(self):

        #Build perturbation edge points
        xyp = self.build_perturbation_edge()

        #Build shape edge points
        xyq = self.build_edge()

        #Add perturbations to quadrature
        xyq = np.concatenate((xyq, xyp))

        #Cleanup
        del xyp

        #Sort by angle
        xyq = xyq[np.argsort(np.arctan2(xyq[:,1], xyq[:,0]))]

        return xyq

    def build_perturbation_edge(self):

        #Initialize
        xyp = np.empty((0,2))

        #Loop through perturbation list and grab quadratures
        for kind, pms in self.sim.perturbations:

            #Build perturbation
            pert = getattr(geometry, kind.capitalize())(self.shape_func, **pms)

            #Get perturbations's quadrature
            xy = pert.build_edge_points(self.sim.radial_nodes, \
                self.sim.theta_nodes, self.bab_sign)

            #Append
            xyp = np.concatenate((xyp, xy))

        return xyp

############################################
############################################
