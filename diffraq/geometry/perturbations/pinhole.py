"""
pinnole.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 05-06-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class of the pinhole perturbation.

"""

import numpy as np
import diffraq.quadrature as quad

class Pinhole(object):

    kind = 'pinhole'

    def __init__(self, parent, **kwargs):
        """
        Keyword arguments:
            - kind:         kind of perturbation
            - xy0:          (x,y) coordinates of center of pinhole [m],
            - radius:       radius of pinhole [m],
            - num_quad:     number of radial quadrature nodes,
        """
        #Point to parent [shape]
        self.parent = parent

        #Set Default parameters
        def_params = {'kind':'Pinhole', 'xy0':[0,0], 'radius':0, 'num_quad':None}
        for k,v in {**def_params, **kwargs}.items():
            setattr(self, k, v)

        #Make sure array
        self.xy0 = np.array(self.xy0)

        #Set default number of nodes
        if self.num_quad is None:
            self.num_quad = max(30, self.radius/self.parent.max_radius*self.parent.radial_nodes)

############################################
#####  Main Scripts #####
############################################

    def build_quadrature(self, sxq, syq, swq):
        #Get quadrature
        xq, yq, wq = self.get_quadrature()

        #Add to parent's quadrature
        sxq = np.concatenate((sxq, xq))
        syq = np.concatenate((syq, yq))
        swq = np.concatenate((swq, wq))

        #Cleanup
        del xq, yq, wq

        return sxq, syq, swq

    def get_quadrature(self):
        #Build circular polar function
        func = lambda t: self.radius * np.ones_like(t)

        #Get quadrature (less theta nodes)
        xq, yq, wq = quad.polar_quad(func, self.num_quad, self.num_quad//4)

        #Use parent's opacity sign
        wq *= self.parent.opq_sign

        #Shift center
        xq += self.xy0[0]
        yq += self.xy0[1]

        return xq, yq, wq

    def build_edge_points(self, sedge):
        #Get edge
        xy = self.get_edge_points()

        #Add to parent's edge points
        sedge = np.concatenate((sedge, xy))

        #Cleanup
        del xy

        return sedge

    def get_edge_points(self):
        #Build circular polar function
        func = lambda t: self.radius * np.ones_like(t)

        #Get quadrature (less theta nodes)
        xy = quad.polar_edge(func, self.num_quad, self.num_quad//4)

        #Shift center
        xy += self.xy0

        return xy

############################################
############################################
