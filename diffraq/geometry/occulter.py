"""
occulter.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-15-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Master class representing the occulter/aperture that holds all
    shapes contributing to the diffraction screen.

"""

import numpy as np
import diffraq.quadrature as quad
import diffraq.geometry as geometry

class Occulter(object):

    def __init__(self, sim, shapes):
        self.sim = sim
        self.load_shapes(shapes)

############################################
#####  Shapes #####
############################################

    def load_shapes(self, shapes=[]):
        #Turn into list
        if not isinstance(shapes, list):
            shapes = [shapes]

        #Finite flag
        self.finite_flag = int(self.sim.is_finite)

        #Multi shape flag
        self.is_multi = len(shapes) > 1

        #Use Babinet? (could be replaced later by single occulter)
        self.is_babinet = not self.sim.is_finite

        #Loop through and build shapes
        self.shape_list = []
        for shp in shapes:
            #Get shape kind
            kind = shp['kind'].capitalize()

            #Build shape
            shp_inst = getattr(geometry, f'{kind}Shape')(self, **shp)

            #Save shape
            self.shape_list.append(shp_inst)

############################################
############################################

############################################
#####  Quadrature #####
############################################

    def build_quadrature(self):

        #Initialize
        self.xq, self.yq, self.wq = np.empty(0), np.empty(0), np.empty(0)

        #Loop through shape list and build quadratures
        for shape in self.shape_list:
            #Build quadrature
            xs, ys, ws = shape.build_shape_quadrature()

            #If multiple shapes, check if we need to flip weights
            if self.is_multi:

                #Get flag that tells if quadrature covers aperture
                does_cover = np.any(np.hypot(xs, ys) <= self.sim.tel_diameter/2)

                #Decide if we need to flip weights (does_cover AND (finite_flag XNOR opaque))
                #We flip weights to subtract opaque region overlapping transparent region
                if does_cover and not (self.finite_flag ^ int(shape.is_opaque)):
                    ws *= -1

            else:

                #Set babinet flag to single occulter opaque flag
                self.is_babinet = shape.is_opaque

            #Append
            self.xq = np.concatenate((self.xq, xs))
            self.yq = np.concatenate((self.yq, ys))
            self.wq = np.concatenate((self.wq, ws))

        #Cleanup
        del xs, ys, ws

############################################
############################################

############################################
#####  Edge Points #####
############################################

    def build_edge(self):

        #Initialize
        self.edge = np.empty((0,2))

        #Loop through shape list and add quadratures
        for shape in self.shape_list:
            #Build edge
            ee = shape.build_shape_edge()

            #Sort by angle
            #TODO: fix sorting for multiple, unconnected apertures
            ee = ee[np.argsort(np.arctan2(ee[:,1] - ee[:,1].mean(), ee[:,0] - ee[:,0].mean()))]

            #Append
            self.edge = np.concatenate((self.edge, ee))

        #Cleanup
        del ee

############################################
############################################
