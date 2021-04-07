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
import imp
import diffraq.geometry as geometry
from diffraq.quadrature import lgwt

class Occulter(object):

    def __init__(self, sim, shapes):
        self.sim = sim

        #Load shapes
        self.load_shapes(shapes)

############################################
#####  Shapes #####
############################################

    def load_shapes(self, shapes):

        #If pointed to, get shapes from occulter file (takes presedence over given shapes list)
        if self.sim.occulter_config is not None:
            mod = imp.load_source('mask', self.sim.occulter_config)
            #Load shape
            shapes = mod.shapes
            #Overwrite finite parameter
            if hasattr(mod, 'is_finite'):
                self.sim.is_finite = mod.is_finite

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
        self.shapes = []
        for shp in shapes:
            #Get shape kind
            kind = shp['kind'].capitalize()

            #Build shape
            shp_inst = getattr(geometry, f'{kind}Shape')(self, **shp)

            #Save shape
            self.shapes.append(shp_inst)

############################################
############################################

############################################
#####  Quadrature #####
############################################

    def build_quadrature(self):

        #Initialize
        self.xq, self.yq, self.wq = np.empty(0), np.empty(0), np.empty(0)

        #Loop through shape list and build quadratures
        for shape in self.shapes:

            #Build quadrature
            xs, ys, ws = shape.build_shape_quadrature()

            #If multiple shapes, check if we need to flip weights
            if self.is_multi:

                #Decide if we need to flip weights (finite_flag XNOR opaque)
                #We flip weights to subtract opaque region overlapping transparent region
                if not (self.finite_flag ^ int(shape.is_opaque)):
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

        #Add occulter motion
        if not np.isclose(self.sim.spin_angle, 0):
            self.xq, self.yq = self.spin_occulter(self.xq, self.yq)

############################################
############################################

############################################
#####  Edge Points #####
############################################

    def build_edge(self):

        #Initialize
        self.edge = np.empty((0,2))

        #Loop through shape list and build edges
        for shape in self.shapes:

            #Build edge
            ee = shape.build_shape_edge()

            #Append
            self.edge = np.concatenate((self.edge, ee))

        #Cleanup
        del ee

        #Add occulter motion
        if not np.isclose(self.sim.spin_angle, 0):
            self.edge = self.spin_occulter(self.edge)

############################################
############################################

############################################
#####  Occulter Motion #####
############################################

    def spin_occulter(self, xx, yy=None):
        #Rotation matrix
        rot_mat = self.build_rot_matrix(np.radians(self.sim.spin_angle))

        #Rotate
        if yy is not None:
            #Separate xy (i.e., quad)
            new = np.stack((xx, yy),1).dot(rot_mat)
            return new[:,0], new[:,1]

        else:
            #Edge
            return xx.dot(rot_mat)

    def build_rot_matrix(self, angle):
        return np.array([[ np.cos(angle), np.sin(angle)],
                         [-np.sin(angle), np.cos(angle)]])

############################################
############################################

############################################
#####   Cleanup #####
############################################

    def clean_up(self):
        #Delete trash
        trash_list = ['xq', 'yq', 'wq', 'edge', 'shapes']

        for att in trash_list:
            if hasattr(self, att):
                delattr(self, att)

############################################
############################################
