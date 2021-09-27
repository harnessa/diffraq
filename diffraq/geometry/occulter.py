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

        #Save shape params
        self.shape_params = shapes

        #Load shapes
        self.load_shapes(shapes)

        #Check if occulter has non-zero attitude
        self.has_spin = not np.isclose(self.sim.spin_angle, 0)
        self.has_tilt = not np.isclose(np.hypot(*self.sim.tilt_angle), 0)
        self.has_attitude = self.has_spin or self.has_tilt

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
            if hasattr(mod, 'occulter_is_finite'):
                self.sim.occulter_is_finite = mod.occulter_is_finite

        #Turn into list
        if not isinstance(shapes, list):
            shapes = [shapes]

        #Finite flag
        self.finite_flag = int(self.sim.occulter_is_finite)

        #Multi shape flag
        self.is_multi = len(shapes) > 1

        #Use Babinet? (could be replaced later by single occulter)
        self.is_babinet = not self.sim.occulter_is_finite

        #Loop through and build shapes
        self.shapes = []
        for shp in shapes:
            #Get shape kind (capitalize first letter only)
            kind = shp['kind'][0].capitalize() + shp['kind'][1:]

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

        #Add occulter attitude
        if self.has_attitude:
            self.xq, self.yq = self.add_occulter_attitude(self.xq, self.yq)

        #Shift occulter
        if self.sim.occulter_shift is not None:
            self.xq += self.sim.occulter_shift[0]
            self.yq += self.sim.occulter_shift[1]

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

        #Add occulter attitude
        if self.has_attitude:
            self.edge = self.add_occulter_attitude(self.edge)

        #Shift occulter
        if self.sim.occulter_shift is not None:
            self.edge += np.array(self.sim.occulter_shift)

############################################
############################################

############################################
#####  Occulter Motion #####
############################################

    def add_occulter_attitude(self, xx, yy=None):
        #If only spin, return simple rotation
        if self.has_spin and not self.has_tilt:
            return self.spin_occulter(xx, yy=yy)
        else:
            #Do full attitude rotation
            return self.tilt_occulter(xx, yy=yy)

    def spin_occulter(self, xx, yy=None):
        #Rotation matrix
        rot_mat = self.build_rot_matrix(np.radians(self.sim.spin_angle))

        #Rotate
        if yy is not None:
            #Separate xy (i.e., quad)
            new = np.stack((xx, yy),1).dot(rot_mat)
            xx, yy = new[:,0], new[:,1]
            del new
            return xx, yy

        else:
            #Edge
            return xx.dot(rot_mat)

    def build_rot_matrix(self, angle):
        return np.array([[ np.cos(angle), np.sin(angle)],
                         [-np.sin(angle), np.cos(angle)]])

    ############################################

    def tilt_occulter(self, xx, yy=None):
        #Rotation matrix
        rot_mat = self.build_full_rot_matrix(np.radians(self.sim.spin_angle), \
            *np.radians(self.sim.tilt_angle))

        #Rotate
        if yy is not None:
            #Separate xy and add 3rd dimension
            new = np.stack((xx, yy, np.zeros_like(xx)),1).dot(rot_mat)
            xx, yy = new[:,0], new[:,1]
            del new
            return xx, yy

        else:
            #Add 3rd dimension to edge
            new = np.hstack((xx, np.zeros_like(xx[:,:1])))
            edge = new[:,:2]
            del new
            return edge

    def build_full_rot_matrix(self, yaw, pit, rol):
        #Note that this is a clockwise rotation -euler angles
        yaw_mat = np.array([[np.cos(yaw), np.sin(yaw), 0.], \
            [-np.sin(yaw), np.cos(yaw), 0], [0,0,1]])
        pit_mat = np.array([[np.cos(pit), 0.,  -np.sin(pit)], [0,1,0], \
            [np.sin(pit), 0, np.cos(pit)]])
        rol_mat = np.array([[1,0,0], [0, np.cos(rol), np.sin(rol)], \
            [0, -np.sin(rol), np.cos(rol)]])
        full_rot_mat = np.linalg.multi_dot((yaw_mat, pit_mat, rol_mat))

        return full_rot_mat

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
