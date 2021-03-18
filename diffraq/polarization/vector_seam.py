"""
vector_seam.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 03-18-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class representing the narrow seam around the edge of an occulter/aperture
    that holds the area quadrature allowing the calculation of
    non-scalar diffraction via the Braunbek method.

"""

import numpy as np
import diffraq.geometry as geometry

class VectorSeam(object):

    def __init__(self, parent, shapes, seam_width):
        self.parent = parent    #Occulter
        self.shapes = shapes
        self.seam_width = seam_width

        #Copy babinet flag
        self.is_babinet = self.parent.is_babinet

        #Load seams
        self.load_seams()

############################################
#####  Seams #####
############################################

    def load_seams(self):
        #Loop through shapes and load seams for each
        self.seams = []
        for shape in self.shapes:

            #Build seam for given shape
            seam = getattr(geometry, f'{shape.kind.capitalize()}Seam')(self, shape)

            #Save seam
            self.seams.append(seam)

############################################
############################################

############################################
#####  Quadrature #####
############################################

    def build_quadrature(self):

        #Initialize
        self.xq, self.yq, self.wq = np.empty(0), np.empty(0), np.empty(0)
        self.dq, self.nq = np.empty(0), np.empty(0)

        #Loop through seams list and build quadratures
        for seam in self.seams:

            #Build quadrature (and edge distance and normal angles)
            xs, ys, ws, ds, ns = seam.build_seam_quadrature()

            #If multiple shapes, check if we need to flip weights
            if self.parent.is_multi:

                #Decide if we need to flip weights (finite_flag XNOR opaque)
                #We flip weights to subtract opaque region overlapping transparent region
                if not (self.parent.finite_flag ^ int(seam.shape.is_opaque)):
                    ws *= -1

            #Append
            self.xq = np.concatenate((self.xq, xs))
            self.yq = np.concatenate((self.yq, ys))
            self.wq = np.concatenate((self.wq, ws))
            self.dq = np.concatenate((self.dq, ds))
            self.nq = np.concatenate((self.nq, ns))

        #Cleanup
        del xs, ys, ws, ds, ns

        #Add occulter motion
        if not np.isclose(self.parent.sim.spin_angle, 0):
            #Rotate quadrature points
            self.xq, self.yq = self.parent.spin_occulter(self.xq, self.yq)

            #Rotate all normal angles
            self.nq += self.parent.sim.spin_angle

############################################
############################################

############################################
#####  Edge Points #####
############################################

    def build_edge(self):

        #Initialize
        self.edge = np.empty((0,2))

        #Loop through seam list and build edges
        for seam in self.seams:

            #Build edge
            ee = seam.build_seam_edge()

            #Append
            self.edge = np.concatenate((self.edge, ee))

        #Cleanup
        del ee

        #Add occulter motion
        if not np.isclose(self.parent.sim.spin_angle, 0):
            self.edge = self.parent.spin_occulter(self.edge)

############################################
############################################
