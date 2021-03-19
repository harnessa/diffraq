"""
vector_sim.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 03-18-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Main class that handles all dealing of vector diffraction
    calculations via the Braunbek method.

"""

import numpy as np
import diffraq.polarization as polar

class VectorSim(object):

    def __init__(self, sim, shapes):
        self.sim = sim
        self.shapes = shapes

        #Build params for screen
        scrn_pms = {'is_sommerfeld':self.sim.is_sommerfeld, \
            'maxwell_file':self.sim.maxwell_file, 'maxwell_func':self.sim.maxwell_func}

        #Load seams
        self.load_seams()

        #Load screen
        self.screen = polar.ThickScreen(**scrn_pms)

        #Get incident polarization state
        self.get_polarization_state()

############################################
#####  Seams #####
############################################

    def load_seams(self):
        #Loop through shapes and load seams for each
        self.seams = []
        for shape in self.shapes:

            #Build seam for given shape
            seam = polar.Seam(shape)

            #Save seam
            self.seams.append(seam)

############################################
############################################

############################################
#####  Polarization #####
############################################

    def get_polarization_state(self):
        #Get polarization state from sim
        pol_state = self.sim.polarization_state

        #Set horizontal and vertical (in lab frame) components of incident polarization
        if pol_state == 'linear':
            #Linear
            self.Ex_comp = np.cos(np.radians(self.sim.polarization_angle))
            self.Ey_comp = np.sin(np.radians(self.sim.polarization_angle))

        elif 'circular' in pol_state:
            #Circular
            XY_phase = np.pi/2.
            if pol_state.startswith('left'):
                XY_phase *= -1
            self.Ex_comp = 1./np.sqrt(2.)
            self.Ey_comp = 1./np.sqrt(2.)*np.exp(1j*XY_phase)

        elif pol_state == 'stokes':
            #Elliptical
            stokes = np.array(self.sim.stokes_parameters).copy()
            stokes /= stokes[0]
            XY_phase = np.arctan2(stokes[3], stokes[2])
            self.Ex_comp = np.sqrt(0.5*(stokes[0] + stokes[1]))
            self.Ey_comp = np.sqrt(0.5*(stokes[0] - stokes[1]))*np.exp(1j*XY_phase)

############################################
############################################

############################################
#####  Quadrature #####
############################################

    def build_quadrature(self):
        #Build area quadrature for seams
        self.build_area_quadrature()

        #Get additive vector field in Braunbek seam
        self.vec_UU = self.screen.get_vector_field( \
            self.dq, self.nq, self.sim.waves, self.Ex_comp, self.Ey_comp)

        #Cleanup
        del self.dq, self.nq

    def build_area_quadrature(self):

        #Initialize
        self.xq, self.yq, self.wq = np.empty(0), np.empty(0), np.empty(0)
        self.dq, self.nq = np.empty(0), np.empty(0)

        #Loop through seams list and build quadratures
        for seam in self.seams:

            #Build quadrature (and edge distance and normal angles)
            xs, ys, ws, ds, ns = seam.build_seam_quadrature(self.sim.seam_width)

            #If multiple shapes, check if we need to flip weights
            if self.sim.occulter.is_multi:

                #Decide if we need to flip weights (finite_flag XNOR opaque)
                #We flip weights to subtract opaque region overlapping transparent region
                if not (self.sim.occulter.finite_flag ^ int(seam.shape.is_opaque)):
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
        if not np.isclose(self.sim.spin_angle, 0):
            #Rotate quadrature points
            self.xq, self.yq = self.sim.occulter.spin_occulter(self.xq, self.yq)

            #Rotate all normal angles
            self.nq += self.sim.spin_angle

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
        if not np.isclose(self.sim.spin_angle, 0):
            self.edge = self.sim.occulter.spin_occulter(self.edge)

############################################
############################################

############################################
#####  Building Total Field #####
############################################

    def build_total_field(self, scl_pupil, vec_pupil, vec_comps):
        #Add scalar field to vector field to create total field in horiz./vert. direction
        pupil = vec_pupil.copy()
        for i in range(len(vec_comps)):
            pupil[:,i] += scl_pupil * vec_comps[i]

        return pupil

############################################
############################################

############################################
#####   Cleanup #####
############################################

    def clean_up(self):
        #Delete trash
        trash_list = ['xq' 'yq', 'wq', 'edge', 'vec_UU']
        for att in trash_list:
            if hasattr(self, att):
                delattr(self, att)

############################################
############################################
