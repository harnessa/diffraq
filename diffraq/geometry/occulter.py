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
            self.xq, self.yq, self.wq, _, _ = \
                self.add_occulter_attitude(self.xq, self.yq, self.wq)

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

    def add_occulter_attitude(self, xx, yy=None, ww=None, nn=None, tt=None, tilt=None, spin=None):
        #If only spin, return simple rotation
        if self.has_spin and not self.has_tilt:
            return self.spin_occulter(xx, yy=yy, ww=ww, nn=nn, spin=spin)
        else:
            #Do full attitude rotation
            return self.tilt_occulter(xx, yy=yy, ww=ww, nn=nn, tt=tt, spin=spin, tilt=tilt)

    def spin_occulter(self, xx, yy=None, ww=None, nn=None, spin=None):
        if spin is None:
            spin = self.sim.spin_angle

        #Rotation matrix
        rot_mat = self.build_rot_matrix(np.radians(spin))

        #Rotate
        if yy is not None:

            #Add spin to normal angle
            if nn is not None:
                nn += np.radians(spin)

            #Separate xy (i.e., quad)
            return *rot_mat.dot(np.stack((xx, yy),0)), ww, nn, None

        else:
            #Edge
            return xx.dot(rot_mat.T)

    ############################################

    def tilt_occulter(self, xx, yy=None, ww=None, nn=None, tt=None, tilt=None, spin=None):
        if tilt is None:
            tilt = self.sim.tilt_angle
        if spin is None:
            spin = self.sim.spin_angle

        #Rotation matrix
        rot_mat = self.build_full_rot_matrix(*np.radians(tilt), np.radians(spin))

        #Rotate
        if yy is not None:

            #Source in tilted frame
            if nn is not None:
                ss0 = -np.stack((xx, yy, np.ones_like(xx)*self.sim.z0),0)
                ss0 = rot_mat.dot(ss0)

            #Separate xy and add 3rd dimension
            xx, yy, zz = rot_mat.dot(np.stack((xx, yy, np.zeros_like(xx)),0))

            #Get new weights from determinant of Jacobian (which is rotaton matrix)
            ww *= np.linalg.det(rot_mat[:2][:,:2])

            #Rotate normal angles
            if nn is not None:

                #Cosine/sine
                cosn = np.cos(nn)
                sinn = np.sin(nn)

                #Rotate source into edge frame
                sx = ss0[0]*cosn - ss0[1]*sinn
                sy = ss0[0]*sinn + ss0[1]*cosn

                #Source in spherical coordinates
                tantheta = -np.hypot(sx, sy) / ss0[2]       #tan(pi-theta)
                phi = np.arctan2(sy,sx)

                #Cleanup
                del ss0, sx, sy

                #Offset angle
                tt = np.stack((np.cos(phi), np.sin(phi)),1) #Omega (in-plane), psi (out-plane)
                tt = np.arctan(tt * tantheta[:,None])

                #Cleanup
                del tantheta, phi

                #Build new normal angle
                newm = rot_mat.dot(np.stack((cosn, sinn, np.zeros_like(nn)),0))

                #Rotated normal angles
                nn = np.arctan2(newm[1], newm[0])


                # import matplotlib.pyplot as plt;plt.ion()
                # theta = np.pi - np.arctan(tantheta)
                # omega = tt[:,0]
                # psi = tt[:,1]
                # plt.figure(); plt.colorbar(plt.scatter(xx, yy, c=np.degrees(theta), s=1.5, cmap=plt.cm.jet))
                # plt.title('theta')
                # plt.axis('equal')
                # plt.figure(); plt.colorbar(plt.scatter(xx, yy, c=np.degrees(phi), s=1.5, cmap=plt.cm.jet))
                # plt.title('phi')
                # plt.axis('equal')
                # xx0, yy0, zz = rot_mat.dot(np.stack((xx, yy, np.zeros_like(xx)),0))
                # plt.figure(); plt.colorbar(plt.scatter(xx, yy, c=zz, s=1.5, cmap=plt.cm.jet))
                # plt.title('zz')
                #
                #
                # fig, axes = plt.subplots(2,2, figsize=(9,9))
                #
                # axes[0,0].plot(np.degrees(theta), 'x')
                # axes[0,0].set_title('theta')
                # axes[0,1].plot(np.degrees(phi), 'x')
                # axes[0,1].set_title('phi')
                # axes[1,0].plot(np.degrees(omega), 'x')
                # axes[1,0].set_title('omega')
                # axes[1,1].plot(np.degrees(psi), 'x')
                # axes[1,1].set_title('psi')
                #
                # breakpoint()

                del newm, cosn, sinn

            del zz

            return xx, yy, ww, nn, tt

        else:
            #Add 3rd dimension to edge and rotate
            return np.hstack((xx, np.zeros_like(xx[:,:1]))).dot(rot_mat.T)[:,:2]

    ############################################

    def build_rot_matrix(self, angle):
        """Clockwise rotation"""
        return np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle),  np.cos(angle)]])

    def build_full_rot_matrix(self, xang, yang, zang):
        """Extrinsic (stationary coordinate frame) clockwise rotation of z-y-x"""
        xr = self.xRot(xang)
        yr = self.yRot(yang)
        zr = self.zRot(zang)
        full_rot_mat = np.linalg.multi_dot((xr, yr, zr))

        return full_rot_mat

    def xRot(self, ang):
        """Clockwise rotation of X axis"""
        return np.array([[1,0,0], [0, np.cos(ang), -np.sin(ang)], \
            [0, np.sin(ang), np.cos(ang)]])

    def yRot(self, ang):
        """Clockwise rotation of Y axis"""
        return np.array([[np.cos(ang), 0.,  np.sin(ang)], [0,1,0], \
            [-np.sin(ang), 0, np.cos(ang)]])

    def zRot(self, ang):
        """Clockwise rotation of Z axis"""
        return np.array([[np.cos(ang), -np.sin(ang), 0.], \
            [np.sin(ang), np.cos(ang), 0], [0,0,1]])

############################################
############################################

############################################
#####   Cleanup #####
############################################

    def clean_up(self):
        #Delete trash
        trash_list = ['xq', 'yq', 'wq', 'edge']

        for att in trash_list:
            if hasattr(self, att):
                delattr(self, att)

############################################
############################################
