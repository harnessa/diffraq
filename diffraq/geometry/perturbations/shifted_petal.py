"""
shifted_petal.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-12-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class of the shifted petal perturbation.

"""

import numpy as np
import diffraq.quadrature as quad

class Shifted_Petal(object):

    kind = 'Shifted_Petal'

    def __init__(self, parent, **kwargs):
        """
        Keyword arguments:
            - kind:         kind of perturbation
            - angles:       [start, end] coordinate angles that define petal [radians] (0 = 3:00, 90 = 12:00, 180 = 9:00, 270 = 6:00),
            - shift:        amount to shift petal [m],
                            > 0 = radially out, < 0 = radially in,
        """

        #Point to parent [shape]
        self.parent = parent

        #Set Default parameters
        def_params = {'kind':'Shifted_Petal', 'angles':[0,0], 'shift':0}
        for k,v in {**def_params, **kwargs}.items():
            setattr(self, k, v)

        #Shift angles if parent is clocked
        if self.parent.is_clocked:
            self.angles -= self.parent.clock_angle

############################################
#####  Main Scripts #####
############################################

    def build_quadrature(self, sxq, syq, swq):
        #Change parent's quad points to shift petal
        sxq, syq, swq = self.shift_petal_points(sxq, syq, swq)

        return sxq, syq, swq

    def build_edge_points(self, sedge):
        #Change parent's edge points to clip petal
        newx, newy, dummy = self.shift_petal_points(sedge[:,0], sedge[:,1], None)
        sedge = np.stack((newx, newy),1)

        #Cleanup
        del newx, newy

        return sedge

############################################
############################################

############################################
#####  Shifting Functions #####
############################################

    def shift_petal_points(self, xp, yp, wp):
        #Find points between specified angles
        inds = self.find_between_angles(xp, yp)

        #Don't include central circle
        inds = inds & (np.hypot(xp, yp) >= self.parent.min_radius)

        #Shift petal along spine of petal tip
        xp[inds] += np.cos(self.angles.mean()) * self.shift
        yp[inds] += np.sin(self.angles.mean()) * self.shift

        #Fill in gaps left by shifted petal
        r0 = self.parent.min_radius
        r1 = r0 + self.shift
        nnew = 50                   #nodes

        #Build across the petal (quad or edge)
        if wp is not None:
            nx, ny, nw = quad.starshade_quad(lambda t: 1/self.parent.num_petals, \
                1, r0, r1, nnew, nnew, has_center=False)
        else:
            nx, ny = quad.starshade_edge(lambda t: 1/self.parent.num_petals, \
                1, r0, r1, nnew).T
            nw = None

        #Rotate to current angle
        nx, ny = np.stack((nx, ny), 1).dot(\
            self.parent.build_rot_matrix(self.angles.mean())).T

        #Add to petal
        xp = np.concatenate((xp, nx))
        yp = np.concatenate((yp, ny))
        if wp is not None:
            wp = np.concatenate((wp, nw))

        #Cleanup
        del inds, nx, ny, nw

        return xp, yp, wp

    def find_between_angles(self, xx, yy):
        #Difference between angles
        dang = self.angles[1] - self.angles[0]

        #Angle bisector (mean angle)
        mang = (self.angles[0] + self.angles[1])/2

        #Get angle between points and bisector
        dot = xx*np.cos(mang) + yy*np.sin(mang)
        det = xx*np.sin(mang) - yy*np.cos(mang)
        diff_angs = np.abs(np.arctan2(det, dot))

        #Indices are where angles are <= dang/2 away
        inds = diff_angs <= dang/2

        #Cleanup
        del dot, det, diff_angs

        return inds

############################################
############################################
