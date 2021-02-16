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

class ShiftedPetal(object):

    kind = 'shiftedPetal'

    def __init__(self, parent, **kwargs):
        """
        Keyword arguments:
            - kind:         kind of perturbation
            - angles:       [start, end] coordinate angles that define petal [radians] (0 = 3:00, 90 = 12:00, 180 = 9:00, 270 = 6:00),
            - max_radius:   maximum radius of shifted petal (for use in lab starshades only),
            - shift:        amount to shift petal [m],
            - direction:    vector direction to shift petal;
                            > 0 = radially out, < 0 = radially in,
        """

        #Point to parent [shape]
        self.parent = parent

        #Set Default parameters
        def_params = {'kind':'shiftedPetal', 'angles':[0,0], 'max_radius':None, \
            'shift':0, 'direction':[0,0]}
        for k,v in {**def_params, **kwargs}.items():
            setattr(self, k, v)

        #Normalize direction
        self.direction = np.atleast_1d(self.direction)
        self.direction /= np.hypot(*self.direction)

############################################
#####  Main Scripts #####
############################################

    def build_quadrature(self, sxq, syq, swq):
        #Change parent's quad points to shift petal
        sxq, syq, swq = self.shift_petal_points(sxq, syq, swq)

        return sxq, syq, swq

    def build_edge_points(self, sedge):
        #Find parent's edge points that are part of this petal
        newx, newy, dummy = self.shift_petal_points(sedge[:,0], sedge[:,1], None)
        sedge = np.stack((newx, newy),1)

        #Cleanup
        del newx, newy

        return sedge

############################################
############################################

############################################
#####  Shared Functions #####
############################################

    def shift_petal_points(self, xp, yp, wp):
        #Find points between specified angles
        inds = self.find_between_angles(xp, yp)

        #If applicable, only shift inner radius (for lab starshade)
        if self.max_radius is not None:
            rads = np.sqrt(xp**2 + yp**2)
            inds = inds & (rads <= self.max_radius)

        # oldx = xp.copy()
        # oldy = yp.copy() #TODO: remove

        #Add to xy
        xp[inds] += self.direction[0] * self.shift
        yp[inds] += self.direction[1] * self.shift

        #Throw out region of outer petal that did not get shifted, but got overlapped by inner
        if self.max_radius is not None:
            bad_inds = self.find_between_angles(xp, yp) & \
                (rads <= self.max_radius + self.shift) & (rads >= self.max_radius)

            xp = xp[~bad_inds]
            yp = yp[~bad_inds]


            # oldx = oldx[~bad_inds]
            # oldy = oldy[~bad_inds] #TODO: remove
            #

            #Throw out weights too (if applicable)
            if wp is not None:
                wp = wp[~bad_inds]

            #Cleanup
            del rads, bad_inds

        # #FIXME: add extra quadrature to fill displaced area??
        #
        # inds = self.find_between_angles(xp, yp) & (np.sqrt(xp**2+yp**2) <= self.max_radius*1.05)
        #
        # pxo = oldx[inds]
        # pyo = oldy[inds]
        # angs = np.arctan2(pyo-pyo.mean(),pxo-pxo.mean())
        # pxo = pxo[np.argsort(angs)]
        # pyo = pyo[np.argsort(angs)]
        #
        # px = xp[inds]
        # py = yp[inds]
        # angs = np.arctan2(py-py.mean(),px-px.mean())
        # px = px[np.argsort(angs)]
        # py = py[np.argsort(angs)]
        #
        #
        # import matplotlib.pyplot as plt;plt.ion()
        # plt.plot(pxo, pyo, '-')
        # plt.plot(px, py, 'x-')
        # # plt.scatter(xp[inds],yp[inds],c=wp[inds],s=1)
        #
        # breakpoint()

        #Cleanup
        del inds

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
