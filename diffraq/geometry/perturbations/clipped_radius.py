"""
clipped_radius.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 03-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class of the clipped radius perturbation.

"""

import numpy as np
import diffraq.quadrature as quad

class Clipped_Radius(object):

    kind = 'Clipped_Radius'

    def __init__(self, parent, **kwargs):
        """
        Keyword arguments:
            - kind:         kind of perturbation
            - angles:       [start, end] coordinate angles that define petal [radians] (0 = 3:00, 90 = 12:00, 180 = 9:00, 270 = 6:00),
            - min_clip:     amount to clipper minimum radius [m],
            - max_clip:     amount to clipper maximum radius [m],
                            > 0 = radially out, < 0 = radially in,
            - min_radius':  minimum radius at which to clip [m],
            - max_radius':  maximum radius at which to clip [m],
        """

        #Point to parent [shape]
        self.parent = parent

        #Set Default parameters
        def_params = {'kind':'Clipped_Radius', 'angles':[0,0], \
            'min_clip':0, 'max_clip':0, 'min_radius':None, 'max_radius':None}
        for k,v in {**def_params, **kwargs}.items():
            setattr(self, k, v)

        #Shift angles if parent is clocked
        if self.parent.is_clocked:
            self.angles -= self.parent.clock_angle

        #Set min/max radii if clipped amount is supplied
        if self.min_radius is None:
            self.min_radius = self.parent.min_radius + self.min_clip
        if self.max_radius is None:
            self.max_radius = self.parent.max_radius + self.max_clip

############################################
#####  Main Scripts #####
############################################

    def build_quadrature(self, sxq, syq, swq):
        #Change parent's quad points to clip petal
        sxq, syq, swq = self.clip_petal_points(sxq, syq, swq)

        return sxq, syq, swq

    def build_edge_points(self, sedge):
        #Change parent's edge points to clip petal
        newx, newy, dummy = self.clip_petal_points(sedge[:,0], sedge[:,1], None)
        sedge = np.stack((newx, newy),1)

        #Cleanup
        del newx, newy

        return sedge

############################################
############################################

############################################
#####  Clipping Functions #####
############################################

    def clip_petal_points(self, xp, yp, wp):
        #Find points between specified angles (will be thrown out)
        ang_inds = self.find_between_angles(xp, yp)

        #Build radii
        rads = np.hypot(xp, yp)

        #Find points outside clipped bounds
        rad_inds = (rads < self.min_radius) | (rads > self.max_radius)

        import matplotlib.pyplot as plt;plt.ion()
        ox = xp.copy()
        oy = yp.copy()

        #Keep only good points
        keep_inds = ~(ang_inds & rad_inds)


        print(wp[~keep_inds].sum())

        xp = xp[keep_inds]
        yp = yp[keep_inds]
        if wp is not None:
            wp = wp[keep_inds]


        # plt.cla()
        # plt.plot(ox, oy, 'x')
        # plt.plot(ox[~keep_inds], oy[~keep_inds], '+')
        # # plt.plot(xp, yp, '+')
        # print(ox.size-xp.size)
        # breakpoint()

        #Cleanup
        del ang_inds, rad_inds, rads

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

        #Indices are where angles are < dang/2 away
        inds = diff_angs < dang/2

        #Cleanup
        del dot, det, diff_angs

        return inds

############################################
############################################
