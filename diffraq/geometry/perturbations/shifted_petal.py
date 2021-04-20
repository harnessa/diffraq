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
            - num_quad:     number of quadrature nodes in gap,
        """

        #Point to parent [shape]
        self.parent = parent

        #Set Default parameters
        def_params = {'kind':'Shifted_Petal', 'angles':[0,0], 'shift':0, 'num_quad':50}
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

        #Don't shift inner circle
        inds = inds & (np.hypot(xp, yp) > self.parent.min_radius)

        #Get mean angle and width
        ang_avg = self.angles.mean()
        ang_wid = np.abs(self.angles[1] - self.angles[0])

        #Shift petal along spine of petal tip
        xp[inds] += np.cos(ang_avg) * self.shift
        yp[inds] += np.sin(ang_avg) * self.shift

        #Build across the petal (quad or edge)
        if wp is not None:
            nx, ny, nw = self.get_gap_quad(ang_avg, ang_wid)

        else:
            nx, ny, nw = self.get_gap_edge(ang_avg, ang_wid)

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
        inds = diff_angs < dang/2

        #Cleanup
        del dot, det, diff_angs

        return inds

############################################
############################################

############################################
#####  Gap between shifted petal #####
############################################

    def get_gap_quad(self, ang_avg, ang_wid):

        #Get old/new edges
        old_edge, new_edge, r0, r1 = self.get_new_edge(ang_avg)

        #Get polar coordinates of edges
        oldr = np.hypot(old_edge[:,0], old_edge[:,1])
        newt = np.arctan2(new_edge[:,1], new_edge[:,0])
        newr = np.hypot(new_edge[:,0], new_edge[:,1])

        #Circle theta around
        if newt[0] < -2*np.pi/self.parent.num_petals:
            newt %= 2*np.pi

        #Cleanup
        del old_edge, new_edge

        #Get radial and theta nodes
        pw, ww = quad.lgwt(self.num_quad, -1, 1)
        pr, wr = quad.lgwt(self.num_quad,  0, 1)

        #Add axis
        wr = wr[:,None]
        pr = pr[:,None]

        #Center theta nodes to current angles
        pw = pw*ang_wid/2 + ang_avg

        #Resample new edge onto theta nodes (flip pw to be increasing)
        newr = np.interp(pw[::-1], newt, newr)

        #Difference in radius
        dr = newr - oldr

        #Radial points
        rr = oldr + pr*dr

        #Get cartesian nodes
        nx = (rr*np.cos(pw)).ravel()
        ny = (rr*np.sin(pw)).ravel()

        #Get weights (radius change is absolute) r*dr = rr *dr*wr, dtheta = ang_wid/2*ww
        nw = (rr * wr * np.abs(dr) * ww * ang_wid/2).ravel()

        #Cleanup
        del pw, ww, pr, wr, oldr, newt, newr, dr, rr

        return nx, ny, nw

    def get_new_edge(self, ang_avg):
        #Fill in gaps left by shifted petal
        r0 = self.parent.min_radius
        r1 = r0 + self.shift

        #Get old/new edges
        the = np.linspace(self.angles[0], self.angles[1], self.num_quad)
        old_edge = r0 * np.stack((np.cos(the), np.sin(the)),1)
        new_edge = old_edge + self.shift * np.array([np.cos(ang_avg), np.sin(ang_avg)])

        del the

        return old_edge, new_edge, r0, r1

    def get_gap_edge(self, ang_avg, ang_wid):

        #Get old/new edges
        old_edge, new_edge, r0, r1 = self.get_new_edge(ang_avg)

        #Get angles of edges
        old_angs = np.arctan2(*old_edge[:,::-1].T) % (2.*np.pi)
        new_angs = np.arctan2(*new_edge[:,::-1].T) % (2.*np.pi)

        #Make lines between edges
        edge = np.empty((0,2))
        for i in range(2):
            #Get points closest to angle
            cur_ang = self.angles[i] % (2.*np.pi)
            old_pt = old_edge[np.argmin(np.abs(old_angs - cur_ang))]
            new_pt = new_edge[np.argmin(np.abs(new_angs - cur_ang))]

            #Build line between old and new
            edge = np.concatenate((edge, self.make_line(old_pt, new_pt, self.num_quad)))

        #Cleanup
        del old_edge, new_edge, old_angs, new_angs

        return edge[:,0], edge[:,1], None

    def make_line(self, r1, r2, num_pts):
        xline = np.linspace(r1[0], r2[0], num_pts)
        yline = np.linspace(r1[1], r2[1], num_pts)
        return np.stack((xline,yline),1)

############################################
############################################
