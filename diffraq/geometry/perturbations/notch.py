"""
notch.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-09-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class of the notched edge perturbation.

"""

import numpy as np
import diffraq.quadrature as quad
from diffraq.geometry import Perturbation

class Notch(Perturbation):

    kind = 'notch'

############################################
#####  Shared Quad + Edge #####
############################################

    def get_pert_quad(self, t0, tf, m, n, bab_sign):
        """ + direction = more material, - direction = less material"""

        #Get nodes
        xq, yq, wq = getattr(self, \
            f'get_quad_{self.shape_func.kind}')(t0, tf, m, n, bab_sign)

        return xq, yq, wq

    def get_pert_edge(self, t0, tf, m, n, bab_sign):
        #Get number of points per side
        npts = 2*max(m,n)

        #Get parameters of test region
        ts = np.linspace(t0, tf, npts)[:,None]

        #Get new / shifted edge
        old_edge, new_edge = self.get_new_edge(ts, bab_sign)

        #Flip to continue CCW
        new_edge = new_edge[::-1]

        #Join with straight lines
        nline = max(9, int(self.height/self.width*npts))
        line1 = self.make_line(old_edge[-1], new_edge[0], nline)
        line2 = self.make_line(new_edge[-1], old_edge[0], nline)

        #Join together to create loci
        loci = np.concatenate((old_edge, line1, new_edge, line2))

        #Go CW if not babinet
        loci = loci[::-bab_sign]

        #Cleanup
        del old_edge, new_edge, line1, line2

        return loci

    def get_new_edge(self, ts, bab_sign):

        #Get loci of edge across test region
        old_edge = self.shape_func.cart_func(ts)

        #Use one normal or local normals
        if self.local_norm:
            #Local normals
            normal = self.shape_func.cart_diff(ts)[:,::-1]
            normal /= np.hypot(*normal.T)[:,None]
        else:
            #Get normal vector of middle of old_edge
            normal = self.shape_func.cart_diff(ts)[:,::-1][len(ts)//2]
            normal /= np.hypot(*normal)

        #Shift edge out by the normal vector (use negative to get direction correct)
        etch = -self.height * np.array([1., -1]) * self.direction * bab_sign
        new_edge = old_edge + etch * normal

        return old_edge, new_edge

############################################
############################################

############################################
#####  Petal Specific Quad #####
############################################

    def get_quad_petal(self, t0, tf, m, n, bab_sign):

        #Get radius range
        r0, p0 = self.shape_func.unpack_param(t0)[:2]
        rf, pf = self.shape_func.unpack_param(tf)[:2]

        #Get radial and theta nodes
        pw, ww = quad.lgwt(n, 0, 1)
        pr, wr = quad.lgwt(m, r0, rf)

        #Add axis
        wr = wr[:,None]
        pr = pr[:,None]

        #Turn into parameterize variable
        ts = self.shape_func.pack_param(pr, p0)

        #Get old and new edge at radial points
        old_edge, new_edge = self.get_new_edge(ts, bab_sign)

        #Get polar coordinates of edges
        oldt = np.arctan2(old_edge[:,1], old_edge[:,0])[:,None]
        newt_tmp = np.arctan2(new_edge[:,1], new_edge[:,0])
        newr = np.hypot(*new_edge.T)

        #Do we need to flip to increasing radius?
        if newr[-1] < newr[0]:
            dir_sign = -1
        else:
            dir_sign = 1

        #Resample new edge onto radial nodes (need to flip b/c of decreasing rad)
        newt = np.interp(pr[:,0][::dir_sign], \
            newr[::dir_sign], newt_tmp[::dir_sign])[::dir_sign][:,None]

        #Difference in theta
        dt = newt - oldt

        #Get cartesian nodes
        xq = (pr*np.cos(oldt + pw*dt)).ravel()
        yq = (pr*np.sin(oldt + pw*dt)).ravel()
        wq = (ww * pr * wr * dt).ravel()

        #Cleanup
        del old_edge, new_edge, oldt, newt, pw, ww, pr, wr, ts, dt

        return xq, yq, wq

############################################
############################################

############################################
#####  Polar Specific Quad #####
############################################

    def get_quad_polar(self, t0, tf, m, n, bab_sign):

        #Get edge loci
        loci = self.get_pert_edge(t0, tf, m, n, bab_sign)

        #Shift to center
        loci_shift = loci.mean(0)
        loci -= loci_shift

        #Get quadrature from loci points
        xq, yq, wq = quad.loci_quad(loci[:,0], loci[:,1], 2*max(m,n))

        #Shift back to edge
        xq += loci_shift[0]
        yq += loci_shift[1]

        #Cleanup
        del loci

        return xq, yq, wq

############################################
############################################
