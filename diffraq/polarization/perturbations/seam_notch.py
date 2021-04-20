"""
seam_notch.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 04-19-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class of the vector seam of the notched edge perturbation.

"""

import numpy as np
from diffraq.geometry import Notch
import diffraq.polarization as polar
import diffraq.quadrature as quad

class Seam_Notch(Notch):

############################################
#####  Main Scripts #####
############################################

    def build_quadrature(self, sxq, syq, swq, sdq, snq, sgw):
        #Get quadrature + negating quadrature
        xq, yq, wq, dq, nq, nxq, nyq, nwq = self.get_quadrature()

        #Add negating quadrature to parent's quad (has same normals and distances)
        sxq = np.concatenate((sxq, nxq))
        syq = np.concatenate((syq, nyq))
        swq = np.concatenate((swq, nwq))
        sdq = np.concatenate((sdq, dq))
        snq = np.concatenate((snq, nq))

        #Add to new quad to parent's quadrature
        sxq = np.concatenate((sxq, xq))
        syq = np.concatenate((syq, yq))
        swq = np.concatenate((swq, wq))
        sdq = np.concatenate((sdq, dq))
        snq = np.concatenate((snq, nq))

        #Cleanup
        del xq, yq, wq, dq, nq, nxq, nyq, nwq

        return sxq, syq, swq, sdq, snq, sgw

    def get_quadrature(self):
        #Get location of perturbation
        t0, tf, m, n = self.get_param_locs()

        #Store angles for use later
        self.t0, self.tf = t0, tf

        #Get perturbation specifc quadrature
        return getattr(self, f'get_quad_{self.parent.kind}')( \
            t0, tf, m, n)

############################################
############################################

############################################
#####  Petal Specific Quad #####
############################################

    def get_quad_petal(self, t0, tf, m, n):

        #Get radius range
        r0, p0 = self.parent.unpack_param(t0)[:2]
        rf, pf = self.parent.unpack_param(tf)[:2]

        #Get radial and theta nodes
        pr, wr = quad.lgwt(m, r0, rf)

        #Add axis
        wr = wr[:,None]
        pr = pr[:,None]

        #Petals width nodes and weights over [0,1]
        pw, ww = quad.lgwt(n, 0, 1)

        #Combine nodes for positive and negative sides of edge
        pw = np.concatenate((pw, -pw[::-1]))
        ww = np.concatenate((ww, ww[::-1]))

        #Turn into parameterize variable
        ts = self.parent.pack_param(pr, p0)

        #Get old edge and shift normal
        old_edge, etch, shift_normal = self.get_new_edge(ts)

        #Build new edge
        new_edge = old_edge + etch*shift_normal

        #Theta value at nodes
        tval = np.arctan2(new_edge[:,1], new_edge[:,0])[:,None]

        #Turn seam_width into angle
        seam_width = self.parent.parent.sim.seam_width / pr

        #Get theta of seam
        tt = tval + pw*seam_width

        #Build nodes
        xq = (pr * np.cos(tt)).ravel()
        yq = (pr * np.sin(tt)).ravel()

        #Build Petal weights (rdr * dtheta)
        wq = (wr * pr * ww * seam_width).ravel()

        #Get normal angle at all theta nodes (flipped negative to point inward to match seam_polar_quad)
        dt = self.parent.cart_diff(ts)[:,:,None]
        dt /= np.hypot(dt[:,0], dt[:,1])[:,None]

        #Build normal angles (x2 for each side of edge)
        nq = (np.ones_like(pw) * np.arctan2(dt[:,0], -dt[:,1])).ravel()

        #Build edge distances
        dq = (pr * seam_width * pw).ravel()

        ###########################################

        #Build negating quadrature from old edge
        old_tval = np.arctan2(old_edge[:,1], old_edge[:,0])[:,None]
        old_tt = old_tval + pw*seam_width
        nxq = (pr * np.cos(old_tt)).ravel()
        nyq = (pr * np.sin(old_tt)).ravel()

        #Weights (negative to cancel out)
        nwq = -(wr * pr * ww * seam_width).ravel()

        #Cleanup
        del old_edge, new_edge, shift_normal, tval, seam_width, tt, old_tval, old_tt

        return xq, yq, wq, dq, nq, nxq, nyq, nwq

############################################
############################################

############################################
#####  Polar Specific Quad #####
############################################

    def get_quad_polar(self, t0, tf, m, n):

        #Get parameters of test region
        ts = np.linspace(t0, tf, n)[:,None]

        #Local copy for reference
        seam_width = self.parent.parent.sim.seam_width

        #Get old edge and shift normal
        old_edge, etch, shift_normal = self.get_new_edge(ts)

        #Build new edge
        new_edge = old_edge + etch*shift_normal

        #Seam quadrature nodes, weights
        pr, wr = quad.lgwt(m, 0, 1)

        #Combine nodes for positive and negative sides of edge
        pr = np.concatenate((pr, -pr[::-1]))
        wr = np.concatenate((wr, wr[::-1]))

        #Get normal angle at all theta nodes (flipped negative to point inward to match seam_polar_quad)
        dt = self.parent.cart_diff(ts)[:,:,None]
        dt /= np.hypot(dt[:,0], dt[:,1])[:,None]

        #dtheta
        wt = abs(tf-t0)/n

        #Nodes (Eq. 11 Barnett 2021) (nx and ny are switched and ny has negative sign)
        xq = new_edge[:,0][:,None] + dt[:,1]*pr*seam_width
        yq = new_edge[:,1][:,None] - dt[:,0]*pr*seam_width

        #Weights (Eq. 12 Barnett 2021)
        wq = wt * seam_width * (wr * (xq * dt[:,1] - yq * dt[:,0])).ravel()

        #Ravel xq, yq
        xq = xq.ravel()
        yq = yq.ravel()

        #Build normal angles (x2 for each side of edge)
        nq = (np.ones_like(pr) * np.arctan2(dt[:,0], -dt[:,1])).ravel()

        #Build edge distances
        dq = seam_width * (np.ones(len(dt))[:,None] * pr).ravel()

        ###########################################

        #Build negating quadrature from old edge
        nxq = old_edge[:,0][:,None] + dt[:,1]*pr*seam_width
        nyq = old_edge[:,1][:,None] - dt[:,0]*pr*seam_width

        #Weights (negative to cancel out)
        nwq = -wt * seam_width * (wr * (nxq * dt[:,1] - nyq * dt[:,0])).ravel()

        #Ravel xq, yq
        nxq = nxq.ravel()
        nyq = nyq.ravel()

        #Cleanup
        del old_edge, new_edge, shift_normal, dt, ts, pr, wr

        return xq, yq, wq, dq, nq, nxq, nyq, nwq

############################################
############################################
