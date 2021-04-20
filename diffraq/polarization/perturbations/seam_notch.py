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
        #Get quadrature
        xq, yq, wq, dq, nq = self.get_quadrature()

        #Get indices where angles overlap
        angs = np.arctan2(syq, sxq)

        #Zero-out overlapped parent quad
        # swq[(angs >= self.tf) & (angs <= self.t0)] = 0.

        bad_inds = np.where((angs >= self.tf) & (angs <= self.t0))[0]

        

        # sxq = np.concatenate((sxq, sxq[bad_inds]))
        # syq = np.concatenate((syq, syq[bad_inds]))
        # swq = np.concatenate((swq, swq[bad_inds] * -1))
        # snq = np.concatenate((snq, snq[bad_inds]))
        # sdq = np.concatenate((sdq, sdq[bad_inds]))




        # import matplotlib.pyplot as plt;plt.ion()
        # plt.colorbar(plt.scatter(sxq, syq, c=swq, s=1))
        # plt.colorbar(plt.scatter(xq, yq, c=wq, s=1))
        # # plt.colorbar(plt.scatter(np.concatenate((sxq, xq)),  np.concatenate((syq, yq)), c=np.concatenate((snq, nq)), s=1))
        # breakpoint()

        #Add to parent's quadrature
        # sxq = np.concatenate((sxq, xq))
        # syq = np.concatenate((syq, yq))
        # swq = np.concatenate((swq, wq))
        # snq = np.concatenate((snq, nq))
        # sdq = np.concatenate((sdq, dq))

        #Cleanup
        # del xq, yq, wq, nq, dq, angs

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

    # def get_quad_petal(self, t0, tf, m, n):
    #
    #     #Get radius range
    #     r0, p0 = self.parent.unpack_param(t0)[:2]
    #     rf, pf = self.parent.unpack_param(tf)[:2]
    #
    #     #Get radial and theta nodes
    #     pw, ww = quad.lgwt(n, 0, 1)
    #     pr, wr = quad.lgwt(m, r0, rf)
    #
    #     #Add axis
    #     wr = wr[:,None]
    #     pr = pr[:,None]
    #
    #     #Turn into parameterize variable
    #     ts = self.parent.pack_param(pr, p0)
    #
    #     #Get old edge at radial node points, and etch and normal
    #     old_edge, etch, normal = self.get_new_edge(ts)
    #
    #     #Get parameters outside bounds of old edge
    #     ts_big = np.linspace(ts.min()-ts.ptp()*.05, ts.max()+ts.ptp()*.05, \
    #         int(len(ts)*1.1))[:,None]
    #     #Sort to be same direction as ts
    #     ts_big = ts_big[::int(np.sign(ts[1]-ts[0]))]
    #     #Get new edge outside of bounds of old edge
    #     old_big, dummy, normal_big = self.get_new_edge(ts_big)
    #
    #
    #     import matplotlib.pyplot as plt;plt.ion()
    #     breakpoint()
    #
    #     #Create new edge
    #     new_edge = old_big + etch*normal_big
    #
    #     #Get polar coordinates of edges
    #     oldt = np.arctan2(old_edge[:,1], old_edge[:,0])[:,None]
    #     oldr = np.hypot(old_edge[:,0], old_edge[:,1])
    #     newt_tmp = np.arctan2(new_edge[:,1], new_edge[:,0])
    #     newr = np.hypot(new_edge[:,0], new_edge[:,1])
    #
    #     #Do we need to flip to increasing radius? (for interpolation)
    #     dir_sign = int(np.sign(newr[1] - newr[0]))
    #     pr_sign = int(np.sign(pr[:,0][1] - pr[:,0][0]))
    #
    #     #Resample new edge onto radial nodes (need to flip b/c of decreasing rad)
    #     newt = np.interp(pr[:,0][::pr_sign], \
    #         newr[::dir_sign], newt_tmp[::dir_sign])[::dir_sign][:,None]
    #
    #     #Difference in theta
    #     dt = newt - oldt
    #
    #     #Theta points
    #     tt = oldt + pw*dt
    #
    #     #Get cartesian nodes
    #     xq = (pr*np.cos(tt)).ravel()
    #     yq = (pr*np.sin(tt)).ravel()
    #
    #     #Get quadrature sign depending if same opaqueness as parent
    #     qd_sign = -(self.parent.opq_sign * self.direction)
    #
    #     #Get weights (theta change is absolute) rdr = wr*pr, dtheta = ww*dt
    #     wq = qd_sign * (ww * pr * wr * np.abs(dt)).ravel()
    #
    #     #Cleanup
    #     del pw, ww, pr, wr, old_edge, new_edge, oldt, newt, ts, dt, tt, \
    #         ts_big, old_big, normal_big,
    #
    #     return xq, yq, wq

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

        #Build normal angles (x2 for each side of edge)
        nq = (np.ones_like(pr) * np.arctan2(dt[:,0], -dt[:,1])).ravel()

        #Build edge distances
        dq = seam_width * (np.ones(len(dt))[:,None] * pr).ravel()

        #Ravel xq, yq
        xq = xq.ravel()
        yq = yq.ravel()

        #Cleanup
        del old_edge, new_edge, shift_normal, dt, ts, pr, wr

        return xq, yq, wq, dq, nq

############################################
############################################
