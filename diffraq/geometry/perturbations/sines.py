"""
sines.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 05-12-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class of the sine wave perturbation.

"""

import numpy as np
import diffraq.quadrature as quad
from scipy.optimize import newton

class Sines(object):

    kind = 'sines'

    def __init__(self, parent, **kwargs):
        """
        Keyword arguments:
            - kind:         kind of perturbation
            - xy0:          (x,y) coordinates of start of sine wave (innermost point) [m],
            - amplitude:    amplitude of sine wave [m],
            - frequency:    frequency of sine wave [1/m],
            - num_cycles:   number of complete cycles in wave,
            - is_kluge:     kluge to match norm to lab (M12P9) sine waves b/c 16-12 petal,
            - num_quad:     number of quadrature nodes across half-cycle,
        """
        #Point to parent [shape]
        self.parent = parent

        #Set Default parameters
        def_params = {'kind':'sines', 'xy0':[0,0], 'amplitude':0, 'frequency':0,
            'num_cycles':1, 'is_kluge':False, 'num_quad':None}
        for k,v in {**def_params, **kwargs}.items():
            setattr(self, k, v)

        #Make sure array
        self.xy0 = np.array(self.xy0)

        #Set default number of nodes
        if self.num_quad is None:
            self.num_quad = int(max(100, self.num_cycles/self.frequency/ \
                self.parent.max_radius*self.parent.radial_nodes))

        #Out of edge quad is less
        self.num_quad_perp = self.num_quad

############################################
#####  Main Scripts #####
############################################

    def build_quadrature(self, sxq, syq, swq):
        #Get quadrature
        xq, yq, wq = self.get_quadrature()

        #Add to parent's quadrature
        sxq = np.concatenate((sxq, xq))
        syq = np.concatenate((syq, yq))
        swq = np.concatenate((swq, wq))

        #Cleanup
        del xq, yq, wq

        return sxq, syq, swq

    def get_quadrature(self):
        #Get location of perturbation
        t0 = self.get_start_point()

        #Get perturbation specifc quadrature
        xq, yq, wq = getattr(self, f'get_quad_{self.parent.kind[:5]}')(t0)

        return xq, yq, wq

    def build_edge_points(self, sedge):
        #Get edge
        xy = self.get_edge_points()

        #Add to parent's edge points
        sedge = np.concatenate((sedge, xy))

        #Cleanup
        del xy

        return sedge

    def get_edge_points(self):
        #Get location of perturbation
        t0 = self.get_start_point()

        if self.parent.kind == 'polar':
            #Get change in parameter for one-half cycle (need to convert from spatial to angle for polar)
            dt = 0.5/(self.frequency*np.hypot(*self.xy0))
            p0 = None

        else:
            #Get radius start
            t0, p0 = self.parent.unpack_param(t0)[:2]

            #Get change in parameter for one=half cycle
            dt = 0.5/self.frequency

        #Get new edge at theta nodes
        old_edge, new_edge, pw, ww = self.get_new_edge(t0, p0, dt)

        del old_edge, pw, ww

        return new_edge

############################################
############################################

############################################
#####  Shared Functions #####
############################################

    def get_start_point(self):
        #Clock starting point to match pre-clocked parent shape
        if self.parent.rot_mat is not None:
            xy0 = self.xy0.dot(self.parent.rot_mat.T)
        else:
            xy0 = self.xy0.copy()

        #Get parameter of edge point closest to starting point
        t0 = self.parent.find_closest_point(xy0)

        return t0

    def get_new_edge(self, t0, p0, dt):

        #Nodes along half-wave
        pa, wa = quad.lgwt(self.num_quad, 0, 1)

        #Add second half of wave
        pa = np.concatenate((pa[::-1], 1+pa[::-1]))
        wa = np.concatenate((wa[::-1],   wa[::-1]))

        #Build all cycles
        pa = (pa + 2*np.arange(self.num_cycles)[:,None]).ravel()
        wa = (wa *     np.ones(self.num_cycles)[:,None]).ravel()

        #Build parameters across all cycles
        ts = t0 + dt * pa

        #Normalize weights
        wa *= dt

        #Turn into parameter if petal
        if p0 is not None:
            ts_use = self.parent.pack_param(ts, np.sign(p0)*1) #Rotate back to origin
        else:
            ts_use = ts.copy()

        #Use kluge to get new edge
        if self.is_kluge:
            #Fix normal with Kluge to replicate lab notches which have issues from scaling to 12 petals
            old_edge, new_edge = self.get_kluge_edge(pa, ts, ts_use)

        else:

            #Get loci of edge across test region and normals
            old_edge, normal = self.parent.cart_func_diffs(ts_use)

            #Local normals
            normal = normal[:,::-1] * np.array([1,-1])
            normal /= np.hypot(normal[:,0], normal[:,1])[:,None]

            #Build sine wave
            sine = self.amplitude * np.sin(np.pi*pa)[:,None]

            #Build new edge
            new_edge = old_edge + normal*sine

            #Cleanup
            del normal, sine

        #If petal, rotate back to original petal position
        if p0 is not None:
            rot_ang = 2*np.pi/self.parent.num_petals*(abs(p0) - 1)
            rot_mat = self.parent.parent.build_rot_matrix(rot_ang)
            old_edge = old_edge.dot(rot_mat)
            new_edge = new_edge.dot(rot_mat)

        return old_edge, new_edge, ts, wa

############################################
############################################

############################################
#####  Petal Specific Quad #####
############################################

    def get_quad_petal(self, t0):

        #Get radius start
        r0, p0 = self.parent.unpack_param(t0)[:2]

        #Get change in parameter for one half-cycle
        dr = 0.5/self.frequency

        #Get new edge at theta nodes
        old_edge, new_edge, pr, wr = self.get_new_edge(r0, p0, dr)

        #Add axis
        wr = wr[:,None]
        pr = pr[:,None]

        #Get theta nodes
        pw, ww = quad.lgwt(self.num_quad_perp, 0, 1)

        #Get polar coordinates of edges
        oldt = np.arctan2(old_edge[:,1], old_edge[:,0])[:,None]
        newt = np.arctan2(new_edge[:,1], new_edge[:,0])
        newr = np.hypot(new_edge[:,0], new_edge[:,1])

        #Resample new edge onto radial nodes
        newt = np.interp(pr[:,0], newr, newt)[:,None]

        #Difference in theta
        dt = newt - oldt

        #Theta points
        tt = oldt + pw*dt

        #Get cartesian nodes
        xq = (pr*np.cos(tt)).ravel()
        yq = (pr*np.sin(tt)).ravel()

        #Get weights rdr = wr*pr, dtheta = ww*dt
        wq = -self.parent.opq_sign * (pr * wr * ww * dt).ravel()

        #Cleanup
        del pw, ww, pr, wr, old_edge, new_edge, oldt, newt, newr, dt, tt

        return xq, yq, wq

############################################
############################################

############################################
#####  Polar Specific Quad #####
############################################

    def get_quad_polar(self, t0, do_test=False):

        #Get change in parameter for one half-cycle (need to convert from spatial to angle for polar)
        dt = 0.5/(self.frequency*np.hypot(*self.xy0))

        #Get new edge at theta nodes
        old_edge, new_edge, pw, ww = self.get_new_edge(t0, None, dt)

        #Get radial and theta nodes
        pr, wr = quad.lgwt(self.num_quad_perp, 0, 1)

        #Add axis
        wr = wr[:,None]
        pr = pr[:,None]

        #Get polar coordinates of edges
        oldr = np.hypot(old_edge[:,0], old_edge[:,1])
        newt = np.arctan2(new_edge[:,1], new_edge[:,0])
        newr = np.hypot(new_edge[:,0], new_edge[:,1])

        #Resample new edge onto theta nodes
        newr = np.interp(pw, newt, newr)

        #Difference in radius
        dr = newr - oldr

        #Cleanup
        del newt, newr

        #Radial points
        rr = oldr + pr*dr

        #Get cartesian nodes
        xq = (rr*np.cos(pw)).ravel()
        yq = (rr*np.sin(pw)).ravel()

        #Get weights r*dr = rr *dr*wr, dtheta = ww
        wq = (rr * wr * dr * ww).ravel() * self.parent.opq_sign

        if do_test:
            assert(np.isclose(abs(ww).sum(), 2*dt*self.num_cycles))

        #Cleanup
        del old_edge, new_edge, oldr, rr, dr, pw, ww, pr, wr

        return xq, yq, wq

############################################
############################################

############################################
#####  Lab Kluge #####
############################################

    def get_kluge_edge(self, pa, ts, ts_use):
        #Scale factor 16 -> 12 petals
        scale = 12/16

        #Build old edge
        old_edge = self.parent.cart_func(ts_use[:,None])

        #####
        #Get cycle phase location at loci r positions for uniformly spaced x values in 16 petal space
        #uniformly spaced x's in 16 petal space
        cos_theta = lambda r: np.cos(scale * self.parent.outline.func(r) * \
            np.pi/self.parent.num_petals)
        x0 = ts.min()*cos_theta(ts.min())
        x1 = ts.max()*cos_theta(ts.max())
        xs = np.linspace(x0, x1, len(ts))
        #turn xs into rs
        min_diff = lambda r: r - xs/cos_theta(r)
        ur = newton(min_diff, ts, full_output=False)
        #uniform cycle nodes
        uxx = 2*self.num_cycles*np.linspace(0, 1, len(ts))
        #interpolated cycle nodes at location of loci r values
        pa = np.interp(ts, ur, uxx)
        #####

        #Build new old edge w/ 16 petals
        a0 = np.arctan2(old_edge[:,1], old_edge[:,0]) * scale
        pet0 = np.hypot(old_edge[:,0], old_edge[:,1])[:,None] * \
            np.stack((np.cos(a0), np.sin(a0)),1)

        #Build normal from this edge
        normal = (np.roll(pet0, 1, axis=0) - pet0)[:,::-1] * np.array([-1,1])
        normal /= np.hypot(normal[:,0], normal[:,1])[:,None]
        #Fix normals on end pieces
        normal[0] = normal[1]
        normal[-1] = normal[-2]

        #Build sine wave
        sine = self.amplitude * np.sin(np.pi*pa)[:,None]

        #Build new edge
        edge0 = pet0 + normal * sine

        #Convert to polar cooords
        anch = np.arctan2(edge0[:,1], edge0[:,0]) / scale

        #Scale and go back to cart coords
        nch = np.hypot(edge0[:,0], edge0[:,1])[:,None] * \
            np.stack((np.cos(anch), np.sin(anch)), 1)

        #New edge
        new_edge = scale*nch + old_edge*(1-scale)

        #Cleanup
        del a0, pet0, normal, edge0, anch, nch, sine

        return old_edge, new_edge

############################################
############################################
