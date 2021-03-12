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

class Notch(object):

    kind = 'notch'

    def __init__(self, parent, **kwargs):
        """
        Keyword arguments:
            - kind:         kind of perturbation
            - xy0:          (x,y) coordinates of start of notch (innermost point) [m],
            - height:       height of notch [m],
            - width:        width (along edge) of notch [m],
            - direction:    direction of notch. 1 = excess material, -1 = less material,
            - local_norm:   True = shift each part of original edge by its local normal,
                            False = shift entire edge by single direction (equal to normal in the middle),
            - num_quad:     number of quadrature nodes in each direction,
        """
        #Point to parent [shape]
        self.parent = parent

        #Point to parent occulter's shape function (for quick access)
        self.outline = self.parent.outline

        #Set Default parameters
        def_params = {'kind':'Notch', 'xy0':[0,0], 'height':0, 'width':0, \
            'direction':1, 'local_norm':True, 'num_quad':None}
        for k,v in {**def_params, **kwargs}.items():
            setattr(self, k, v)

        #Make sure array
        self.xy0 = np.array(self.xy0)

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
        t0, tf, m, n = self.get_param_locs()

        #Get perturbation specifc quadrature
        xq, yq, wq = getattr(self, f'get_quad_{self.outline.kind}')( \
            t0, tf, m, n, self.parent.opq_sign)

        return xq, yq, wq

    def build_edge_points(self, sedge):
        #Get edge
        xy = self.get_edge_points()

        #Sort edge points
        xy = self.parent.sort_edge_points(xy, 1)

        #Add to parent's edge points
        sedge = np.concatenate((sedge, xy))

        #Cleanup
        del xy

        return sedge

    def get_edge_points(self):
        #Get location of perturbation
        t0, tf, m, n = self.get_param_locs()

        #Get perturbation specifc edge points
        xy = self.get_pert_edge(t0, tf, m, self.parent.opq_sign)

        return xy

############################################
############################################

############################################
#####  Shared Functions #####
############################################

    def get_param_locs(self):
        #Clock starting point to match pre-clocked parent shape
        if self.parent.is_clocked:
            xy0 = self.xy0.dot(self.parent.clock_mat.T)
        else:
            xy0 = self.xy0.copy()

        #Get parameter of edge point closest to starting point
        t0 = self.outline.find_closest_point(xy0)

        #Get parameter to where the cart. distance between is equal to pert. width
        tf = self.outline.find_width_point(t0, self.width)

        #Get number of nodes
        if self.num_quad is not None:
            m, n = self.num_quad, self.num_quad

        else:

            #Get number of radial nodes to match parent
            drad = np.hypot(*self.outline.cart_func(tf)) - np.hypot(*self.outline.cart_func(t0))
            m = max(100, int(drad / (self.parent.max_radius - self.parent.min_radius) * \
                self.parent.radial_nodes))

            #Get number of theta nodes to match parent
            dthe = np.abs(np.arctan2(*self.outline.cart_func(tf)[::-1]) - \
                np.arctan2(*self.outline.cart_func(t0)[::-1]))
            n = max(50, int(dthe / (2.*np.pi) * self.parent.theta_nodes))

        return t0, tf, m, n

    def get_pert_edge(self, t0, tf, npts, opq_sign):
        #Get parameters of test region
        ts = np.linspace(t0, tf, npts)[:,None]

        #Get new / shifted edge
        old_edge, etch, normal = self.get_new_edge(ts, opq_sign)
        new_edge = old_edge + etch*normal

        #Flip to continue CCW
        new_edge = new_edge[::-1]

        #Join with straight lines
        nline = max(9, int(self.height/self.width*npts))
        line1 = self.make_line(old_edge[-1], new_edge[0], nline)
        line2 = self.make_line(new_edge[-1], old_edge[0], nline)

        #Join together to create loci
        loci = np.concatenate((old_edge, line1, new_edge, line2))

        #Cleanup
        del old_edge, new_edge, line1, line2

        return loci

    def get_new_edge(self, ts, opq_sign):

        #Get loci of edge across test region
        old_edge = self.outline.cart_func(ts)

        #Use one normal or local normals
        if self.local_norm:
            #Local normals
            normal = self.outline.cart_diff(ts)[:,::-1]
            normal /= np.hypot(*normal.T)[:,None]
        else:
            #Get normal vector of middle of old_edge
            normal = self.outline.cart_diff(ts)[:,::-1][len(ts)//2]
            normal /= np.hypot(*normal)

        #Shift edge out by the normal vector
        etch = -self.height * np.array([1., -1]) * self.direction * opq_sign

        return old_edge, etch, normal

    def make_line(self, r1, r2, num_pts):
        xline = np.linspace(r1[0], r2[0], num_pts)
        yline = np.linspace(r1[1], r2[1], num_pts)
        return np.stack((xline,yline),1)[1:-1]

############################################
############################################

############################################
#####  Petal Specific Quad #####
############################################

    def get_quad_petal(self, t0, tf, m, n, opq_sign):

        #Get radius range
        r0, p0 = self.outline.unpack_param(t0)[:2]
        rf, pf = self.outline.unpack_param(tf)[:2]

        #Get radial and theta nodes
        pw, ww = quad.lgwt(n, 0, 1)
        pr, wr = quad.lgwt(m, r0, rf)

        #Add axis
        wr = wr[:,None]
        pr = pr[:,None]

        #Turn into parameterize variable
        ts = self.outline.pack_param(pr, p0)

        #Get old edge at radial node points, and etch and normal
        old_edge, etch, normal = self.get_new_edge(ts, opq_sign)

        new_dummy = old_edge + etch*normal  #TODO: remove

        #Get parameters outside bounds of old edge
        ts_big = np.linspace(ts.min()-ts.ptp()*.05, ts.max()+ts.ptp()*.05, \
            int(len(ts)*1.1))[:,None]
        #Sort to be same direction as ts
        ts_big = ts_big[::int(np.sign(ts[1]-ts[0]))]
        #Get new edge outside of bounds of old edge
        old_big, dummy, normal_big = self.get_new_edge(ts_big, opq_sign)

        #Create new edge
        new_edge = old_big + etch*normal_big

        #Get polar coordinates of edges
        oldt = np.arctan2(old_edge[:,1], old_edge[:,0])[:,None]
        oldr = np.hypot(*old_edge.T)
        newt_tmp = np.arctan2(new_edge[:,1], new_edge[:,0])
        newr = np.hypot(*new_edge.T)

        #Do we need to flip to increasing radius? (for interpolation)
        dir_sign = int(np.sign(newr[1] - newr[0]))
        pr_sign = int(np.sign(pr[:,0][1] - pr[:,0][0]))

        #Resample new edge onto radial nodes (need to flip b/c of decreasing rad)
        newt = np.interp(pr[:,0][::pr_sign], \
            newr[::dir_sign], newt_tmp[::dir_sign])[::dir_sign][:,None]

        #Difference in theta
        dt = newt - oldt

        #Theta points
        tt = oldt + pw*dt

        #Get cartesian nodes
        xq = (pr*np.cos(tt)).ravel()
        yq = (pr*np.sin(tt)).ravel()

        #Get quadrature sign depending if same opaqueness as parent
        qd_sign = -(opq_sign * self.direction)

        #Get weights (theta change is absolute) rdr = wr*pr, dtheta = ww*dt
        wq = qd_sign * (ww * pr * wr * np.abs(dt)).ravel()


        import matplotlib.pyplot as plt;plt.ion()
        plt.plot(oldr, oldt, 'x')
        plt.plot(np.hypot(*new_dummy.T),  np.arctan2(*new_dummy[:,::-1].T), '+')
        plt.plot(pr[:,0], oldt + dt, '*')

        plt.figure()
        plt.plot(*old_edge.T, 'x')
        plt.plot(*new_dummy.T, '+')
        plt.plot(*new_edge.T, '*')
        # breakpoint()


        tru_a = self.height*self.width
        print((np.abs(wq.sum())-tru_a)/tru_a*100)

        plt.figure()
        plt.scatter(xq, yq, c=wq, s=1)

        breakpoint()
        #Cleanup
        del pw, ww, pr, wr, old_edge, new_edge, oldt, newt, ts, dt, tt, \
            ts_big, old_big, normal_big,

        return xq, yq, wq

############################################
############################################

############################################
#####  Polar Specific Quad #####
############################################

    def get_quad_polar(self, t0, tf, m, n, opq_sign):

        #Get edge loci
        loci = self.get_pert_edge(t0, tf, m, opq_sign)

        #Shift to center
        loci_shift = loci.mean(0)
        loci -= loci_shift

        #Get quadrature from loci points
        xq, yq, wq = quad.loci_quad(loci[:,0], loci[:,1], max(m,n))

        #Shift back to edge
        xq += loci_shift[0]
        yq += loci_shift[1]

        #Cleanup
        del loci

        return xq, yq, wq

############################################
############################################
