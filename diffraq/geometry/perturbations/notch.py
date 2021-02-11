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

    def get_pert_quad(self, t0, tf, m, n, bab_etch):
        """ + direction = more material, - direction = less material"""

        #Get edge loci
        loci, npts = self.get_pert_edge(t0, tf, m, n, bab_etch)

        #Shift to center
        loci_shift = loci.mean(0)
        loci -= loci_shift

        #Get quadrature from loci points
        xq, yq, wq = quad.loci_quad(loci[:,0], loci[:,1], npts)

        #Shift back to edge
        xq += loci_shift[0]
        yq += loci_shift[1]

        #Cleanup
        del loci

        return xq, yq, wq

    def get_pert_edge(self, t0, tf, m, n, bab_etch):

        #Get number of points per side
        npts = 2*max(m,n)

        #Get loci of edge across test region
        ts = np.linspace(t0, tf, npts)[:,None]
        old_edge = self.shape_func.cart_func(ts)

        #Get normal vector of middle of old_edge
        normal = self.shape_func.cart_diff(ts)[:,::-1][npts//2]
        normal /= np.hypot(*normal)

        #Shift edge out by the normal vector (use negative to get direction correct)
        etch = -self.height * np.array([1., -1]) * self.direction * bab_etch
        new_edge = old_edge + etch * normal

        #Flip to continue CCW
        new_edge = new_edge[::-1]

        #Join with straight lines
        nline = max(9, int(self.height/self.width*npts))
        line1 = self.make_line(old_edge[-1], new_edge[0], nline)
        line2 = self.make_line(new_edge[-1], old_edge[0], nline)

        #Join together to create loci
        loci = np.concatenate((old_edge, line1, new_edge, line2))

        #Go CW if not babinet
        loci = loci[::-bab_etch]

        #Cleanup
        del old_edge, new_edge, line1, line2, ts

        return loci, npts
