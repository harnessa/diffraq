"""
test_seam.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 03-18-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test of building narrow seam around edges for non-scalar diffraction.

"""

import diffraq
import numpy as np

class Test_Seam(object):

    seam_width = 1e-1

    def run_all_tests(self):
        tsts = ['polar', 'cartesian', 'petal']
        for t in tsts:
            getattr(self, f'test_{t}')()

############################################

    def test_polar(self):
        #Real polar function
        a = 0.3
        gfunc = lambda t: 1 + a*np.cos(3*t)

        #Simulation parameters
        params = {'radial_nodes':100, 'theta_nodes':50, 'do_run_vector':True,\
            'seam_width':self.seam_width}

        #Simulated shapes
        shapes = {'kind':'polar', 'edge_func':gfunc}

        #Build simulator
        sim = diffraq.Simulator(params, shapes)

        #Build polar seam
        sim.occulter.vector.build_quadrature()

        breakpoint()

############################################

if __name__ == '__main__':

    ts = Test_Seam()
    ts.run_all_tests()
