"""
test_occulter.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-27-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test of Occulter functions.

"""

import diffraq
import numpy as np

class Test_Occulter(object):

    def run_all_tests(self):
        tsts = ['polar', 'circle', 'analytic_starshade', 'numeric_starshade']
        for t in tsts:
            getattr(self, f'test_{t}')()

############################################

    def test_polar(self):
        #Build occulter with dummy class
        sim = Dummy_Sim()
        occulter = diffraq.world.Occulter(sim)

        #Real polar function
        a = 0.3
        gfunc = lambda t: 1 + a*np.cos(3*t)

        #Add polar function
        sim.apod_func = gfunc

        #Build polar occulter
        occulter.build_quad_polar()

        #Build directly
        xq, yq, wq = diffraq.quadrature.polar_quad(gfunc, sim.radial_nodes, sim.theta_nodes)

        #Check they are all the ame
        assert(np.isclose(xq, occulter.xq).all() and np.isclose(yq, occulter.yq).all() and \
               np.isclose(wq, occulter.wq).all())

############################################

    def test_circle(self):
        #Build occulter with dummy class
        sim = Dummy_Sim()
        occulter = diffraq.world.Occulter(sim)

        #Real circle function
        r0 = 0.3
        gfunc = lambda t: r0 * np.ones_like(t)

        #Add radius to sim
        sim.circle_rad = r0

        #Build circle occulter
        occulter.build_quad_circle()

        #Build directly
        xq, yq, wq = diffraq.quadrature.polar_quad(gfunc, sim.radial_nodes, sim.theta_nodes)

        #Check they are all the ame
        assert(np.isclose(xq, occulter.xq).all() and np.isclose(yq, occulter.yq).all() and \
               np.isclose(wq, occulter.wq).all())

############################################
############################################

class Dummy_Sim(object):

    def __init__(self):
        self.radial_nodes = 100
        self.theta_nodes = 100

############################################
############################################

if __name__ == '__main__':

    tt = Test_Occulter()
    tt.run_all_tests()
