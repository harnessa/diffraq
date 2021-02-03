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
        tsts = ['polar', 'circle', 'starshades']
        for t in tsts:
            getattr(self, f'test_{t}')()

############################################

    def test_polar(self):
        #Build simulator
        sim = diffraq.Simulator({'radial_nodes':100, 'theta_nodes':100, \
            'occulter_shape':'polar'})

        #Real polar function
        a = 0.3
        gfunc = lambda t: 1 + a*np.cos(3*t)

        #Add polar function
        sim.apod_func = gfunc

        #Build polar occulter
        sim.occulter.build_quadrature()

        #Build directly
        xq, yq, wq = diffraq.quadrature.polar_quad(gfunc, sim.radial_nodes, sim.theta_nodes)

        #Check they are all the ame
        assert(np.isclose(xq, sim.occulter.xq).all() and np.isclose(yq, sim.occulter.yq).all() and \
               np.isclose(wq, sim.occulter.wq).all())

        #Cleanup
        del xq, yq, wq

############################################

    def test_circle(self):
        #Build simulator
        sim = diffraq.Simulator({'radial_nodes':100, 'theta_nodes':100, \
            'occulter_shape':'circle'})

        #Real circle function
        r0 = 0.3
        gfunc = lambda t: r0 * np.ones_like(t)

        #Add radius to sim
        sim.circle_rad = r0

        #Build circle occulter
        sim.occulter.build_quadrature()

        #Build directly
        xq, yq, wq = diffraq.quadrature.polar_quad(gfunc, sim.radial_nodes, sim.theta_nodes)

        #Check they are all the ame
        assert(np.isclose(xq, sim.occulter.xq).all() and np.isclose(yq, sim.occulter.yq).all() and \
               np.isclose(wq, sim.occulter.wq).all())

        #Cleanup
        del xq, yq, wq

############################################

    def test_starshades(self):
        #Build simulator
        sim = diffraq.Simulator({'radial_nodes':100, 'theta_nodes':100, \
            'occulter_shape':'starshade', 'is_babinet':True})

        #HG function and file
        ss_Afunc = lambda r: np.exp(-((r-sim.ss_rmin)/(sim.ss_rmax-sim.ss_rmin)/0.6)**6)
        ss_Afile = f'{diffraq.int_data_dir}/Test_Data/test_apod_file.txt'

        #Analytic vs numeric
        afunc_dict = {'analytic':ss_Afunc, 'numeric':None}
        afile_dict = {'analytic':None,     'numeric':ss_Afile}

        #Test analytic and numeric
        for ss in ['analytic', 'numeric']:

            #Set apod values
            sim.apod_func = afunc_dict[ss]
            sim.apod_file = afile_dict[ss]

            #Build occulter
            sim.occulter.build_quadrature()

            #Get quadrature for comparison
            xq, yq, wq = diffraq.quadrature.starshade_quad(ss_Afunc, sim.num_petals, \
                sim.ss_rmin, sim.ss_rmax, sim.radial_nodes, sim.theta_nodes)

            #Check they are all the ame
            assert(np.isclose(xq, sim.occulter.xq).all() and np.isclose(yq, sim.occulter.yq).all() and \
                   np.isclose(wq, sim.occulter.wq).all())

        #Cleanup
        del xq, yq, wq

############################################
############################################

if __name__ == '__main__':

    tt = Test_Occulter()
    tt.run_all_tests()
