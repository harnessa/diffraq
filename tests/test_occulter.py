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
        tsts = ['polar', 'circle', 'cartesian', 'starshades', 'loci']
        for t in tsts:
            getattr(self, f'test_{t}')()

############################################

    def test_polar(self):
        #Build simulator
        sim = diffraq.Simulator({'radial_nodes':100, 'theta_nodes':110, \
            'occulter_shape':'polar'})

        #Real polar function
        a = 0.3
        gfunc = lambda t: 1 + a*np.cos(3*t)

        #Add polar function
        sim.apod_func = gfunc

        #Build polar occulter
        sim.occulter.set_shape_function()
        sim.occulter.build_quadrature()

        #Build directly
        xq, yq, wq = diffraq.quadrature.polar_quad(gfunc, sim.radial_nodes, sim.theta_nodes)

        #Check they are all the same
        assert(np.isclose(xq, sim.occulter.xq).all() and np.isclose(yq, sim.occulter.yq).all() and \
               np.isclose(wq, sim.occulter.wq).all())

        #Cleanup
        del xq, yq, wq

############################################

    def test_circle(self):

        #Real circle function
        r0 = 0.3
        gfunc = lambda t: r0 * np.ones_like(t)

        #Build simulator
        sim = diffraq.Simulator({'radial_nodes':100, 'theta_nodes':110, \
            'occulter_shape':'circle', 'circle_rad':r0})

        #Add radius to sim
        sim.circle_rad = r0

        #Build circle occulter
        sim.occulter.set_shape_function()
        sim.occulter.build_quadrature()

        #Build directly
        xq, yq, wq = diffraq.quadrature.polar_quad(gfunc, sim.radial_nodes, sim.theta_nodes)

        #Check they are all the same
        assert(np.isclose(xq, sim.occulter.xq).all() and np.isclose(yq, sim.occulter.yq).all() and \
               np.isclose(wq, sim.occulter.wq).all())

        #Cleanup
        del xq, yq, wq

############################################

    def test_cartesian(self):
        #Build simulator
        sim = diffraq.Simulator({'radial_nodes':100, 'theta_nodes':110, \
            'occulter_shape':'cartesian'})

        #Kite occulter
        func = lambda t: np.hstack((0.5*np.cos(t) + 0.5*np.cos(2*t), np.sin(t)))
        diff = lambda t: np.hstack((-0.5*np.sin(t) - np.sin(2*t), np.cos(t)))

        #Add functions
        sim.apod_func = func
        sim.apod_diff = diff

        #Build polar occulter
        sim.occulter.set_shape_function()
        sim.occulter.build_quadrature()

        #Build directly
        xq, yq, wq = diffraq.quadrature.cartesian_quad(func, diff, \
            sim.radial_nodes, sim.theta_nodes)

        #Check they are all the same
        assert(np.isclose(xq, sim.occulter.xq).all() and np.isclose(yq, sim.occulter.yq).all() and \
               np.isclose(wq, sim.occulter.wq).all())

        #Cleanup
        del xq, yq, wq

############################################

    def test_starshades(self):
        #Build simulator
        sim = diffraq.Simulator({'radial_nodes':100, 'theta_nodes':110, \
            'occulter_shape':'starshade', 'is_babinet':True})

        #HG function and file
        ss_Afunc = lambda r: np.exp(-((r-sim.ss_rmin)/(sim.ss_rmax-sim.ss_rmin)/0.6)**6)
        ss_Afile = f'{diffraq.int_data_dir}/Test_Data/hg_apod_file.txt'

        #Analytic vs numeric
        afunc_dict = {'analytic':ss_Afunc, 'numeric':None}
        afile_dict = {'analytic':None,     'numeric':ss_Afile}

        #Test analytic and numeric
        for ss in ['analytic', 'numeric']:

            #Set apod values
            sim.apod_func = afunc_dict[ss]
            sim.apod_file = afile_dict[ss]

            #Build occulter
            sim.occulter.set_shape_function()
            sim.occulter.build_quadrature()

            #Get quadrature for comparison
            xq, yq, wq = diffraq.quadrature.starshade_quad(ss_Afunc, sim.num_petals, \
                sim.ss_rmin, sim.ss_rmax, sim.radial_nodes, sim.theta_nodes)

            #Check they are all the same
            assert(np.isclose(xq, sim.occulter.xq).all() and np.isclose(yq, sim.occulter.yq).all() and \
                   np.isclose(wq, sim.occulter.wq).all())

        #Cleanup
        del xq, yq, wq

############################################

    def test_loci(self):
        tol = 1e-3

        #Build simulator
        sim = diffraq.Simulator({'radial_nodes':100, 'occulter_shape':'loci'})

        #Point to loci file
        sim.loci_file = f'{diffraq.int_data_dir}/Test_Data/kite_loci_file.txt'

        #Build loci occulter
        sim.occulter.build_quadrature()

        #Get number of points in loci file
        edge = sim.occulter.get_edge_points()
        theta_nodes = len(edge)

        #Kite occulter
        func = lambda t: np.hstack((0.5*np.cos(t) + 0.5*np.cos(2*t), np.sin(t)))
        diff = lambda t: np.hstack((-0.5*np.sin(t) - np.sin(2*t), np.cos(t)))

        #Add functions
        sim.apod_func = func
        sim.apod_diff = diff

        #Build directly
        xq, yq, wq = diffraq.quadrature.cartesian_quad(func, diff, \
            sim.radial_nodes, theta_nodes)

        #Check they are all close
        assert((np.abs(sim.occulter.xq - xq).max() < tol) and \
               (np.abs(sim.occulter.yq - yq).max() < tol) and \
               (np.abs(sim.occulter.wq - wq).max() < tol))

        #Cleanup
        del xq, yq, wq, edge

############################################
############################################

if __name__ == '__main__':

    tt = Test_Occulter()
    tt.run_all_tests()
