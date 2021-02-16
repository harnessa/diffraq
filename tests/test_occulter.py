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
        #Real polar function
        a = 0.3
        gfunc = lambda t: 1 + a*np.cos(3*t)

        #Simulation parameters
        params = {'radial_nodes':100, 'theta_nodes':100}

        #Simulated shapes
        shapes = {'kind':'polar', 'edge_func':gfunc}

        #Build simulator
        sim = diffraq.Simulator(params, shapes)

        #Build polar occulter
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

        #Simulation parameters
        params = {'radial_nodes':100, 'theta_nodes':100}

        #Simulated shapes
        shapes = {'kind':'circle', 'max_radius':r0}

        #Build simulator
        sim = diffraq.Simulator(params, shapes)

        #Build circle occulter
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
        #Kite occulter
        func = lambda t: np.hstack((0.5*np.cos(t) + 0.5*np.cos(2*t), np.sin(t)))
        diff = lambda t: np.hstack((-0.5*np.sin(t) - np.sin(2*t), np.cos(t)))

        #Simulation parameters
        params = {'radial_nodes':100, 'theta_nodes':100}

        #Simulated shapes
        shapes = {'kind':'cartesian', 'edge_func':func, 'edge_diff':diff}

        #Build simulator
        sim = diffraq.Simulator(params, shapes)

        #Build polar occulter
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
        rmin, rmax = 5, 13
        num_petals = 16

        #HG function and file
        ss_Afunc = lambda r: np.exp(-((r-rmin)/(rmax-rmin)/0.6)**6)
        ss_Afile = f'{diffraq.int_data_dir}/Test_Data/hg_apod_file.txt'

        #Analytic vs numeric
        afunc_dict = {'analytic':ss_Afunc, 'numeric':None}
        afile_dict = {'analytic':None,     'numeric':ss_Afile}

        #Simulation parameters
        params = {'radial_nodes':100, 'theta_nodes':100}

        #Simulated shape
        shape = {'kind':'starshade', 'is_opaque':True, \
            'min_radius':rmin, 'max_radius':rmax, 'num_petals':num_petals}

        #Test analytic and numeric
        for ss in ['analytic', 'numeric']:

            #Set apod values
            shape['edge_func'] = afunc_dict[ss]
            shape['edge_file'] = afile_dict[ss]

            #Build simulator
            sim = diffraq.Simulator(params, shape)

            #Build occulter
            sim.occulter.build_quadrature()

            #Get quadrature for comparison
            xq, yq, wq = diffraq.quadrature.starshade_quad(ss_Afunc, num_petals, \
                rmin, rmax, sim.radial_nodes, sim.theta_nodes)

            #Check they are all the same
            assert(np.isclose(xq, sim.occulter.xq).all() and np.isclose(yq, sim.occulter.yq).all() and \
                   np.isclose(wq, sim.occulter.wq).all())

        #Cleanup
        del xq, yq, wq

############################################

    def test_loci(self):
        tol = 1e-3

        #Point to loci file
        loci_file = f'{diffraq.int_data_dir}/Test_Data/kite_loci_file.txt'

        #Simulation parameters
        params = {'radial_nodes':100, 'theta_nodes':100}

        #Simulated shapes
        shapes = {'kind':'loci', 'loci_file':loci_file}

        #Build simulator
        sim = diffraq.Simulator(params, shapes)

        #Build loci occulter
        sim.occulter.build_quadrature()

        #Get number of points in loci file
        sim.occulter.build_edge()
        theta_nodes = len(sim.occulter.edge)

        #Kite occulter
        func = lambda t: np.hstack((0.5*np.cos(t) + 0.5*np.cos(2*t), np.sin(t)))
        diff = lambda t: np.hstack((-0.5*np.sin(t) - np.sin(2*t), np.cos(t)))

        #Build directly
        xq, yq, wq = diffraq.quadrature.cartesian_quad(func, diff, \
            sim.radial_nodes, theta_nodes)

        #Check they are all close
        assert((np.abs(sim.occulter.xq - xq).max() < tol) and \
               (np.abs(sim.occulter.yq - yq).max() < tol) and \
               (np.abs(sim.occulter.wq - wq).max() < tol))

        #Cleanup
        del xq, yq, wq

############################################
############################################

if __name__ == '__main__':

    tt = Test_Occulter()
    tt.run_all_tests()
