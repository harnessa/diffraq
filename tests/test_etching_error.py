"""
test_etching_error.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-04-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test of etching error by applying error to circle and comparing
    to analytic results.

"""

import diffraq
import numpy as np

class Test_Etching(object):

    tol = 1e-6

    num_pts = 256
    radial_nodes = 40
    theta_nodes = 40
    zz = 15e6
    z0 = 1e19
    tel_diameter = 2.4
    circle_rad = 12
    etch = 1

    def run_all_tests(self):
        for oa in ['occulter', 'aperture']:
            for oe in ['over', 'under']:
                getattr(self, f'test_{oa}_{oe}')()

############################################

    def run_calculation(self, is_opaque, etch_sign):
        #Load simulator
        params = {
            'radial_nodes':     self.radial_nodes,
            'theta_nodes':      self.theta_nodes,
            'num_pts':          self.num_pts,
            'tel_diameter':     self.tel_diameter,
            'zz':               self.zz,
            'z0':               self.z0,
            'skip_image':       True,
        }

        cart_func = lambda t: self.circle_rad * np.hstack((np.cos(t), np.sin(t)))
        cart_diff = lambda t: self.circle_rad * np.hstack((-np.sin(t), np.cos(t)))
        polar_func = lambda t: self.circle_rad * np.ones_like(t)
        polar_diff = lambda t: np.zeros_like(t)
        alp, gam = 10, self.circle_rad*6
        petal_func = lambda r: alp*np.exp(-r**2/gam)
        petal_diff = lambda r: alp*-2*r/gam * np.exp(-r**2/gam)

        shape = {'is_opaque':is_opaque, 'max_radius':self.circle_rad}

        utru = None

        #Loop over occulter types (petal is only for visual checking, no area calculation)
        for occ_shape in ['polar', 'cartesian', 'petal'][:2]:
            #Loop over function vs interpolation (don't reun function for now)
            for is_func in [False, True][:1]:

                #Only do function for cartesian
                if is_func and occ_shape != 'cartesian':
                    continue

                #Set parameters
                shape['kind'] = occ_shape

                if occ_shape == 'cartesian':
                    shape['edge_func'] = cart_func
                    shape['edge_diff'] = cart_diff
                    shape['etch_error'] = self.etch * etch_sign
                elif occ_shape == 'polar':
                    shape['edge_func'] = polar_func
                    shape['edge_diff'] = polar_diff
                    shape['etch_error'] = self.etch * etch_sign
                elif occ_shape == 'petal':
                    shape['edge_func'] = petal_func
                    shape['edge_diff'] = petal_diff
                    shape['etch_error'] = self.etch / 5 * etch_sign

                #Add data points if not function
                if is_func:
                    shape['edge_data'] = None
                else:
                    npts = self.radial_nodes * self.theta_nodes
                    if occ_shape == 'petal':
                        tt = np.linspace(self.circle_rad/2, self.circle_rad, npts)[:,None]
                    else:
                        tt = np.linspace(0, 2*np.pi, npts)[:,None]
                    shape['edge_data'] = np.hstack((tt, shape['edge_func'](tt)))

                #Load simulator
                sim = diffraq.Simulator(params, shapes=shape)

                #Build quad
                sim.occulter.build_quadrature()

                #Compare area
                area = sim.occulter.wq.sum()
                tru_area = np.pi*(self.circle_rad + self.etch*etch_sign)**2

                assert(np.isclose(area, tru_area))

                #Get pupil field from sim
                pupil, grid_pts = sim.calc_pupil_field()
                pupil = pupil[0][len(pupil[0])//2]

                #Calculate analytic solution (once)
                if utru is None:
                    utru = diffraq.utils.solution_util.calculate_circle_solution(grid_pts, \
                        sim.waves[0], sim.zz, sim.z0, \
                        self.circle_rad + self.etch*etch_sign, is_opaque)

                #Compare
                assert(np.abs(pupil - utru).max() < self.tol)

        #Clean up
        del pupil, grid_pts, sim, utru

############################################

    def test_occulter_over(self):
        self.run_calculation(True, 1)

    def test_occulter_under(self):
        self.run_calculation(True, -1)

    def test_aperture_over(self):
        self.run_calculation(False, 1)

    def test_aperture_under(self):
        self.run_calculation(False, -1)

############################################

if __name__ == '__main__':

    te = Test_Etching()
    te.run_all_tests()
