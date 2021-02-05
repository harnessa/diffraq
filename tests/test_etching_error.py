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

    tol = 1e-8

    num_pts = 256
    radial_nodes = 400
    theta_nodes = 400
    zz = 15e6
    z0 = 1e19
    tel_diameter = 2.4
    circle_rad = 12
    etch = 1e-3

    def run_all_tests(self):
        for oa in ['occulter', 'aperture']:
            for oe in ['over', 'under']:
                getattr(self, f'test_{oa}_{oe}')()

############################################

    def run_calculation(self, is_babinet, etch_sign):
        """Test opaque circular occulter with plane wave"""
        #Load simulator
        params = {
            'occulter_shape':   'circle',
            'is_babinet':       is_babinet,
            'circle_rad':       self.circle_rad,
            'etching_error':    self.etch*etch_sign,
            'radial_nodes':     self.radial_nodes,
            'theta_nodes':      self.theta_nodes,
            'num_pts':          self.num_pts,
            'tel_diameter':     self.tel_diameter,
            'zz':               self.zz,
            'z0':               self.z0,
            'skip_image':       True,
        }

        sim = diffraq.Simulator(params)

        #Get pupil field from sim
        pupil, grid_pts = sim.calc_pupil_field()
        pupil = pupil[0][len(pupil[0])//2]

        #Calculate analytic solution
        utru = diffraq.utils.solution_util.calculate_circle_solution(grid_pts, \
            sim.waves[0], sim.zz, sim.z0, \
            sim.circle_rad + self.etching_error*etch_sign, is_babinet)

        import matplotlib.pyplot as plt;plt.ion()
        breakpoint()
        #Compare
        assert(np.abs(pupil - utru).max() < self.tol)

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
