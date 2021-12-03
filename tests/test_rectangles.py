"""
test_rectangles.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 12-03-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test of pupil field calculation of rectangular occulter and comparison
    to analytic solution. This tests diverging beam.

"""

import diffraq
import numpy as np

class Test_Rectangles(object):

    tol = 1e-6

    num_pts = 256
    radial_nodes = 600
    theta_nodes = 600
    z0 = 1e19
    zz = 50
    circle_rad = 12.e-3

    def run_all_tests(self):
        for op in ['plane', 'spherical']:
            getattr(self, f'test_{op}')()

############################################

    def run_calculation(self, z0):
        #Load simulator
        params = {
            'radial_nodes':     self.radial_nodes,
            'theta_nodes':      self.theta_nodes,
            'num_pts':          self.num_pts,
            'tel_diameter':     1.5*self.circle_rad*(self.zz + z0)/z0,
            'zz':               self.zz,
            'z0':               z0,
            'skip_image':       True,
        }

        #Loop over aspect ratio
        for aspect in [1, 1/3, 1.25]:

            shape = {'kind':'rectangle', 'is_opaque':False, \
                'width':self.circle_rad, 'height':self.circle_rad*aspect}

            #Load simulator
            sim = diffraq.Simulator(params, shapes=shape)

            #Get pupil field from sim
            pupil, grid_pts = sim.calc_pupil_field()

            #Build 2D observation points
            grid_2D = np.tile(grid_pts, (len(grid_pts),1))

            #Calculate analytic solution
            utru = diffraq.utils.solution_util.calculate_rectangle_solution(grid_2D, \
                sim.waves[0], sim.zz, sim.z0, shape['width'], shape['height'])

            #Compare
            assert(np.abs(pupil - utru).max() < self.tol)

        #Clean up
        del pupil, grid_pts, sim, utru

############################################

    def test_plane(self):
        self.run_calculation(self.z0)

    def test_spherical(self):
        self.run_calculation(self.zz)

############################################

if __name__ == '__main__':

    tr = Test_Rectangles()
    tr.run_all_tests()
