"""
test_circles.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-02-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test of pupil field calculation of circular occulter and comparison
    to analytic solution given by Lommel Functions. This tests diverging beam.

"""

import diffraq
import numpy as np

class Test_Circles(object):

    tol = 1e-8

    num_pts = 256
    radial_nodes = 400
    theta_nodes = 400
    zz = 15e6
    z0 = 1e19
    circle_rad = 12

    def run_all_tests(self):
        tsts = ['occulter_plane', 'occulter_spherical', 'aperture_plane',\
            'aperture_spherical'][:]
        for t in tsts:
            getattr(self, f'test_{t}')()

############################################

    def run_calculation(self, is_babinet, z0):
        """Test opaque circular occulter with plane wave"""
        #Load simulator
        params = {
            'occulter_shape':   'circle',
            'is_babinet':       is_babinet,
            'circle_rad':       self.circle_rad,
            'radial_nodes':     self.radial_nodes,
            'theta_nodes':      self.theta_nodes,
            'num_pts':          self.num_pts,
            'tel_diameter':     3*self.circle_rad*(self.zz + z0)/z0,
            'zz':               self.zz,
            'z0':               z0,
            'skip_image':       True,
        }

        sim = diffraq.Simulator(params)

        #Get pupil field from sim
        pupil, grid_pts = sim.calc_pupil_field()
        pupil = pupil[0][len(pupil[0])//2]

        #Get pupil field from diffraction_points directly
        pupil_pts = self.calc_diff_points(sim, grid_pts)
        pupil_pts = pupil_pts[0][len(pupil_pts[0])//2]

        #Calculate analytic solution
        utru = diffraq.utils.solution_util.calculate_circle_solution(grid_pts, \
            sim.waves[0], sim.zz, sim.z0, sim.circle_rad, is_babinet)

        #Compare
        assert(np.abs(pupil - utru).max() < self.tol)
        assert(np.abs(pupil_pts - utru).max() < self.tol)

############################################

    def test_occulter_plane(self):
        self.run_calculation(True, self.z0)

    def test_occulter_spherical(self):
        self.run_calculation(True, self.zz)

    def test_aperture_plane(self):
        self.run_calculation(False, self.z0)

    def test_aperture_spherical(self):
        self.run_calculation(False, self.zz)

############################################

    def calc_diff_points(self, sim, grid_pts):

        #Create empty pupil field array
        pupil = np.empty((len(sim.waves), sim.num_pts, sim.num_pts)) + 0j

        #Flatten grid
        grid_2D = np.tile(grid_pts, (len(grid_pts),1)).T
        xi = grid_2D.flatten()
        eta = grid_2D.T.flatten()

        #Run diffraction calculation over wavelength
        for iw in range(len(sim.waves)):

            #lambda * z
            lamzz = sim.waves[iw] * sim.zz
            lamz0 = sim.waves[iw] * sim.z0

            #Calculate diffraction
            uu = diffraq.diffraction.diffract_points(sim.occulter.xq, \
                sim.occulter.yq, sim.occulter.wq, lamzz, xi, eta, sim.fft_tol,
                is_babinet=sim.is_babinet, lamz0=lamz0)

            #Store
            pupil[iw] = uu.reshape(grid_2D.shape)

        #Cleanup
        del grid_2D, xi, eta

        return pupil

############################################

if __name__ == '__main__':

    tc = Test_Circles()
    tc.run_all_tests()
