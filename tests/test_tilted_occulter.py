"""
test_tilted_occulter.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-12-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test of tilted occulter using circles.

"""

import diffraq
import numpy as np

class Test_Tilted(object):

    num_pts = 256*2
    radial_nodes = 600
    theta_nodes = 600
    zz = 50
    z0 = 27
    circle_rad = 12e-3
    xtilt = 20
    ytilt = 30

    def run_all_tests(self):
        for oa in ['occulter', 'aperture']:
            for op in ['plane', 'spherical']:
                getattr(self, f'test_{oa}_{op}')()

############################################

    def run_calculation(self, is_opaque, z0):
        #Load simulator
        params = {
            'radial_nodes':     self.radial_nodes,
            'theta_nodes':      self.theta_nodes,
            'num_pts':          self.num_pts,
            'tel_diameter':     1.25*self.circle_rad*(self.zz + z0)/z0,
            'zz':               self.zz,
            'z0':               z0,
            'skip_image':       True,
        }

        cir_func = lambda t: self.circle_rad * np.ones_like(t)
        cir_diff = lambda t: np.zeros_like(t)

        #Permutations of tilts
        tilts = [[self.xtilt,0], [0, self.ytilt]]

        for tilt in tilts:

            #Circle
            circle = {'kind':'polar', 'is_opaque':is_opaque, 'max_radius':self.circle_rad,
                'edge_func':cir_func, 'edge_diff':cir_diff}

            params['tilt_angle'] = tilt
            cir_sim = diffraq.Simulator(params, shapes=circle)

            #Ellipse
            ella = self.circle_rad*np.cos(np.radians(tilt[0]))
            ellb = self.circle_rad*np.cos(np.radians(tilt[1]))
            ell_func = lambda t: ella*ellb / np.sqrt(ella**2*np.cos(t)**2 + ellb**2*np.sin(t)**2)
            ell_diff = lambda t: -ella*ellb * np.cos(t)*np.sin(t)*(-ella**2 + ellb**2) / \
                (ella**2*np.cos(t)**2 + ellb**2*np.sin(t)**2)**(3/2)

            tol = 1e-8

            ellipse = {'kind':'polar', 'is_opaque':is_opaque, 'max_radius':self.circle_rad,
                'edge_func':ell_func, 'edge_diff':ell_diff}

            params['tilt_angle'] = [0, 0]
            ell_sim = diffraq.Simulator(params, shapes=ellipse)

            #Get pupil field from sim
            cir_pupil, cir_grid_pts = cir_sim.calc_pupil_field()
            ell_pupil, ell_grid_pts = ell_sim.calc_pupil_field()

            cir_pupil = cir_pupil[0]
            ell_pupil = ell_pupil[0]

            #Compare
            diff = np.abs(cir_pupil - ell_pupil)
            assert(diff.max() < tol)

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

if __name__ == '__main__':

    tt = Test_Tilted()
    tt.run_all_tests()
