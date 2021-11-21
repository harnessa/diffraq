"""
test_focuser.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-28-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test propagation of field to image plane.

"""

import diffraq
import numpy as np
from scipy.special import j1

class Test_Focuser(object):
    prop_tol = 0.5

    def run_all_tests(self):
        tsts = ['single_lens']
        for t in tsts:
            getattr(self, f'test_{t}')()

############################################

    def test_single_lens(self):
        waves = np.array([0.3e-6, 0.6e-6, 1.2e-6])

        #Build simulator
        sim = diffraq.Simulator({'waves':waves, 'tel_diameter':5e-3, \
            'focal_length':0.5, 'image_size':74})
        sim.load_focuser()

        #Build uniform pupil image
        pupil = np.ones((len(waves), sim.num_pts, sim.num_pts)) + 0j
        grid_pts = diffraq.utils.image_util.get_grid_points(sim.num_pts, sim.tel_diameter)

        #Get images
        image, image_pts = sim.focuser.calculate_image(pupil, grid_pts)

        #Get Airy disk
        et = np.tile(image_pts, (image.shape[-1],1))
        rr = np.sqrt(et**2 + et.T**2)
        rr[rr == 0] = 1e-12         #j1 blows up at r=0
        xx = 2*np.pi/waves[:,None,None] * sim.tel_diameter/2 * np.sin(rr)
        area = np.pi*sim.tel_diameter**2/4
        I0 = area**2/waves[:,None,None]**2/sim.focuser.image_distance**2
        airy = I0 * (2.*j1(xx)/xx)**2

        #Check
        for i in range(len(waves)):
            assert(np.abs(airy[i] - image[i]).mean() < self.prop_tol)

        #Cleanup
        del pupil, image, airy, rr, xx, grid_pts, image_pts

if __name__ == '__main__':

    tf = Test_Focuser()
    tf.run_all_tests()
