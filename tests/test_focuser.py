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
    prop_tol = 1e-4

    def run_all_tests(self):
        tsts = ['sampling', 'propagation']
        for t in tsts:
            getattr(self, f'test_{t}')()

############################################

    def test_sampling(self):
        waves = [0.3e-6, 0.6e-6, 1.2e-6]
        true_pad = [4, 8, 16]
        targ_pad = [2.30769231, 4.61538462, 9.23076923]     #D=2.4, f=240

        #Build simulator
        sim = diffraq.Simulator({'waves':waves})

        #Check
        for i in range(len(waves)):
            assert(np.isclose(true_pad[i], sim.focuser.true_pad[i]) and \
                np.isclose(targ_pad[i], sim.focuser.target_pad[i]))

############################################

    def test_propagation(self):
        waves = np.array([0.6e-6, 1.2e-6])

        #Build simulator
        sim = diffraq.Simulator({'waves':waves})

        #Build uniform pupil image
        pupil = np.ones((len(waves), sim.num_pts, sim.num_pts)) + 0j

        #Get images
        image = sim.focuser.calculate_image(pupil)

        #Get Airy disk
        rr = diffraq.utils.image_util.get_image_radii(image.shape[-2:]) * \
            sim.focuser.image_res
        rr[rr == 0] = 1e-12         #j1 blows up at r=0
        xx = 2*np.pi/waves[:,None,None] * sim.tel_diameter/2 * np.sin(rr)
        airy = (2.*j1(xx)/xx)**2

        #Check
        for i in range(len(waves)):
            assert(np.abs(airy[i] - image[i]).mean() < self.prop_tol)

if __name__ == '__main__':

    tf = Test_Focuser()
    tf.run_all_tests()
