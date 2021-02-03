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
        true_NN = [594, 1188, 2376]
        targ_NN = [590.76923077, 1181.53846154, 2363.07692308]  #D=2.4, f=240

        #Build simulator
        sim = diffraq.Simulator({'waves':waves})
        sim.load_focuser()

        #Check
        for i in range(len(waves)):
            assert(np.isclose(true_NN[i], sim.focuser.true_NN[i]) and \
                np.isclose(targ_NN[i], sim.focuser.targ_NN[i]))

############################################

    def test_propagation(self):
        waves = np.array([0.3e-6, 0.6e-6, 1.2e-6])

        #Build simulator
        sim = diffraq.Simulator({'waves':waves})
        sim.load_focuser()

        #Build uniform pupil image
        pupil = np.ones((len(waves), sim.num_pts, sim.num_pts)) + 0j

        #Get images
        image, grid_pts = sim.focuser.calculate_image(pupil)

        #Get Airy disk
        et = np.tile(grid_pts, (image.shape[-1],1))
        rr = np.sqrt(et**2 + et.T**2)
        rr[rr == 0] = 1e-12         #j1 blows up at r=0
        xx = 2*np.pi/waves[:,None,None] * sim.tel_diameter/2 * np.sin(rr)
        airy = (2.*j1(xx)/xx)**2

        #Check
        for i in range(len(waves)):
            assert(np.abs(airy[i] - image[i]).mean() < self.prop_tol)

        #Cleanup
        del pupil, image, airy, rr

if __name__ == '__main__':

    tf = Test_Focuser()
    tf.run_all_tests()
