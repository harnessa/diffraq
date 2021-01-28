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

class Test_Focuser(object):

    def run_all_tests(self):
        tsts = ['sampling', 'propagation']
        for t in tsts:
            getattr(self, f'test_{t}')()

############################################

    def test_sampling(self):
        waves = [0.3e-6, 0.6e-6, 1.2e-6]
        true_pad = [2, 4, 8]
        targ_pad = [1.15384615, 2.30769231, 4.61538462]     #D=2.4, f=120

        #Build simulator
        sim = diffraq.Simulator({'waves':waves})

        #Check
        for i in range(len(waves)):
            assert(np.isclose(true_pad[i], sim.focuser.true_pad[i]) and \
                np.isclose(targ_pad[i], sim.focuser.target_pad[i]))

############################################

    def test_propagation(self):
        waves = [0.6e-6, 1.2e-6]

        #Build simulator
        sim = diffraq.Simulator({'waves':waves})

        #Build uniform pupil image
        pupil = np.ones((len(waves), sim.num_pts, sim.num_pts)) + 0j

        #Get images
        image = sim.focuser.calculate_image(pupil)

        import matplotlib.pyplot as plt;plt.ion()
        plt.imshow(image[0])
        breakpoint()

if __name__ == '__main__':

    tf = Test_Focuser()
    tf.run_all_tests()
