"""
test_perturbations.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test quadrature of adding various perturbations.

"""

import diffraq
import numpy as np

class Test_Perturbations(object):

    circle_rad = 12
    tol = 1e-5

    def run_all_tests(self):
        for dd in ['notch', 'sines'][:1]:
            getattr(self, f'test_{dd}')()

############################################

    def test_notch(self):
        #Load simulator
        params = {
            'occulter_shape':   'circle',
            'circle_rad':       self.circle_rad,
        }

        #Build notch
        xy0 = [self.circle_rad*np.cos(np.pi/6), self.circle_rad*np.sin(np.pi/6)]
        height = 1.25
        width = 2
        notch = {'xy0':xy0, 'height':height, 'width':width}

        #Areas
        disk_area = np.pi*self.circle_rad**2
        notch_area = height * width

        #Loop over occulter/aperture
        for bab in [False, True]:

            #Loop over positive and negative notches:
            for nch in [1, -1]:

                #Add direction
                notch['direction'] = nch

                #Add to parameters
                params['is_babinet'] = bab
                params['perturbations'] = {'notch':notch}

                #Generate simulator
                sim = diffraq.Simulator(params)

                #Get perturbation quadrature
                xp, yp, wp = sim.occulter.build_perturbation_quadrature()

                #Get quadrature
                sim.occulter.build_quadrature()

                #Get current notch area
                cur_area = notch_area*nch
                cur_disk_area = disk_area + cur_area

                #Assert true
                assert(np.isclose(cur_area, wp.sum()) and \
                    (cur_disk_area, sim.occulter.wq.sum()))

############################################

if __name__ == '__main__':

    tp = Test_Perturbations()
    tp.run_all_tests()
