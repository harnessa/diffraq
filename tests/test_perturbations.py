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

    def run_all_tests(self):
        for dd in ['notch', 'sines']:
            getattr(self, f'test_{dd}')()

############################################

    def test_notch(self):
        #Load simulator
        params = {
            'occulter_shape':   'circle',
            'circle_rad':       self.circle_rad,
        }


        #Build notch
        center = [self.circle_rad*np.cos(np.pi/6), self.circle_rad*np.sin(np.pi/6)]
        height = 1
        width = 2
        notch = {'center':center, 'height':height, 'width':width}

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

                #Get quadrature
                sim.occulter.build_quadrature()
                breakpoint()

############################################

if __name__ == '__main__':

    tp = Test_Perturbations()
    tp.run_all_tests()
