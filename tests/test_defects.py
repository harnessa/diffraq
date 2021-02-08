"""
test_defects.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test quadrature of adding various defects.

"""

import diffraq
import numpy as np

class Test_Defects(object):

    circle_rad = 12

    def run_all_tests(self):
        for dd in ['notch', 'sines']:
            getattr(self, f'test_{dd}')()

############################################

    def test_notch(self):
        #Load simulator
        params = {
            'occulter_shape':   'circle',
            'is_babinet':       True,
            'circle_rad':       self.circle_rad,
        }

        #Loop over occulter/aperture
        for bab in [False, True]:

            params['is_babinet'] = bab

            #Generate simulator
            sim = diffraq.Simulator(params)

            #Get quadrature
            sim.occulter.build_quadrature()
            breakpoint()

############################################

if __name__ == '__main__':

    td = Test_Defects()
    td.run_all_tests()
