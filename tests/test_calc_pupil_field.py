"""
test_calc_pupil_field.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-18-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test of pupil field calculation.

"""

import diffraq
import numpy as np

class Test_calc_pupil_field(object):

    def test_pupil_field(self):

        fresnums = [10,100,1000]
        shapes = ['circle', 'polar']

        for shape in shapes:

            #Load simulator
            params = {
                'do_save':          False,
                'z1':               1,
                'waves':            1/np.array(fresnums),
                'occulter_shape':   shape,
                'circle_rad':       1,
                'apod_func':        "lambda t: 1 + 0.3*np.cos(3*t)",
            }

            sim = diffraq.Simulator(params, do_start_up=True)

            #Get pupil field
            pupil = sim.calculate_pupil_field()

            #Get theoretical field

if __name__ == '__main__':

    tst = Test_calc_pupil_field()
    tst.test_pupil_field()
