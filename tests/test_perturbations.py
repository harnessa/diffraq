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

        #Build notch
        xy0 = [self.circle_rad*np.cos(np.pi/6), self.circle_rad*np.sin(np.pi/6)]
        height = 1.25
        width = 2
        notch = {'kind':'notch', 'xy0':xy0, 'height':height, 'width':width, \
            'local_norm':False}

        #Simulation parameters
        params = {'radial_nodes':100, 'theta_nodes':100}

        #Areas
        disk_area = np.pi*self.circle_rad**2
        notch_area = height * width

        #Loop over occulter/aperture
        for is_opq in [False, True]:

            #Loop over positive and negative notches:
            for nch in [1, -1]:

                #Add direction to notch
                notch['direction'] = nch

                #Build shape
                shapes = {'kind':'circle', 'max_radius':self.circle_rad, \
                    'is_opaque':is_opq, 'perturbations':notch}

                #Build simulator
                sim = diffraq.Simulator(params, shapes)

                #Add perturbation
                pert = diffraq.geometry.Notch(sim.occulter.shapes[0], **notch)

                #Get perturbation quadrature
                xp, yp, wp = pert.get_quadrature()

                #Get quadrature
                sim.occulter.build_quadrature()

                #Get current notch area
                cur_area = notch_area*nch
                cur_disk_area = disk_area + cur_area

                #Assert true
                assert(np.isclose(cur_area, wp.sum()) and \
                    np.isclose(cur_disk_area, sim.occulter.wq.sum()))

############################################

    def test_sines(self):
        #Load simulator
        params = {
            'occulter_shape':   'circle',
            'circle_rad':       self.circle_rad,
        }


    # breakpoint()

############################################

if __name__ == '__main__':

    tp = Test_Perturbations()
    tp.run_all_tests()
