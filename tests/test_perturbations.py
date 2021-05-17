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

    circle_rad = 12.5
    tol = 1e-5

    def run_all_tests(self):
        for dd in ['notch', 'shifted_petal', 'pinhole', 'sines']:
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

                #Get current notch area (sign of sum is direction * opaque)
                cur_area = -notch_area*nch * (2*int(is_opq) -1)
                cur_disk_area = disk_area + cur_area

                #Assert true
                assert(np.isclose(cur_area, wp.sum()) and \
                    np.isclose(cur_disk_area, sim.occulter.wq.sum()))

############################################

    def test_shifted_petal(self):

        #Simulated shape
        num_pet = 12
        rmin, rmax = 8e-3, 10e-3
        max_apod = 0.9

        ss_Afunc = lambda r: 1 + max_apod*(np.exp(-((r-rmin)/(rmax-rmin)/0.6)**6) - 1)
        inn_shape = {'kind':'starshade', 'is_opaque':True, 'num_petals':num_pet, \
            'edge_func':ss_Afunc, 'min_radius':rmin, 'max_radius':rmax}

        #Get nominal area
        sim = diffraq.Simulator({}, inn_shape)
        sim.occulter.build_quadrature()
        area0 = -sim.occulter.wq.sum()
        sim.clean_up()

        #Petal shifts to test
        petal_shifts = {1: 7.5e-6, 5: 25e-6, 9: 10.5e-6}

        #Loop over shifts
        for pet_num, shift in petal_shifts.items():

            #Angles between petals (add 1/2 petal since opaque)
            angles = np.pi/num_pet * np.array([2*pet_num-1, 2*pet_num + 1])

            #Build shifted petal perturbation
            pert_n = {'kind':'shifted_petal', 'angles':angles, 'shift':shift}

            #Add perturbation to shape
            inn_shape['perturbations'] = [pert_n]

            #Build sim
            sim = diffraq.Simulator({}, inn_shape)

            #Get area
            sim.occulter.build_quadrature()
            cur_area = -sim.occulter.wq.sum()

            #Get difference in area
            darea = area0 - cur_area

            #True area difference
            da_tru = np.pi*((rmin + shift)**2 - rmin**2) / num_pet

            #Cleanup
            sim.clean_up()

            #Check is true
            assert(np.isclose(darea, da_tru))

############################################

    def test_pinhole(self):

        #Build notch
        xy0 = [self.circle_rad/2*np.cos(np.pi/6), self.circle_rad/2*np.sin(np.pi/6)]
        pinhole = {'kind':'pinhole', 'xy0':xy0, 'radius':0.5}

        #Simulation parameters
        params = {'radial_nodes':100, 'theta_nodes':100}

        #Areas
        disk_area = np.pi*self.circle_rad**2
        hole_area = np.pi * pinhole['radius']**2

        #Build shape
        shapes = {'kind':'circle', 'max_radius':self.circle_rad, \
            'is_opaque':True, 'perturbations':pinhole}

        #Build simulator
        sim = diffraq.Simulator(params, shapes)

        #Add perturbation
        pert = diffraq.geometry.Pinhole(sim.occulter.shapes[0], **pinhole)

        #Get perturbation quadrature
        xp, yp, wp = pert.get_quadrature()

        #Get quadrature
        sim.occulter.build_quadrature()

        #Get current notch area
        cur_area = -hole_area
        cur_disk_area = disk_area + cur_area

        #Assert true
        assert(np.isclose(cur_area, wp.sum()) and \
            np.isclose(cur_disk_area, sim.occulter.wq.sum()))

############################################

    def test_sines(self):
        #Simulation parameters
        params = {'radial_nodes':100, 'theta_nodes':100}

        #Sine wave
        xy0 = [self.circle_rad*np.cos(np.pi/3), self.circle_rad*np.sin(np.pi/3)]
        sine = {'kind':'sines', 'xy0':xy0, 'amplitude':1.5, 'frequency':10,
            'num_cycles':5}

        #Loop over occulter/aperture
        for is_opq in [False, True]:

            #Build shape
            shape = {'kind':'circle', 'max_radius':self.circle_rad, \
                'is_opaque':is_opq}

            #Get nominal area
            sim = diffraq.Simulator(params, shape)
            sim.occulter.build_quadrature()
            area0 = sim.occulter.wq.sum()
            sim.clean_up()

            #Add perturbation
            shape['perturbations'] = sine

            #Build simulator
            sim = diffraq.Simulator(params, shape)

            #Add perturbation
            pert = diffraq.geometry.Sines(sim.occulter.shapes[0], **sine)

            #Get perturbation quadrature
            t0 = pert.get_start_point()
            xp, yp, wp = pert.get_quad_polar(t0, do_test=True)

            #Get quadrature
            sim.occulter.build_quadrature()

            #Get sine area
            sin_area = wp.sum()

            #Get total area
            area = sim.occulter.wq.sum()

            #Assert areas are close
            assert(np.isclose(area0 + sin_area, area))

############################################

if __name__ == '__main__':

    tp = Test_Perturbations()
    tp.run_all_tests()
