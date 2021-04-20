"""
test_seam_perturbations.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 04-19-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test quadrature of vector seams on perturbations.

"""

import diffraq
import numpy as np

class Test_Seam_Perturbations(object):

    circle_rad = 12
    seam_width = 0.1
    tol = 2e-4

    def run_all_tests(self):
        for dd in ['notch', 'shifted_petal'][:1]:
            getattr(self, f'test_{dd}')()

############################################

    def test_notch(self):

        #Build notch
        xy0 = [self.circle_rad*np.cos(np.pi/6), self.circle_rad*np.sin(np.pi/6)]
        height = 1.25
        width = 1.75
        notch = {'kind':'notch', 'xy0':xy0, 'height':height, 'width':width, \
            'local_norm':False}

        #Maxwell function
        maxwell_func = lambda d, w: [np.heaviside(d, 1)+0j]*2

        #Simulation parameters
        params = {'radial_nodes':100, 'theta_nodes':100, 'do_run_vector':True, \
            'seam_width':self.seam_width, 'maxwell_func':maxwell_func}

        #Areas
        disk_area = np.pi*self.circle_rad**2

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
                pert = diffraq.polarization.Seam_Notch(sim.occulter.shapes[0], **notch)

                #Get perturbation quadrature
                xq, yq, wq, dq, nq, nxq, nyq, nwq = pert.get_quadrature()

                #Get full seam area
                rmax = self.circle_rad - height*nch*(2*int(is_opq)-1)
                alp = 2*np.arcsin(width/2/self.circle_rad)  #angle of notch
                tru_seam_area = np.pi*((rmax + self.seam_width)**2 - \
                    (rmax - self.seam_width)**2) * alp/(2*np.pi)
                seam_area = wq.sum()

                #Check seam area is close
                assert(abs(1 - seam_area/tru_seam_area) < self.tol)

                #Get area of open seam (in aperture)
                tru_open_area = tru_seam_area/2
                open_area = (wq * maxwell_func(dq, 0)[0].real).sum()

                #Check open area is half area (performs worse)
                assert(abs(1 - open_area/tru_open_area) < self.tol*50)

                #Get full seam quadrature
                xq, yq, wq, nq, dq, gw = \
                    sim.vector.seams[0].build_seam_quadrature(self.seam_width)

                #Full seam area (full area - original seam  + new seam)
                full_A = np.pi*((self.circle_rad + self.seam_width)**2 - \
                    (self.circle_rad - self.seam_width)**2)
                old_A = full_A * alp/(2.*np.pi)
                new_A = tru_seam_area
                tru_full_area = full_A - old_A + new_A
                full_area = wq.sum()

                assert(abs(1 - full_area/tru_full_area) < self.tol)

        #Cleanup
        sim.clean_up()
        del sim, pert, xq, yq, wq, nq, dq, nxq, nyq, nwq

############################################
    #
    # def _test_shifted_petal(self):
    #
    #     #Simulated shape
    #     num_pet = 12
    #     rmin, rmax = 8e-3, 10e-3
    #     max_apod = 0.9
    #
    #     ss_Afunc = lambda r: 1 + max_apod*(np.exp(-((r-rmin)/(rmax-rmin)/0.6)**6) - 1)
    #     inn_shape = {'kind':'starshade', 'is_opaque':True, 'num_petals':num_pet, \
    #         'edge_func':ss_Afunc, 'min_radius':rmin, 'max_radius':rmax}
    #
    #     #Get nominal area
    #     sim = diffraq.Simulator({}, inn_shape)
    #     sim.occulter.build_quadrature()
    #     area0 = -sim.occulter.wq.sum()
    #     sim.clean_up()
    #
    #     #Petal shifts to test
    #     petal_shifts = {1: 7.5e-6, 5: 25e-6, 9: 10.5e-6}
    #
    #     #Loop over shifts
    #     for pet_num, shift in petal_shifts.items():
    #
    #         #Angles between petals
    #         angles = np.pi/num_pet * np.array([2*(pet_num-1), 2*pet_num])
    #
    #         #Build shifted petal perturbation
    #         pert_n = {'kind':'shifted_petal', 'angles':angles, 'shift':shift}
    #
    #         #Add perturbation to shape
    #         inn_shape['perturbations'] = [pert_n]
    #
    #         #Build sim
    #         sim = diffraq.Simulator({}, inn_shape)
    #
    #         #Get area
    #         sim.occulter.build_quadrature()
    #         cur_area = -sim.occulter.wq.sum()
    #
    #         #Get difference in area
    #         darea = area0 - cur_area
    #
    #         #True area difference
    #         da_tru = np.pi*((rmin + shift)**2 - rmin**2) / num_pet
    #
    #         #Cleanup
    #         sim.clean_up()
    #
    #         #Check is true
    #         assert(np.isclose(darea, da_tru))

############################################

if __name__ == '__main__':

    tp = Test_Seam_Perturbations()
    tp.run_all_tests()
