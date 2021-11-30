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
    prop_tol = 0.1

    def run_all_tests(self):
        tsts = ['single_lens', 'lens_system']
        for t in tsts:
            getattr(self, f'test_{t}')()

############################################

    def test_single_lens(self):
        waves = np.array([0.3e-6, 0.6e-6, 1.2e-6])

        Dtel = 5e-3
        focal_length = 0.5

        lens_system = {
            'element_0': {'kind':'lens', 'lens_name':'simple', \
            'focal_length':focal_length, 'diameter':Dtel, 'distance':focal_length},
        }

        #Loop through with and without lens systems
        for lenses in [None, lens_system]:

            #Build simulator
            sim = diffraq.Simulator({'num_pts':512, 'waves':waves, 'tel_diameter':Dtel, \
                'focal_length':focal_length, 'image_size':74, 'lens_system':lenses})
            sim.load_focuser()

            #Build uniform pupil image
            pupil = np.ones((len(waves), sim.num_pts, sim.num_pts)) + 0j
            grid_pts = diffraq.utils.image_util.get_grid_points(sim.num_pts, sim.tel_diameter)

            #Get images
            image, image_pts = sim.focuser.calculate_image(pupil[:,None], grid_pts)
            image = image[:,0]

            #Get Airy disk
            et = np.tile(image_pts, (image.shape[-1],1))
            rr = np.sqrt(et**2 + et.T**2)
            rr[rr == 0] = 1e-12         #j1 blows up at r=0
            xx = 2*np.pi/waves[:,None,None] * sim.tel_diameter/2 * np.sin(rr)
            area = np.pi*sim.tel_diameter**2/4
            I0 = area**2/waves[:,None,None]**2/sim.focuser.image_distance**2
            airy = I0 * (2.*j1(xx)/xx)**2

            #Check
            for i in range(len(waves)):
                assert(np.abs(airy[i] - image[i]).mean() < self.prop_tol)

        #Cleanup
        del pupil, image, airy, rr, xx, grid_pts, image_pts

############################################

    def test_lens_system(self):
        waves = np.array([0.641e-6, 0.725e-6])

        telD = 5e-3
        focal_length = 0.5

        lens_system = {
            'element_0': {'kind':'aperture',  \
                'distance':10e-3, 'diameter':telD},
            'element_1': {'kind':'lens', 'lens_name':'AC508-150-A-ML', \
                'distance':167.84e-3, 'diameter':telD},
            'element_2': {'kind':'lens', 'lens_name':'AC064-015-A-ML', \
                'distance':59.85e-3},
        }

        #Build simulator
        sim = diffraq.Simulator({'num_pts':512, 'waves':waves, 'tel_diameter':telD, \
            'image_size':74, 'focal_length':focal_length, 'lens_system':lens_system})
        sim.load_focuser()

        #Build uniform pupil image
        pupil = np.ones((len(waves), sim.num_pts, sim.num_pts)) + 0j
        grid_pts = diffraq.utils.image_util.get_grid_points(sim.num_pts, sim.tel_diameter)

        #Get images
        image, image_pts = sim.focuser.calculate_image(pupil[:,None], grid_pts)
        image = image[:,0]

        #Not airy disk, so compare peaks only
        area = np.pi*sim.tel_diameter**2/4
        for i in range(len(waves)):
            I0 = area**2/waves[i]**2/sim.focuser.image_distance**2
            assert(abs(I0 - image[i].max()) < 200)  #TODO: improve this

if __name__ == '__main__':

    tf = Test_Focuser()
    tf.run_all_tests()
