"""
test_occulter_file.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-16-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test building of occulter from occulter file

"""

import diffraq
import numpy as np
import copy

class Test_Occulter_File(object):

    def run_all_tests(self):
        for tt in ['diffraction']:
            getattr(self, f'test_{tt}')()

############################################

    radial_nodes = 2000
    theta_nodes = 100
    num_pet = 12
    apod_dir = f'{diffraq.int_data_dir}/Test_Data'

    def get_sims(self, radial_nodes, theta_nodes):
        params = {
            'z0':               27.5,
            'zz':               50,
            'waves':            .641e-6,
            'tel_diameter':     3e-3,
            'skip_image':       True,
            'free_on_end':      False,
            'verbose':          False,
            'radial_nodes':     radial_nodes,
            'theta_nodes':      theta_nodes,
            'is_finite':        True,
        }

        #Load regular sim
        shape = {'kind':'starshade', 'is_opaque':False, 'num_petals':self.num_pet, \
            'edge_file':f'{self.apod_dir}/inv_apod_file.txt', 'has_center':False}
        reg_sim = diffraq.Simulator(params, shape)

        #Load occulter sim
        params['radial_nodes'] = params['radial_nodes'] //2
        params['theta_nodes'] = params['theta_nodes'] //2
        params['occulter_file'] = f'{self.apod_dir}/inv_apod_occulter.cfg'
        occ_sim = diffraq.Simulator(params)

        return reg_sim, occ_sim

############################################

    def test_diffraction(self):
        #Get sims
        reg_sim, occ_sim = self.get_sims(self.radial_nodes, self.theta_nodes)

        #Run sims
        occ_sim.run_sim()
        occ_img = np.abs(occ_sim.pupil[0])**2
        occ_area = occ_sim.occulter.wq.sum()
        occ_sim.clean_up()

        reg_sim.run_sim()
        reg_img = np.abs(reg_sim.pupil[0])**2
        reg_area = reg_sim.occulter.wq.sum()
        reg_sim.clean_up()

        #Compare areas
        assert(np.abs(reg_area - occ_area)/occ_area < 1e-5)

        #Compare images
        assert(np.abs(reg_img.max() - occ_img.max()) < 1e-10)

        #Cleanup
        del occ_sim, reg_sim, occ_img, reg_img

############################################

if __name__ == '__main__':

    to = Test_Occulter_File()
    to.run_all_tests()
