"""
test_unique_petal.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 06-01-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test petals with individual apodization functions.

"""

import diffraq
import numpy as np

class Test_Unique_Petal(object):

    radial_nodes = 2000
    theta_nodes = 100
    num_pet = 12
    etch_error = 1e-6
    reg_edge = f'{diffraq.int_data_dir}/Test_Data/inv_apod_file.h5'

    def run_all(self):
        for tt in ['unique', 'test_etch_error', 'notch', 'all', 'vector'][-1:]:
            getattr(self, f'test_{tt}')()

    def get_params(self, w_vector):
        params = {
            'z0':                   27.5,
            'zz':                   50,
            'waves':                .641e-6,
            'tel_diameter':         3e-3,
            'skip_image':           True,
            'free_on_end':          False,
            'verbose':              False,
            'radial_nodes':         self.radial_nodes,
            'theta_nodes':          self.theta_nodes,
            'occulter_is_finite':   True,
        }

        if w_vector:
            new_params = {
                'seam_radial_nodes':    100,
                'seam_theta_nodes':     100,

                ### Vector ###
                'seam_width':           5e-6,
                'do_run_vector':        True,
                'is_sommerfeld':        True,
            }
        else:
            new_params = {
                'do_run_vector':        False,
            }

        params = {**params, **new_params}

        return params

############################################

    def run_test(self, reg_etch=None, unq_edge=None, unq_etch=None, w_pert=False, w_vector=False):
        #Get sim params
        params = self.get_params(w_vector)

        if w_pert:
            pert = {'xy0':[16.944e-3, 4.067e-3], 'height':40.2e-6, 'width':30.0e-6,
                'kind':'notch', 'direction':1, 'local_norm':False, 'num_quad':50}
        else:
            pert = []

        #Load regular sim
        reg_shape = {'kind':'starshade', 'is_opaque':False, 'num_petals':self.num_pet, \
            'edge_file':self.reg_edge, 'has_center':False, 'etch_error':reg_etch, \
            'perturbations':pert}
        reg_sim = diffraq.Simulator(params, reg_shape)
        reg_sim.run_sim()
        reg_pupil = reg_sim.pupil[0].copy()

        #Load unique sim
        unq_shape = {'kind':'uniquePetal', 'is_opaque':False, 'num_petals':self.num_pet, \
            'edge_file':self.reg_edge, 'has_center':False, 'perturbations':pert, \
            'unique_edges':unq_edge, 'etch_error':unq_etch}
        unq_sim = diffraq.Simulator(params, unq_shape)
        unq_sim.run_sim()
        unq_pupil = unq_sim.pupil[0].copy()

        #Make sure weights are the same
        assert(np.isclose(reg_sim.occulter.wq.sum(), unq_sim.occulter.wq.sum()))

        #Make sure pupils are the same
        assert(np.allclose(reg_pupil, unq_pupil))

        #Cleanup
        reg_sim.clean_up()
        unq_sim.clean_up()
        del reg_pupil, unq_pupil, reg_sim, unq_sim

############################################

    def test_unique(self):
        two = self.run_test(unq_edge={self.reg_edge:[10,11]})
        one = self.run_test(unq_edge={self.reg_edge:[10,11], self.reg_edge:[3,4]})

    def test_etch_error(self):
        return self.run_test(reg_etch=self.etch_error, unq_etch=np.ones(self.num_pet)*self.etch_error)

    def test_notch(self):
        return self.run_test(w_pert=True)

    def test_all(self):
        return self.run_test(unq_edge={self.reg_edge:[10,11]}, reg_etch=self.etch_error, \
            unq_etch=np.ones(self.num_pet)*self.etch_error, w_pert=True)

    # def test_vector(self):
    #     return self.run_test(unq_edge={self.reg_edge:[10,11]}, reg_etch=self.etch_error, \
    #         unq_etch=np.ones(self.num_pet)*self.etch_error, w_vector=True)

############################################

if __name__ == '__main__':

    td = Test_Unique_Petal()
    td.run_all()
