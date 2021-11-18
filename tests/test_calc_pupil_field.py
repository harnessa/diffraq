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

        ngrid = 20
        grid_width = 3
        fresnums = [1,10,50]
        shapes = ['circle', 'polar']
        wave = 0.5e-6
        tol = 5e-3              #tolerance

        #Simulator params
        params = {
            'waves':            wave,
            'radial_nodes':     250,
            'theta_nodes':      600,
            'tel_diameter':     grid_width,
            'num_pts':          ngrid,
            'skip_image':       True,
        }

        shapes = [{'kind':'circle', 'max_radius':1},
            {'kind':'polar', 'edge_func':lambda t: 1 + 0.3*np.cos(3*t)}]

        for shape in shapes[:1]:

            #Compare to theoretical value for each Fresnel number
            for i in range(len(fresnums))[-1:]:

                params['zz'] = 1/(fresnums[i]*wave)

                #Build simulator
                sim = diffraq.Simulator(params, shape)


                import time
                tik = time.perf_counter()

                #Get pupil field
                pupil, grid_pts = sim.calc_pupil_field()

                tok = time.perf_counter()
                print(f'time: {tok-tik:.2f}')

                #Get 2D grid for theoretical calculation
                grid_2D = np.tile(grid_pts, (ngrid,1))

                #Theoretical value
                utru = diffraq.utils.solution_util.direct_integration( \
                    fresnums[i], pupil[0].shape, sim.occulter.xq, sim.occulter.yq, \
                    sim.occulter.wq, grid_2D) * np.exp(1j*2*np.pi/sim.waves[0]*sim.z0)

                #Assert max difference is close to specified tolerance
                max_diff = tol

                # assert(np.abs(utru - pupil[0]).max() < max_diff)
                print(np.abs(utru - pupil[0]).max())
                # import matplotlib.pyplot as plt;plt.ion()
                # plt.imshow(abs(pupil[0]))
                # breakpoint()

        #Cleanup
        sim.clean_up()
        del grid_pts, grid_2D, pupil, utru

if __name__ == '__main__':

    tst = Test_calc_pupil_field()
    tst.test_pupil_field()
