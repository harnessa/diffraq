"""
test_starshades.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-28-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test diffraction of analytic and numeric starshades

"""

import diffraq
import numpy as np

class Test_Starshades(object):

    def test_starshades(self):
        #Max contrast with default paramters
        max_con = 1e-8

        #Build simulator
        sim = diffraq.Simulator({'radial_nodes':100, 'theta_nodes':100, \
            'occulter_shape':'starshade', 'is_babinet':True})

        #Build target
        grid_pts = diffraq.utils.image_util.get_grid_points(128, sim.tel_diameter)

        #HG function and file
        ss_Afunc = lambda r: np.exp(-((r-sim.ss_rmin)/(sim.ss_rmax-sim.ss_rmin)/0.6)**6)
        ss_Afile = f'{diffraq.int_data_dir}/Test_Data/hg_apod_file.txt'
        ss_Lfile = f'{diffraq.int_data_dir}/Test_Data/hg_loci_file.txt'

        #Analytic vs numeric
        afunc_dict = {'analytic':ss_Afunc, 'numeric':None,      'loci':None}
        afile_dict = {'analytic':None,     'numeric':ss_Afile,  'loci':None}

        #Lambdaz
        lamz = sim.waves[0] * sim.zz

        #Test analytic and numeric
        UUs = []
        for ss in ['analytic', 'numeric', 'loci']:

            #Set apod values
            sim.apod_func = afunc_dict[ss]
            sim.apod_file = afile_dict[ss]

            #Loci run
            if ss == 'loci':
                sim.occulter_shape = 'loci'
                sim.loci_file = ss_Lfile

            #Build occulter
            sim.occulter.build_quadrature()

            #Calculate diffraction
            uu = diffraq.diffraction.diffract_grid(sim.occulter.xq, \
                sim.occulter.yq, sim.occulter.wq, lamz, grid_pts, sim.fft_tol, \
                is_babinet=True)

            #Store
            UUs.append(uu)

            #TODO: get better comparison
            #Check if below max contrast
            assert(np.abs(uu).max()**2 < max_con)

        #Check if results agree with each other (conservative)
        assert((np.abs(UUs[1] - UUs[0]).max() < max_con) & \
               (np.abs(UUs[2] - UUs[0]).max() < max_con) & \
               (np.abs(UUs[2] - UUs[1]).max() < max_con))

        #Cleanup
        sim.clean_up()
        del uu, UUs

if __name__ == '__main__':

    ts = Test_Starshades()
    ts.test_starshades()
