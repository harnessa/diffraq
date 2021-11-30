"""
test_phase_screen.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 1-29-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test of generating random phase screens.

"""

import diffraq
import numpy as np

class Test_Phase_Screen(object):

    def run_all_tests(self):
        for tt in ['uniform', 'kolmogorov'][:]:
            getattr(self, f'test_{tt}')()

############################################

    def test_uniform(self):
        radius = 25e-3
        tel_diam = 5e-3
        zz = 50
        wave = 0.6e-6
        dthick = 0.5e-6
        size_scale = 5e-3

        #Screens
        screens = {'kind':'thickness', 'thick_amplitude':dthick, 'thick_scale':size_scale}

        #Sim params
        params = {
            'zz':               zz,
            'waves':            wave,
            'radial_nodes':     120,
            'theta_nodes':      350,
            'tel_diameter':     tel_diam,
            'skip_image':       True,
            'beam_screens':     screens,
            'random_seed':      42,
        }

        #Build sim
        shape = {'kind':'circle', 'max_radius':radius}
        sim = diffraq.Simulator(params, shape)

        #Get coordinates
        sim.occulter.build_quadrature()
        xq, yq, wq = sim.occulter.xq, sim.occulter.yq, sim.occulter.wq

        # npts = 128
        # xq, yq = (np.indices((npts, npts))/npts - 1/2) * 2*radius
        # wq = np.ones(xq.size)/xq.size

        #Get random field
        fld = sim.beam.screens[0].get_field(xq.flatten(), yq.flatten(), wq, wave)

        import matplotlib.pyplot as plt;plt.ion()
        # plt.figure(); plt.colorbar(plt.imshow(np.angle(fld).reshape((npts,npts))))
        plt.figure(); plt.colorbar(plt.scatter(xq, yq, c=np.angle(fld), s=1))

        breakpoint()

############################################

    def test_kolmogorov(self):
        radius = 25e-3
        tel_diam = 5e-3
        zz = 50
        wave = 0.6e-6
        fried_r0 = 0.1

        #Screens
        screens = {'kind':'atmosphere', 'spectrum':'kolmogorov', 'fried_r0':fried_r0}

        #Sim params
        params = {
            'zz':               zz,
            'waves':            wave,
            'radial_nodes':     120,
            'theta_nodes':      350,
            'tel_diameter':     tel_diam,
            'skip_image':       True,
            'beam_screens':     screens,
            'random_seed':      42,
        }

        #Build sim
        shape = {'kind':'circle', 'max_radius':radius}
        sim = diffraq.Simulator(params, shape)

        #Get coordinates
        sim.occulter.build_quadrature()
        xq, yq, wq = sim.occulter.xq, sim.occulter.yq, sim.occulter.wq

        # npts = 256
        # xq, yq = (np.indices((npts, npts))/npts - 1/2) * 2*radius
        # xq = xq.flatten()
        # yq = yq.flatten()
        # wq = np.ones_like(yq)/yq.size


        #Get random field
        fld = sim.beam.screens[0].get_field(xq, yq, wq, wave)

        import matplotlib.pyplot as plt;plt.ion()

        plt.figure(); plt.colorbar(plt.scatter(xq, yq, c=np.angle(fld), s=1))

        breakpoint()

############################################

if __name__ == '__main__':

    tp = Test_Phase_Screen()
    tp.run_all_tests()
