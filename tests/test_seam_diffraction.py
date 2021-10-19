"""
test_seam_diffraction.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 10-19-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test diffraction of Braunbek seams by comparing to annulus

"""

import diffraq
import numpy as np

class Test_Seam_Diffraction(object):

    tol = 2e-8

    num_pts = 256
    radial_nodes = 400
    theta_nodes = 400   
    zz = 15e6
    z0 = 1e19
    rad_inn = 10
    rad_out = 12

    def run_all_tests(self):
        for oa in ['occulter', 'aperture'][1:]:
            for op in ['plane', 'spherical'][:]:
                getattr(self, f'test_{oa}_{op}')()

    def run_calculation(self, is_aperture, z0):

        #Load simulator
        params = {
            'radial_nodes':         self.radial_nodes,
            'theta_nodes':          self.theta_nodes,
            'num_pts':              self.num_pts,
            'tel_diameter':         2.4,
            'zz':                   self.zz,
            'z0':                   z0,
            'skip_image':           True,

            ### Vector ###
            'seam_radial_nodes':    self.radial_nodes,
            'seam_theta_nodes':     self.theta_nodes,
            'seam_width':           (self.rad_out - self.rad_inn)/2,
            'do_run_vector':        True,
            'maxwell_func':         (lambda d,w: [np.ones_like(d)+0j]*2),

        }

        #Shape
        shape = {'kind':'circle', 'max_radius':(self.rad_inn + self.rad_out)/2, \
            'is_opaque':is_aperture}

        #Set finite
        params['occulter_is_finite'] = is_aperture

        #Load simulator
        sim = diffraq.Simulator(params, shape)

        #Get pupil field from sim
        grid_pts = diffraq.utils.image_util.get_grid_points(sim.num_pts, sim.tel_diameter)
        vec_pupil = sim.vector_diffraction_calculation(grid_pts, is_babinet=not is_aperture)
        pupil = vec_pupil[0,0][vec_pupil.shape[-1]//2]

        #Calculate analytic solution
        utru_inn = diffraq.utils.solution_util.calculate_circle_solution(grid_pts, \
            sim.waves[0], sim.zz, sim.z0, self.rad_inn, False)

        utru_out = diffraq.utils.solution_util.calculate_circle_solution(grid_pts, \
            sim.waves[0], sim.zz, sim.z0, self.rad_out, False)

        utru = utru_out - utru_inn
        if not is_aperture:
            zscl = sim.z0 / (sim.zz + sim.z0)
            u0 = np.exp(1j*2*np.pi/sim.waves[0]*grid_pts**2./(2*(sim.zz + sim.z0))) * zscl
            u0 *= np.exp(1j*2.*np.pi/sim.waves[0] * sim.zz)
            utru = u0 - utru

        #Compare
        assert(np.abs(pupil - utru).max() < self.tol)

        #Clean up
        del pupil, grid_pts, sim, utru_inn, utru_out, utru

############################################

    def test_occulter_plane(self):
        self.run_calculation(False, self.z0)

    def test_occulter_spherical(self):
        self.run_calculation(False, self.zz)

    def test_aperture_plane(self):
        self.run_calculation(True, self.z0)

    def test_aperture_spherical(self):
        self.run_calculation(True, self.zz)


############################################

if __name__ == '__main__':

    ts = Test_Seam_Diffraction()
    ts.run_all_tests()
