"""
test_vector_rectangle.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 03-18-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test of vector diffraction of rectangle via Braunbek method.

"""

import diffraq
import numpy as np

class Test_Vector_Rect(object):

    tol = 5e-4

    num_pts = 256
    radial_nodes = 40#0
    theta_nodes = 30#0
    zz = 1e5
    z0 = 1e19
    tel_diameter = 2.4

    seam_width = 0.1
    circle_rad = 1.5

    def run_all_tests(self):
        funs = ['quadrature', 'diffraction']
        for f in funs:
            getattr(self, f'test_{f}')()

############################################
############################################

    def test_quadrature(self):

        #Maxwell functions
        maxwell_func = lambda d, w: [np.heaviside(d, 1)+0j]*2

        #Simulation parameters
        params = {'radial_nodes':self.radial_nodes, 'theta_nodes':self.theta_nodes, \
            'do_run_vector':True, 'seam_width':self.seam_width, \
            'maxwell_func':maxwell_func, 'seam_radial_nodes':2*self.radial_nodes, 'seam_theta_nodes':self.theta_nodes*2}

        #Loop over aspect ratio
        for aspect in [1, 1/3, 1.3][-1:]:

            #Simulated shapes
            shape = {'kind':'rectangle', 'is_opaque':False, \
                'width':self.circle_rad, 'height':self.circle_rad*aspect}

            #Build simulator
            sim = diffraq.Simulator(params, shape)

            #Build polar seam
            xs, ys, ws, ds, ns, gw = \
                sim.vector.seams[0].build_seam_quadrature(self.seam_width)

            #Get area of open seam (in aperture)
            area = (ws * maxwell_func(ds,0)[0].real).sum()

            #True area
            # tru_area =
            import matplotlib.pyplot as plt;plt.ion()
            breakpoint()
            if is_opaque:
                tru_area = np.pi*((self.circle_rad + self.seam_width)**2 - \
                    self.circle_rad**2)
            else:
                tru_area = np.pi*(self.circle_rad**2 - \
                    (self.circle_rad - self.seam_width)**2)

            #Compare quadrature area
            assert(np.isclose(area, tru_area))

            #Build quadrature
            sim.vector.build_quadrature()

            #Get incident field
            sfld, pfld = sim.vector.screen.get_edge_field( \
                sim.vector.dq, sim.vector.gw, sim.waves[0])

            #Average field sould be 1/2 (seam is half on screen, half off)\
            assert(np.isclose(sfld.real.mean(), 0.5))
            assert(np.isclose(pfld.real.mean(), 0.5))

        #Cleanup
        sim.clean_up()
        del xs, ys, ws, ds, ns, sim

############################################

    def test_diffraction(self):
        #Loop over opacity
        for is_opaque in [False, True]:

            #Maxwell functions (add opaque screen into aperture, or aperture into screen)
            val = [1, 0][int(is_opaque)]
            maxwell_func = lambda d, w: [np.heaviside(-d, val)+0j]*2

            #Simulation parameters
            params = {
                'radial_nodes':     self.radial_nodes,
                'theta_nodes':      self.theta_nodes,
                'num_pts':          self.num_pts,
                'tel_diameter':     self.tel_diameter,
                'zz':               self.zz,
                'z0':               self.z0,
                'skip_image':       True,
                'do_run_vector':    True,
                'seam_width':       self.seam_width,
                'maxwell_func':     maxwell_func,
            }

            utru = None
            for ang in [0, 136]:

                #Simulated shapes
                shapes = {'kind':kind, 'edge_func':edge_func, \
                    'edge_diff':edge_diff, 'is_opaque':is_opaque}

                #Polarization angle
                params['polarization_angle'] = ang

                #Build simulator
                sim = diffraq.Simulator(params, shapes)

                #Get pupil field from sim
                scl_pupil, grid_pts = sim.calc_pupil_field()

                #Build total field
                pupil = sim.vector.build_polarized_field(scl_pupil, sim.vec_pupil, \
                    sim.vec_comps, 0)

                #Turn into unpolarized intensity (no analyzer)
                pupil = np.abs(pupil[0][0])**2 + np.abs(pupil[0][1])**2
                pupil = pupil[len(pupil)//2]

                #Calculate analytic solution (once)
                if utru is None:
                    crad = self.circle_rad + self.seam_width*[1, -1][int(is_opaque)]
                    utru = diffraq.utils.solution_util.calculate_circle_solution(grid_pts, \
                        sim.waves[0], sim.zz, sim.z0, crad, is_opaque)

                    #turn into intensity
                    utru = np.abs(utru)**2

                #Assert they match
                assert(np.abs(pupil - utru).max() < self.tol)

        #Cleanup
        sim.clean_up()
        del pupil, grid_pts, scl_pupil, utru

############################################
############################################

if __name__ == '__main__':

    tv = Test_Vector_Rect()
    tv.run_all_tests()
