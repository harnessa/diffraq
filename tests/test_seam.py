"""
test_seam.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 03-18-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test of building narrow seam around edges for non-scalar diffraction.

"""

import diffraq
import numpy as np

class Test_Seam(object):

    tol = 5e-4

    num_pts = 256
    radial_nodes = 400
    theta_nodes = 400
    zz = 1e5
    z0 = 1e19
    tel_diameter = 2.4

    seam_width = 1e-1
    circle_rad = 1

    def run_all_tests(self):
        funs = ['polar', 'cartesian', 'petal'][1:]
        tsts = ['quadrature', 'diffraction']
        for f in funs:
            for t in tsts:
                getattr(self, f'test_{f}_{t}')()

############################################
############################################

    def do_test_quadrature(self, edge_func, edge_diff):

        #Maxwell functions
        maxwell_func = [lambda d, w: np.heaviside(d, 1)+0j for i in range(2)]

        #Simulation parameters
        params = {'radial_nodes':100, 'theta_nodes':50, 'do_run_vector':True,\
            'seam_width':self.seam_width, 'maxwell_func':maxwell_func}

        #Loop over opacity
        for is_opaque in [False, True]:

            #Simulated shapes
            shapes = {'kind':'polar', 'edge_func':edge_func, \
                'edge_diff':edge_diff, 'is_opaque':is_opaque}

            #Should get same answer over multiple angles
            for ang in [0, 128]:

                #Polarization angle
                params['polarization_angle'] = ang

                #Build simulator
                sim = diffraq.Simulator(params, shapes)

                #Build polar seam
                xs, ys, ws, ds, ns = \
                    sim.vector.seams[0].build_seam_quadrature(self.seam_width)

                #Get area of open seam (in aperture)
                area = (ws * maxwell_func[0](ds,0).real).sum()

                #True area
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

                #Average field sould be 1/2 (seem is half on screen, half off)
                avg_fld = np.hypot(*sim.vector.vec_UU.real.mean((2,0)))

                assert(np.isclose(avg_fld, 0.5))

        #Cleanup
        sim.clean_up()
        del xs, ys, ws, ds, ns, sim

############################################

    def do_test_diffraction(self, edge_func, edge_diff):
        #Loop over opacity
        for is_opaque in [False, True]:

            #Maxwell functions (add opaque screen into aperture, or aperture into screen)
            val = [1, 0][int(is_opaque)]
            maxwell_func = [lambda d, w: np.heaviside(-d, val)+0j for i in range(2)]

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
            for ang in [0, 186]:

                #Simulated shapes
                shapes = {'kind':'polar', 'edge_func':edge_func, \
                    'edge_diff':edge_diff, 'is_opaque':is_opaque}

                #Polarization angle
                params['polarization_angle'] = ang

                #Build simulator
                sim = diffraq.Simulator(params, shapes)

                #Get pupil field from sim
                scl_pupil, grid_pts = sim.calc_pupil_field()

                #Build total field
                pupil = sim.vector.build_total_field(scl_pupil, sim.vec_pupil, sim.vec_comps)

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

                # import matplotlib.pyplot as plt;plt.ion()
                # plt.figure()
                # plt.plot(pupil)
                # plt.plot(utru, '--')
                # breakpoint()


        #Cleanup
        sim.clean_up()
        del pupil, grid_pts, scl_pupil, utru

############################################
############################################

    def test_polar_quad(self):
        polar_func = lambda t: self.circle_rad * np.ones_like(t)
        polar_diff = lambda t: np.zeros_like(t)

        self.do_test_quadrature(polar_func, polar_diff)

    def test_polar_diff(self):
        polar_func = lambda t: self.circle_rad * np.ones_like(t)
        polar_diff = lambda t: np.zeros_like(t)

        self.do_test_diffraction(polar_func, polar_diff)

    ############################################

    def _test_cart_quadrature(self):
        cart_func = lambda t: self.circle_rad * np.hstack((np.cos(t), np.sin(t)))
        cart_diff = lambda t: self.circle_rad * np.hstack((-np.sin(t), np.cos(t)))

        self.do_test_quadrature(cart_func, cart_diff)

    def _test_cart_diffraction(self):
        cart_func = lambda t: self.circle_rad * np.hstack((np.cos(t), np.sin(t)))
        cart_diff = lambda t: self.circle_rad * np.hstack((-np.sin(t), np.cos(t)))

        self.do_test_diffraction(cart_func, cart_diff)

############################################
############################################

if __name__ == '__main__':

    ts = Test_Seam()
    ts.run_all_tests()
