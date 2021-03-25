"""
test_vector.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 03-18-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test of vector diffraction via Braunbek method.

"""

import diffraq
import numpy as np

class Test_Vector(object):

    tol = 5e-4

    num_pts = 256
    radial_nodes = 400
    theta_nodes = 400
    zz = 1e5
    z0 = 1e19
    tel_diameter = 2.4

    seam_width = 0.1
    circle_rad = 1

    def run_all_tests(self):
        funs = ['polar', 'cartesian', 'petal']
        for f in funs:
            getattr(self, f'test_{f}')()

############################################
############################################

    def do_test_quadrature(self, kind, edge_func, edge_diff):

        #Maxwell functions
        maxwell_func = [lambda d, w: np.heaviside(d, 1)+0j for i in range(2)]

        #Simulation parameters
        params = {'radial_nodes':self.radial_nodes, 'theta_nodes':self.theta_nodes, \
            'do_run_vector':True, 'seam_width':self.seam_width, \
            'maxwell_func':maxwell_func, }

        #Loop over opacity
        for is_opaque in [False, True]:

            #Simulated shapes
            shapes = {'kind':kind, 'edge_func':edge_func, \
                'edge_diff':edge_diff, 'is_opaque':is_opaque}

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

    def do_test_diffraction(self, kind, edge_func, edge_diff):
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

    def test_polar(self):
        polar_func = lambda t: self.circle_rad * np.ones_like(t)
        polar_diff = lambda t: np.zeros_like(t)

        self.do_test_quadrature('polar', polar_func, polar_diff)
        self.do_test_diffraction('polar', polar_func, polar_diff)

    ############################################

    def test_cartesian(self):
        cart_func = lambda t: self.circle_rad * np.hstack((np.cos(t), np.sin(t)))
        cart_diff = lambda t: self.circle_rad * np.hstack((-np.sin(t), np.cos(t)))

        self.do_test_quadrature('cartesian', cart_func, cart_diff)
        self.do_test_diffraction('cartesian', cart_func, cart_diff)

############################################
############################################

    def test_petal(self):
        #Test HG
        r0, r1 = 12, 16
        hga, hgb, hgn = 8,5, 6
        num_pet = 12
        petal_func = lambda r: np.exp(-((r - hga)/hgb)**hgn)
        petal_diff = lambda r: petal_func(r) * (-hgn/hgb)*((r-hga)/hgb)**(hgn-1)

        etch = 0.005
        seam_width = 2.*abs(etch)             #run larger than etch b/c not normal to edge

        #Sim params
        params = {
            'radial_nodes':     self.radial_nodes//2,
            'theta_nodes':      self.theta_nodes//2,
            'num_pts':          self.num_pts,
            'tel_diameter':     self.tel_diameter,
            'zz':               self.zz,
            'z0':               self.z0,
            'skip_image':       True,
            'seam_width':       seam_width,
        }

        #Build etched shape
        etch_shape = {'kind':'petal', 'is_opaque':True, 'num_petals':num_pet, \
            'min_radius':r0, 'max_radius':r1, 'etch_error':etch}
        npts = self.radial_nodes * self.theta_nodes
        rr = np.linspace(r0, r1, npts)[:,None]
        etch_shape['edge_data'] = np.hstack((rr, petal_func(rr)))

        #Build etch simulation
        etch_sim = diffraq.Simulator(params, shapes=etch_shape)

        #Get etched starshade area
        etch_sim.occulter.build_quadrature()
        etch_area = etch_sim.occulter.wq.sum()

        #Build vector shape
        vec_shape = etch_shape
        vec_shape['etch_error'] = None

        #Build vector sim
        params['do_run_vector'] = True
        params['theta_nodes'] *= 2
        params['radial_nodes'] *= 2
        vec_sim = diffraq.Simulator(params, shapes=vec_shape)

        #Get vector starshade area
        vec_sim.occulter.build_quadrature()
        vec_area = vec_sim.occulter.wq.sum()

        #Build polar seam
        xs, ys, ws, ds, ns = \
            vec_sim.vector.seams[0].build_seam_quadrature(seam_width)

        #Build function that simulates overetch by truncating at certain distance normal to edge
        #Doesn't match etch_error exactly b/c seam is not normal to edge, but normal to radius
        area_func = lambda d: np.heaviside(-d, 1)*np.heaviside(etch+d,1)

        #Get area of open seam (in aperture)
        seam_area = (ws * area_func(ds)).sum()

        #Compare areas
        varea = vec_area + seam_area
        assert(np.abs(varea - etch_area) < 5e-3)

############################################
############################################

if __name__ == '__main__':

    tv = Test_Vector()
    # tv.run_all_tests()
    tv.test_petal()
