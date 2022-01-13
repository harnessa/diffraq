"""
test_etching_error.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-04-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test of etching error by applying error to circle and comparing
    to analytic results.

"""

import diffraq
import numpy as np

class Test_Etching(object):

    tol = 1e-6

    num_pts = 256
    radial_nodes = 40
    theta_nodes = 40
    zz = 15e6
    z0 = 1e19
    tel_diameter = 2.4
    circle_rad = 12
    etch = 1

    def run_all_tests(self):
        for oa in ['occulter', 'aperture']:
            for oe in ['over', 'under']:
                getattr(self, f'test_{oa}_{oe}')()

        self.test_start_petal()

############################################

    def run_calculation(self, is_opaque, etch_sign):
        #Load simulator
        params = {
            'radial_nodes':     self.radial_nodes,
            'theta_nodes':      self.theta_nodes,
            'num_pts':          self.num_pts,
            'tel_diameter':     self.tel_diameter,
            'zz':               self.zz,
            'z0':               self.z0,
            'skip_image':       True,
        }

        cart_func = lambda t: self.circle_rad * np.hstack((np.cos(t), np.sin(t)))
        cart_diff = lambda t: self.circle_rad * np.hstack((-np.sin(t), np.cos(t)))
        polar_func = lambda t: self.circle_rad * np.ones_like(t)
        polar_diff = lambda t: np.zeros_like(t)

        shape = {'is_opaque':is_opaque, 'max_radius':self.circle_rad}

        utru = None

        #Loop over occulter types
        for occ_shape in ['polar', 'cartesian']:
            #Loop over function vs interpolation (don't run function for now)
            for is_func in [False, True][:1]:

                #Only do function for cartesian
                if is_func and occ_shape != 'cartesian':
                    continue

                #Set parameters
                shape['kind'] = occ_shape

                if occ_shape == 'cartesian':
                    shape['edge_func'] = cart_func
                    shape['edge_diff'] = cart_diff
                    shape['etch_error'] = self.etch * etch_sign
                elif occ_shape == 'polar':
                    shape['edge_func'] = polar_func
                    shape['edge_diff'] = polar_diff
                    shape['etch_error'] = self.etch * etch_sign

                #Add data points if not function
                if is_func:
                    shape['edge_data'] = None
                else:
                    npts = self.radial_nodes * self.theta_nodes
                    tt = np.linspace(0, 2*np.pi, npts)[:,None]
                    shape['edge_data'] = np.hstack((tt, shape['edge_func'](tt)))

                #Load simulator
                sim = diffraq.Simulator(params, shapes=shape)

                #Build quad
                sim.occulter.build_quadrature()

                #Calculate area
                area = sim.occulter.wq.sum()

                #Get true area
                opq_sign = -(2*int(is_opaque) - 1)
                tru_area = np.pi*(self.circle_rad + self.etch*etch_sign*opq_sign)**2

                #Compare areas
                assert(np.isclose(area, tru_area))

                #Get pupil field from sim
                pupil, grid_pts = sim.calc_pupil_field()
                pupil = pupil[0][len(pupil[0])//2]

                #Calculate analytic solution (once)
                if utru is None:
                    utru = diffraq.utils.solution_util.calculate_circle_solution(grid_pts, \
                        sim.waves[0], sim.zz, sim.z0, \
                        self.circle_rad + self.etch*etch_sign*opq_sign, is_opaque)

                #Compare
                assert(np.abs(pupil - utru).max() < self.tol)

        #Clean up
        del pupil, grid_pts, sim, utru

############################################

    def test_occulter_over(self):
        self.run_calculation(True, 1)

    def test_occulter_under(self):
        self.run_calculation(True, -1)

    def test_aperture_over(self):
        self.run_calculation(False, 1)

    def test_aperture_under(self):
        self.run_calculation(False, -1)

    def test_star_petal(self):
        for opq in [False, True]:
            for es in [1,-1]:
                self.run_star_petal(opq, es)

############################################
############################################

    def run_star_petal(self, is_opaque, etch_sign):

        #Load simulator
        params = {
            'radial_nodes':     self.radial_nodes,
            'theta_nodes':      self.theta_nodes,
            'num_pts':          self.num_pts,
            'tel_diameter':     self.tel_diameter,
            'zz':               self.zz,
            'z0':               self.z0,
            'skip_image':       True,
        }
        num_pet = 12
        etch = 10e-6

        shape = {'kind':'starshade', 'num_petals':num_pet, 'is_opaque':is_opaque,\
            'has_center':True, 'etch_error':etch * etch_sign,
            'edge_file':f'{diffraq.int_data_dir}/Test_Data/star_apod_file.h5'}

        #Load simulator
        sim = diffraq.Simulator(params, shapes=shape)

        #Build quad
        sim.occulter.build_quadrature()

        #Calculate area
        area = sim.occulter.wq.sum()

        #True area
        x0 = sim.occulter.shapes[0].min_radius * np.cos(np.pi/num_pet)
        x1 = sim.occulter.shapes[0].max_radius
        y0 = x0 * np.tan(np.pi/num_pet)
        ply_area = num_pet * (2*y0)**2 / (4*np.tan(np.pi/num_pet))
        tri_area = 2 * 0.5*y0*(x1-x0)
        ech_area = 2*num_pet * etch * np.hypot(x1-x0, 0-y0)
        ech_area *= -1 * etch_sign * [1,-1][int(is_opaque)]
        tru_area = num_pet*tri_area + ply_area + ech_area

        #Assert areas are close
        assert(np.isclose(tru_area, sim.occulter.wq.sum()))

        #Cleanup
        sim.clean_up()

############################################
############################################

    def run_petal_calculation(self):
        #Load simulator
        params = {
            'radial_nodes':     self.radial_nodes,
            'theta_nodes':      self.theta_nodes,
            'num_pts':          self.num_pts,
            'tel_diameter':     self.tel_diameter,
            'zz':               self.zz,
            'z0':               self.z0,
            'skip_image':       True,
        }

        etch = 0.01

        r0, r1 = 10, 14
        hga, hgb, hgn = 8,5, 6
        num_pet = 12
        petal_func = lambda r: np.exp(-((r - hga)/hgb)**hgn)
        petal_diff = lambda r: petal_func(r) * (-hgn/hgb)*((r-hga)/hgb)**(hgn-1)

        is_opaque = True
        opq_sign = -(2*int(is_opaque) - 1)

        #Don't test, just visually inspect
        for etch_sign in [-1, 1]:

            shape = {'kind':'petal', 'is_opaque':is_opaque, 'num_petals':num_pet, \
                'min_radius':r0, 'max_radius':r1, 'etch_error':etch*etch_sign}

            #Radii
            npts = self.radial_nodes
            rr = np.linspace(r0, r1, npts)[:,None]

            #Build edge from scratch
            trut = petal_func(rr)*np.pi/num_pet
            truxy = rr*np.hstack((np.cos(trut), np.sin(trut)))

            #build normals
            normal = (np.roll(truxy, 1, axis=0) - truxy)[:,::-1] * np.array([-1,1])
            normal /= np.hypot(*normal.T)[:,None]
            normal[0] = normal[1]

            old = truxy.copy()

            #Add etch
            truxy += normal * etch*etch_sign*opq_sign

            #Build full mask
            mask = np.empty((0,2))
            for i in range(num_pet):
                ang = 2.*np.pi/num_pet * i
                rot_mat = lambda a: np.array(([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]))
                mask = np.concatenate((mask, (truxy * np.array([1,-1])).dot(rot_mat(ang).T)))
                mask = np.concatenate((mask, truxy.dot(rot_mat(ang),T)[::-1]))

            shape['edge_data'] = np.hstack((rr, petal_func(rr)))

            #Load simulator
            sim = diffraq.Simulator(params, shapes=shape)

            #Build edge
            sim.occulter.build_edge()

            #cycle mask around
            mask = np.roll(mask, -np.argmin(np.hypot(*(mask - sim.occulter.edge[0]).T)), axis=0)

            #Compare edges
            diff = np.hypot(*(mask - sim.occulter.edge).T)

            #FIXME: fix end points

            import matplotlib.pyplot as plt;plt.ion()
            plt.cla()
            plt.plot(*sim.occulter.edge.T)
            plt.plot(*mask.T, '--')
            # plt.plot(*old.T, '-.')
            plt.xlim([9,15])
            plt.ylim([0,3])
            breakpoint()

        #cleanup
        sim.clean_up()
        del mask, old, sim

############################################
############################################

if __name__ == '__main__':

    te = Test_Etching()
    # te.run_all_tests()
    # te.run_petal_calculation()
    te.test_star_petal()
