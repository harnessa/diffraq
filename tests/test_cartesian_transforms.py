"""
test_cartesian_transforms.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test of cartesian coordinate transformations for various types of shapes.

"""

import numpy as np
import diffraq.geometry

class Test_Cart_Transforms(object):

    radial_nodes = 400
    theta_nodes = 400
    circle_rad = 12
    tol = 2e-8
    tol2 = 5e-5

    def run_all_tests(self):
        for tt in ['polar_cart', 'petal_cart', 'closest_point']:
            getattr(self, f'test_{tt}')()

############################################
############################################

    def test_polar_cart(self):

        #Other function
        pole_func = lambda t: 2.*np.cos(3*t)
        pole_diff = lambda t: -6*np.sin(3*t)

        cart_func = lambda t: 2.*np.cos(3*t) * np.hstack(( np.cos(t), np.sin(t)))
        cart_diff = lambda t: np.hstack((-6*np.sin(3*t)*np.cos(t) - 2*np.cos(3*t)*np.sin(t), \
            -6*np.sin(3*t)*np.sin(t) + 2*np.cos(3*t)*np.cos(t)))

        #Build sim
        sim = diffraq.Simulator()

        #Loop over analytic or numerical differentiation
        for w_diff in [False, True]:

            if w_diff:
                d1 = pole_diff
                d2 = cart_diff
            else:
                d1, d2 = None, None

            #Build shapes
            sf1 = diffraq.geometry.PolarShape(sim.occulter, \
                **{'edge_func':pole_func, 'edge_diff':d1})
            sf2 = diffraq.geometry.CartesianShape(sim.occulter, \
                **{'edge_func':cart_func, 'edge_diff':d2})

            #Check cartesian function
            tt = np.linspace(0, 2*np.pi, self.radial_nodes * self.theta_nodes)[:,None]

            for val in ['func', 'diff', 'diff_2nd']:
                tol = [self.tol, self.tol2][val == 'diff_2nd']
                val1 = getattr(sf1, f'cart_{val}')(tt)
                val2 = getattr(sf2, f'cart_{val}')(tt)

                assert(np.allclose(val1, val2, atol=tol))

        #Cleanup
        del tt, sim, sf1, sf2, val1, val2

############################################

    def test_petal_cart(self):
        r0, r1 = 8, 15
        hga, hgb, hgn = 8,4, 6
        num_pet = 12
        petal_func = lambda r: np.exp(-((r - hga)/hgb)**hgn)
        petal_diff = lambda r: petal_func(r) * (-hgn/hgb)*((r-hga)/hgb)**(hgn-1)

        petal_func_p = lambda r: petal_func(r) * np.pi/num_pet
        petal_diff_p = lambda r: petal_diff(r) * np.pi/num_pet
        cart_func = lambda r: r * np.hstack((np.cos(petal_func_p(r)), np.sin(petal_func_p(r))))
        cart_diff = lambda r: np.hstack(( \
            np.cos(petal_func_p(r)) - r*np.sin(petal_func_p(r))*petal_diff_p(r),
            np.sin(petal_func_p(r)) + r*np.cos(petal_func_p(r))*petal_diff_p(r) ))

        #Build sim
        sim = diffraq.Simulator()

        #Loop over analytic or numerical differentiation
        for w_diff in [False, True]:

            if w_diff:
                d1 = petal_diff
                d2 = cart_diff
            else:
                d1, d2 = None, None

            #Build shapes
            sf1 = diffraq.geometry.PetalShape(sim.occulter, \
                **{'edge_func':petal_func, 'edge_diff':d1, 'num_petals':num_pet})
            sf2 = diffraq.geometry.CartesianShape(sim.occulter, \
                **{'edge_func':cart_func, 'edge_diff':d2})

            #Check cartesian function
            rr = np.linspace(r0, r1, self.radial_nodes * self.theta_nodes)[:,None]

            for val in ['func', 'diff', 'diff_2nd']:
                tol = [self.tol, self.tol2][val == 'diff_2nd']
                val1 = getattr(sf1, f'cart_{val}')(rr)
                val2 = getattr(sf2, f'cart_{val}')(rr)

                assert(np.allclose(val1, val2, atol=tol))

        #Cleanup
        del rr, sim, sf1, sf2, val1, val2

############################################

############################################

    def test_closest_point(self):

        #Other function
        pole_func = lambda t: 2.*np.cos(3*t)
        pole_diff = lambda t: -6*np.sin(3*t)

        cart_pfunc = lambda t: 2.*np.cos(3*t) * np.hstack(( np.cos(t), np.sin(t)))
        cart_pdiff = lambda t: np.hstack((-6*np.sin(3*t)*np.cos(t) - 2*np.cos(3*t)*np.sin(t), \
            -6*np.sin(3*t)*np.sin(t) + 2*np.cos(3*t)*np.cos(t)))

        r0, r1 = 8, 15
        hga, hgb, hgn = 8,4, 6
        num_pet = 12
        petal_func = lambda r: np.exp(-((r - hga)/hgb)**hgn)
        petal_diff = lambda r: petal_func(r) * (-hgn/hgb)*((r-hga)/hgb)**(hgn-1)

        petal_func_p = lambda r: petal_func(r) * np.pi/num_pet
        petal_diff_p = lambda r: petal_diff(r) * np.pi/num_pet
        cart_rfunc = lambda r: r * np.hstack((np.cos(petal_func_p(r)), np.sin(petal_func_p(r))))
        cart_rdiff = lambda r: np.hstack(( \
            np.cos(petal_func_p(r)) - r*np.sin(petal_func_p(r))*petal_diff_p(r),
            np.sin(petal_func_p(r)) + r*np.cos(petal_func_p(r))*petal_diff_p(r) ))

        #Build sim
        sim = diffraq.Simulator()

        #Loop over analytic or numerical differentiation
        for kind in ['polar', 'petal']:
            for w_diff in [False, True]:

                if kind == 'polar':
                    if w_diff:
                        d1 = pole_diff
                        d2 = cart_pdiff
                    else:
                        d1, d2 = None, None
                    sf1 = diffraq.geometry.PolarShape(sim.occulter, \
                        **{'edge_func':pole_func, 'edge_diff':d1})
                    sf2 = diffraq.geometry.CartesianShape(sim.occulter, \
                        **{'edge_func':cart_pfunc, 'edge_diff':d2})
                    point = np.array([1.399, 0.363])

                else:
                    if w_diff:
                        d1 = petal_diff
                        d2 = cart_rdiff
                    else:
                        d1, d2 = None, None
                    sf1 = diffraq.geometry.PetalShape(sim.occulter, \
                        **{'edge_func':petal_func, 'edge_diff':d1, 'num_petals':num_pet})
                    sf2 = diffraq.geometry.CartesianShape(sim.occulter, \
                        **{'edge_func':cart_rfunc, 'edge_diff':d2})
                    point = np.array([11.97, 1.117])

                #Get closest point
                c1 = sf1.cart_func(sf1.find_closest_point(point))
                c2 = sf2.cart_func(sf2.find_closest_point(point))

                #Make sure it matches
                assert((np.hypot(*(c1 - point)) < 1e-3) & (np.hypot(*(c2 - point)) < 1e-3))

        #Cleanup
        del sim, sf1, sf2, c1, c2

############################################

if __name__ == '__main__':

    ts = Test_Cart_Transforms()
    ts.run_all_tests()
