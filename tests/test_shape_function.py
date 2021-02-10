"""
test_shape_function.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test of apodization function and its diffatives

"""

import numpy as np
from diffraq.geometry import Cartesian_Shape_Func, Polar_Shape_Func, Radial_Shape_Func

class Test_Shape_Function(object):

    radial_nodes = 400
    theta_nodes = 400
    circle_rad = 12
    tol = 2e-8
    tol2 = 4e-5

    def run_all_tests(self):
        for tt in ['polar', 'starshade', 'cartesian', 'polar_cart', 'radial_cart']:
            getattr(self, f'test_{tt}')()

############################################

    def do_test(self, kind, func, diff, tt):

        #Build points to interpolate
        points = np.hstack((tt, func(tt)))

        #Get appropriate shape function
        if kind == 'cart':
            Shape_Func = Cartesian_Shape_Func
        else:
            Shape_Func = Polar_Shape_Func

        #Get different shape functions
        sf1 = Shape_Func(func,   diff=diff, diff_2nd=None)
        sf2 = Shape_Func(func,   diff=None, diff_2nd=None)
        sf3 = Shape_Func(points, diff=None, diff_2nd=None)

        for val in ['func', 'diff', 'diff_2nd']:
            tol = [self.tol, self.tol2][val == 'diff_2nd']
            assert(np.isclose(getattr(sf1, val)(tt), getattr(sf2, val)(tt)).all() & \
                   np.isclose(getattr(sf1, val)(tt), getattr(sf3, val)(tt), atol=tol).all() & \
                   np.isclose(getattr(sf2, val)(tt), getattr(sf3, val)(tt), atol=tol).all())

############################################

    def test_polar_cart(self):

        #Other function
        pole_func = lambda t: 2.*np.cos(3*t)
        pole_diff = lambda t: -6*np.sin(3*t)

        cart_func = lambda t: 2.*np.cos(3*t) * np.hstack(( np.cos(t), np.sin(t)))
        cart_diff = lambda t: np.hstack((-6*np.sin(3*t)*np.cos(t) - 2*np.cos(3*t)*np.sin(t), \
            -6*np.sin(3*t)*np.sin(t) + 2*np.cos(3*t)*np.cos(t)))

        #Loop over analytic or numerical differentiation
        for w_diff in [False, True]:

            if w_diff:
                d1 = pole_diff
                d2 = cart_diff
            else:
                d1, d2 = None, None

            sf1 = Polar_Shape_Func(pole_func,     diff=d1, diff_2nd=None)
            sf2 = Cartesian_Shape_Func(cart_func, diff=d2, diff_2nd=None)

            #Check cartesian function
            tt = np.linspace(0, 2*np.pi, self.radial_nodes * self.theta_nodes)[:,None]

            for val in ['func', 'diff', 'diff_2nd']:
                tol = [self.tol, self.tol2][val == 'diff_2nd']
                assert( np.isclose(getattr(sf1, f'cart_{val}')(tt), \
                    getattr(sf2, f'cart_{val}')(tt), atol=tol).all())

        #Cleanup
        del tt

############################################

    def test_radial_cart(self):
        r0, r1 = 8, 15
        hga, hgb, hgn = 8,4, 6
        radi_func = lambda r: np.exp(-((r - hga)/hgb)**hgn) * np.pi/12
        radi_diff = lambda r: radi_func(r) * (-hgn/hgb)*((r-hga)/hgb)**(hgn-1)

        cart_func = lambda r: r * np.hstack((np.cos(radi_func(r)), np.sin(radi_func(r))))
        cart_diff = lambda r: np.hstack(( \
            np.cos(radi_func(r)) - r*np.sin(radi_func(r))*radi_diff(r),
            np.sin(radi_func(r)) + r*np.cos(radi_func(r))*radi_diff(r) ))

        #Loop over analytic or numerical differentiation
        for w_diff in [False, True]:

            if w_diff:
                d1 = radi_diff
                d2 = cart_diff
            else:
                d1, d2 = None, None

            sf1 = Radial_Shape_Func(radi_func,    diff=d1, diff_2nd=None)
            sf2 = Cartesian_Shape_Func(cart_func, diff=d2, diff_2nd=None)

            #Check cartesian function
            rr = np.linspace(r0, r1, self.radial_nodes * self.theta_nodes)[:,None]

            for val in ['func', 'diff', 'diff_2nd']:
                tol = [self.tol, self.tol2][val == 'diff_2nd']
                assert( np.isclose(getattr(sf1, f'cart_{val}')(rr), \
                    getattr(sf2, f'cart_{val}')(rr), atol=tol).all())

        #Cleanup
        del rr

############################################

    def test_polar(self):
        func = lambda t: self.circle_rad * np.ones_like(t)
        diff = lambda t: np.zeros_like(t)

        tt = np.linspace(0, 2*np.pi, self.radial_nodes * self.theta_nodes)[:,None]
        self.do_test('polar', func, diff, tt)

        #Cleanup
        del tt

############################################

    def test_cartesian(self):
        func = lambda t: np.hstack((0.5*np.cos(t) + 0.5*np.cos(2*t), np.sin(t)))
        diff = lambda t: np.hstack((-0.5*np.sin(t) - np.sin(2*t), np.cos(t)))

        tt = np.linspace(0, 2*np.pi, self.radial_nodes * self.theta_nodes)[:,None]
        self.do_test('cart', func, diff, tt)

        #Cleanup
        del tt

############################################

    def test_starshade(self):
        r0, r1 = 8, 15
        hga, hgb, hgn = 8,4, 6
        func = lambda r: np.exp(-((r - hga)/hgb)**hgn)
        diff = lambda r: func(r) * (-hgn/hgb)*((r-hga)/hgb)**(hgn-1)

        rr = np.linspace(r0, r1, self.radial_nodes * self.theta_nodes)[:,None]

        self.do_test('apod', func, diff, rr)

        #Cleanup
        del rr

############################################

if __name__ == '__main__':

    ts = Test_Shape_Function()
    ts.run_all_tests()
