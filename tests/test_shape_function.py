"""
test_shape_function.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test of apodization function and its derivatives

"""

import numpy as np
from diffraq.occulter import Shape_Function

class Test_Shape_Function(object):

    radial_nodes = 400
    theta_nodes = 400
    circle_rad = 12
    tol = 1e-9
    tol2 = 4e-5

    def run_all_tests(self):
        for tt in ['polar', 'cartesian', 'starshade']:
            getattr(self, f'test_{tt}')()

############################################

    def do_test(self, func, deriv, tt):

        #Build points to interpolate
        points = np.hstack((tt, func(tt)))

        #Get different shape functions
        sf1 = Shape_Function(func,   deriv=deriv, deriv_2nd=None)
        sf2 = Shape_Function(func,   deriv=None,  deriv_2nd=None)
        sf3 = Shape_Function(points, deriv=None,  deriv_2nd=None)

        # breakpoint()
        print(np.abs(sf1.func(tt) - sf3.func(tt)).max(), \
            np.abs(sf1.deriv(tt) - sf3.deriv(tt)).max(), \
            np.abs(sf1.deriv_2nd(tt) - sf3.deriv_2nd(tt)).max())

        for val in ['func', 'deriv', 'deriv_2nd']:
            tol = [self.tol, self.tol2][val == 'deriv_2nd']
            assert( np.isclose(getattr(sf1, val)(tt), getattr(sf2, val)(tt)).all() & \
                   np.isclose(getattr(sf1, val)(tt), getattr(sf3, val)(tt), atol=tol).all() & \
                   np.isclose(getattr(sf2, val)(tt), getattr(sf3, val)(tt), atol=tol).all())

############################################

    def test_polar(self):
        func = lambda t: self.circle_rad * np.ones_like(t)
        deriv = lambda t: np.zeros_like(t)

        tt = np.linspace(0, 2*np.pi, self.radial_nodes * self.theta_nodes)[:,None]
        self.do_test(func, deriv, tt)

        #Cleanup
        del tt

############################################

    def test_cartesian(self):
        xfunc = lambda t:0.5*np.cos(t) + 0.5*np.cos(2*t)
        xderiv = lambda t: -0.5*np.sin(t) - np.sin(2*t)
        yfunc = lambda t: np.sin(t)
        yderiv = lambda t: np.cos(t)

        tt = np.linspace(0, 2*np.pi, self.radial_nodes * self.theta_nodes)[:,None]
        self.do_test(xfunc, xderiv, tt)
        self.do_test(yfunc, yderiv, tt)

        #Cleanup
        del tt

############################################

    def test_starshade(self):
        r0, r1 = 8, 15
        hga, hgb, hgn = 8,4, 6
        func = lambda r: np.exp(-((r - hga)/hgb)**hgn)
        deriv = lambda r: func(r) * (-hgn/hgb)*((r-hga)/hgb)**(hgn-1)

        tt = np.linspace(0, 2*np.pi, self.radial_nodes * self.theta_nodes)[:,None]
        rr = np.linspace(r0, r1, len(tt))[:,None]

        self.do_test(func, deriv, rr)

        #Cleanup
        del rr, tt

############################################

if __name__ == '__main__':

    ts = Test_Shape_Function()
    ts.run_all_tests()
