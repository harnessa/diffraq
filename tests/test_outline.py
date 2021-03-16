"""
test_outline.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test of outline function and its derivatives

"""

import numpy as np
import diffraq

class Test_Outline(object):

    radial_nodes = 400
    theta_nodes = 400
    circle_rad = 12
    tol = 2e-8
    tol2 = 5e-5

    def run_all_tests(self):
        for tt in ['polar', 'petal', 'cartesian']:
            getattr(self, f'test_{tt}')()

############################################

    def do_outline_test(self, kind, func, diff, tt):

        #Build points to interpolate
        points = np.hstack((tt, func(tt)))

        #Get different outlines
        sf1 = diffraq.geometry.LambdaOutline(func, diff=diff)
        sf2 = diffraq.geometry.LambdaOutline(func, diff=None)
        if kind == 'cartesian':
            sf3 = diffraq.geometry.Cart_InterpOutline(points, with_2nd=True)
        else:
            sf3 = diffraq.geometry.InterpOutline(points, with_2nd=True)

        #Check all values are the same
        for val in ['func', 'diff', 'diff_2nd']:
            tol = [self.tol, self.tol2][val == 'diff_2nd']
            val1 = getattr(sf1, val)(tt)
            val2 = getattr(sf2, val)(tt)
            val3 = getattr(sf3, val)(tt)

            assert(np.allclose(val1, val2) & np.allclose(val1, val3, atol=tol))

        #Clean up
        del points, val1, val2, val3, sf1, sf2, sf3

    def do_shape_test(self, kind, func, diff, tt):

        #Build points to interpolate
        points = np.hstack((tt, func(tt)))

        #Build sim
        sim = diffraq.Simulator()

        #Get shape
        shape = getattr(diffraq.geometry, f'{kind.capitalize()}Shape')

        #Build various shapes
        sf1 = shape(sim.occulter, **{'edge_func':func, 'edge_diff':diff})
        sf2 = shape(sim.occulter, **{'edge_func':func, 'edge_diff':None})
        sf3 = shape(sim.occulter, **{'edge_data':points})

        #Check all values are the same (dont run 2nd derivative for this one - don't specify directly to shape)
        for val in ['func', 'diff']:
            val1 = getattr(sf1.outline, val)(tt)
            val2 = getattr(sf2.outline, val)(tt)
            val3 = getattr(sf3.outline, val)(tt)

            assert(np.allclose(val1, val2) & np.allclose(val1, val3, atol=self.tol))

        #Clean up
        del points, val1, val2, val3, sf1, sf2, sf3, sim

############################################
############################################

############################################

    def test_polar(self):
        func = lambda t: self.circle_rad * np.ones_like(t)
        diff = lambda t: np.zeros_like(t)

        tt = np.linspace(0, 2*np.pi, self.radial_nodes * self.theta_nodes)[:,None]
        self.do_outline_test('polar', func, diff, tt)
        self.do_shape_test(  'polar', func, diff, tt)

        #Cleanup
        del tt

############################################

    def test_cartesian(self):
        func = lambda t: np.hstack((0.5*np.cos(t) + 0.5*np.cos(2*t), np.sin(t)))
        diff = lambda t: np.hstack((-0.5*np.sin(t) - np.sin(2*t), np.cos(t)))

        tt = np.linspace(0, 2*np.pi, self.radial_nodes * self.theta_nodes)[:,None]
        self.do_outline_test('cartesian', func, diff, tt)
        self.do_shape_test(  'cartesian', func, diff, tt)

        #Cleanup
        del tt

############################################

    def test_petal(self):
        r0, r1 = 8, 15
        hga, hgb, hgn = 8,4, 6
        func = lambda r: np.exp(-((r - hga)/hgb)**hgn)
        diff = lambda r: func(r) * (-hgn/hgb)*((r-hga)/hgb)**(hgn-1)

        rr = np.linspace(r0, r1, self.radial_nodes * self.theta_nodes)[:,None]

        self.do_outline_test('petal', func, diff, rr)
        self.do_shape_test(  'petal', func, diff, rr)

        #Cleanup
        del rr

############################################
############################################

if __name__ == '__main__':

    ts = Test_Outline()
    ts.run_all_tests()
