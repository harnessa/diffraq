"""
test_lgwt.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test of LGWT algorithm.

"""

import diffraq
import numpy as np

class Test_Quadrature(object):

    def run_all_tests(self):
        tsts = ['lgwt', 'polar', 'starshade']
        for t in tsts:
            getattr(self, f'test_{t}')()

############################################

    def test_lgwt(self):

        #Test points
        limits = [[-1,1], [-1,0], [0,1], [5,10]]
        nums = [5, 10, 100, 1001, 5000]

        for (a,b) in limits:
            for N in nums:

                #Truth
                p1, w2 = np.polynomial.legendre.leggauss(N)
                p1 = p1[::-1]
                w2 = w2[::-1]
                p2 = (a*(1-p1) + b*(1+p1))/2    #Linear map from [-1,1] to [a,b]
                w2 *= (b-a)/2                   #Normalize the weights

                #Diffraq answer
                pq, wq = diffraq.quad.lgwt(N, a, b)

                assert((np.isclose(pq, p2).all()) & (np.isclose(wq, w2).all()))

############################################

    def test_polar(self):
        for m in range(10,100,10):
            for n in range(10,100,10):
                for a in np.arange(0.1, 1, 0.1):
                    #Smooth radial function on [0, 2pi)
                    gfunc = lambda t: 1 + a*np.cos(3*t)

                    #Get quad
                    xq, yq, wq = diffraq.quad.polar_quad(gfunc, m, n)

                    #Assert with analytic area formula
                    assert(np.isclose(wq.sum(), np.pi*(1. + a**2/2)))

############################################

    def test_starshade(self):

        #Starshade
        num_pet = 16
        ss_r0 = 7; ss_r1 = 14
        ss_Afunc = lambda r: np.exp(-((r-ss_r0)/(ss_r1-ss_r0)/0.6)**6)
        hg_area = 167.74245316719933         #Calculated once, not sure if correct

        #Disk
        disk_r0 = 0.7; disk_r1 = 1.3
        disk_Afunc = lambda t: 1 + 0*t

        for m in range(10,100,10):
            for n in range(10,1000,100):

                ### Test DISK ###
                #Get quad
                xq, yq, wq = diffraq.quad.starshade_quad(1, disk_Afunc, \
                    disk_r0, disk_r1, m, n)

                #Assert with analytic area formula
                assert(np.isclose(wq.sum(), np.pi*disk_r1**2))

                ### Test STARSHADE ###
                xq, yq, wq = diffraq.quad.starshade_quad(num_pet, ss_Afunc, \
                    ss_r0, ss_r1, m, n)

                #TODO: need better starshade assertion
                assert(np.isclose(wq.sum(), hg_area))

        # breakpoint()

############################################

if __name__ == '__main__':

    tq = Test_Quadrature()
    # tq.run_all_tests()
    # tq.test_polar()
    tq.test_starshade()
