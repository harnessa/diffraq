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
        tsts = ['lgwt', 'polar', 'cartesian', 'starshade', 'loci']
        for t in tsts:
            getattr(self, f'test_{t}')()

############################################

    def test_lgwt(self):

        #Test points
        limits = [[-1,1], [-1,0], [5,10]]
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
                pq, wq = diffraq.quadrature.lgwt(N, a, b)

                assert((np.isclose(pq, p2).all()) & (np.isclose(wq, w2).all()))

        #Cleanup
        del pq, wq, p2, w2

############################################

    def test_polar(self):
        for m in range(20,100,20):
            for n in range(20,100,20):
                for a in np.arange(0.1, 0.9, 0.1):
                    #Smooth radial function on [0, 2pi)
                    gfunc = lambda t: 1 + a*np.cos(3*t)

                    #Get quad
                    xq, yq, wq = diffraq.quadrature.polar_quad(gfunc, m, n)

                    #Assert with analytic area formula
                    assert(np.isclose(wq.sum(), np.pi*(1. + a**2/2)))

        #Cleanup
        del xq, yq, wq

############################################

    def test_cartesian(self):
        for m in range(100,200,20):
            for n in range(100,200,20):
                #Kite occulter
                func = lambda t: np.hstack((0.5*np.cos(t) + 0.5*np.cos(2*t), np.sin(t)))
                diff = lambda t: np.hstack((-0.5*np.sin(t) - np.sin(2*t), np.cos(t)))

                #Get quad
                xq, yq, wq = diffraq.quadrature.cartesian_quad(func, diff, m, n)

                #Assert with analytic area formula
                assert(np.isclose(wq.sum(), np.pi/2))

        #Cleanup
        del xq, yq, wq

############################################

    def test_starshade(self):

        #Starshade
        num_pet = 16
        ss_r0 = 7; ss_r1 = 14
        ss_Afunc = lambda r: np.exp(-((r-ss_r0)/(ss_r1-ss_r0)/0.6)**6)
        hg_area = 374.7984608053253       #Calculated once, not sure if correct

        #Disk
        disk_r0 = 0.7; disk_r1 = 1.3
        disk_Afunc = lambda t: 1 + 0*t

        for m in range(20,100,10):
            for n in range(20,100,10):

                ### Test DISK ###
                #Get quad
                xq, yq, wq = diffraq.quadrature.starshade_quad(disk_Afunc, 1, \
                    disk_r0, disk_r1, m, n)

                #Assert all are the same shape
                nd = int(np.ceil(0.3*n*1)) #number in disk
                assert(xq.shape == yq.shape == wq.shape == (m*n + m*nd,))

                #Assert with analytic area formula
                assert(np.isclose(wq.sum(), np.pi*disk_r1**2))

                ## Test STARSHADE ###
                #Get quad
                xq, yq, wq = diffraq.quadrature.starshade_quad(ss_Afunc, num_pet, \
                    ss_r0, ss_r1, m, n)

                #Assert all are the same shape
                nd = int(np.ceil(0.3*n*num_pet)) #number in disk
                assert(xq.shape == yq.shape == wq.shape == (m*n*num_pet + m*nd,))

                #TODO: need better starshade assertion
                assert(np.isclose(wq.sum(), hg_area))

        #Cleanup
        del xq, yq, wq

############################################

    def test_loci(self):
        
        #Load loci
        loci = np.genfromtxt(f'{diffraq.int_data_dir}/Test_Data/kite_loci_file.txt', delimiter=',')

        for m in range(100,200,20):
            for dn in [2, 5, 10]:

                #Get quad
                xq, yq, wq = diffraq.quadrature.loci_quad(*loci[::dn].T, m)

                #Assert with analytic area formula
                assert(np.isclose(wq.sum(), np.pi/2))

        #Cleanup
        del xq, yq, wq, loci

############################################

if __name__ == '__main__':

    tq = Test_Quadrature()
    tq.run_all_tests()
