"""
test_quadrature.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test of building areal quadrature.

"""

import diffraq
import numpy as np
import h5py
from scipy.special import fresnel

class Test_Quadrature(object):

    def run_all_tests(self):
        tsts = ['lgwt_A', 'lgwt_B', 'polar', 'cartesian', 'starshade',
            'loci', 'triangle', 'integration'][1:2]
        for t in tsts[-1:]:
            getattr(self, f'test_{t}')()

############################################

    def test_lgwt_A(self):

        #Test points
        limits = [[-1,1], [-1,0], [5,10]]
        nums = [5, 10, 1001]

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

                assert((np.allclose(pq, p2)) & (np.allclose(wq, w2)))

        #Cleanup
        del pq, wq, p2, w2

############################################

    def test_lgwt_B(self):

        #Test points
        limits = [[-1,1], [0,1]]
        nums = [500, 5000]

        for (a,b) in limits:
            for N in nums:

                #Diffraq answer
                pq, wq = diffraq.quadrature.lgwt(N, a, b)

                #fresnaq answer
                p2, w2 = diffraq.quadrature.fresnaq_lgwt(N, a, b)

                assert((np.allclose(pq, p2)) & (np.allclose(wq, w2)))

        #Cleanup
        del pq, wq, p2, w2

############################################

    def test_polar(self):
        for m in range(20,80,20):
            for n in range(20,80,20):
                for a in np.arange(0.1, 0.9, 0.3):
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
        for m in range(100,200,50):
            for n in range(100,200,50):
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

        for m in range(20,80,20):
            for n in range(20,80,20):

                ### Test DISK ###
                #Get quad
                xq, yq, wq = diffraq.quadrature.petal_quad(disk_Afunc, 1, \
                    disk_r0, disk_r1, m, n)

                #Assert all are the same shape
                nd = int(np.ceil(0.3*n*1)) #number in disk
                assert(xq.shape == yq.shape == wq.shape == (m*n + m*nd,))

                #Assert with analytic area formula
                assert(np.isclose(wq.sum(), np.pi*disk_r1**2))

                ## Test STARSHADE ###
                #Get quad
                xq, yq, wq = diffraq.quadrature.petal_quad(ss_Afunc, num_pet, \
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
        with h5py.File(f'{diffraq.int_data_dir}/Test_Data/kite_loci_file.h5', 'r') as f:
            loci = f['loci'][()]

        for m in range(100,200,50):
            for dn in [2, 5, 10]:

                #Get quad
                xq, yq, wq = diffraq.quadrature.loci_quad(*loci[::dn].T, m)

                #Assert with analytic area formula
                assert(np.isclose(wq.sum(), np.pi/2))

        #Cleanup
        del xq, yq, wq, loci

############################################

    def test_triangle(self):
        #Build vertices
        vv = np.exp(2j*np.pi/3*(np.arange(3)+1))
        vx = vv.real
        vy = vv.imag

        #True area
        tru_area = 0.5 * ((vx[1] - vx[0])*(vy[2] - vy[0]) - (vy[1] - vy[0])*(vx[2] - vx[0]))

        for m in range(20,80,20):

            #Get quad
            xq, yq, wq = diffraq.quadrature.triangle_quad(vx, vy, m)

            #Make sure area agrees
            assert(np.isclose(wq.sum(), tru_area))

            #yq should be odd
            assert(np.isclose((wq*yq**3).sum(), 0))

############################################

    def test_integration(self):

        zz = 10
        npts = 100
        pw, ww = diffraq.quadrature.lgwt(npts, 0, zz)

        #Integrate cosine/sine
        Sa = np.sum(np.sin(np.pi*pw**2/2)*ww)
        Ca = np.sum(np.cos(np.pi*pw**2/2)*ww)
        ans = Ca + 1j*Sa

        #Integrate exponential
        exp = np.sum(np.exp(1j*np.pi*pw**2/2)*ww)

        #Truth
        St, Ct = fresnel(zz)
        tru = Ct + 1j*St

        #Assert true
        assert(np.isclose(tru, exp) and np.isclose(tru,ans))

############################################

if __name__ == '__main__':

    tq = Test_Quadrature()
    tq.run_all_tests()
