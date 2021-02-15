"""
lgwt.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Gauss-Legendre quadrature scheme on a 1D interval.
    Taken from FRESNAQ's lgwt.m (Barnett 2021).

"""

import numpy as np

def lgwt(N, a, b):
    """
    pq, wq = lgwt(N, a, b)

    computes the N-point Legendre-Gauss nodes p and weights w on a 1D interval [a,b].

    Inputs:
        N = # node points
        a = lower bound
        b = upper bound

    Outputs:
        pq = numpy array of nodes
        wq = numpy array of weights
    """

    #Use numpy legendre (isn't tested above 100, but agrees for me)
    # if N < 10000:
    if True:
        #Get nodes, weights on [-1,1]
        p1, wq = np.polynomial.legendre.leggauss(N)
        p1 = p1[::-1]
        wq = wq[::-1]

        #Linear map from [-1,1] to [a,b]
        pq = (a*(1-p1) + b*(1+p1))/2

        #Normalize the weights
        wq *= (b-a)/2

        return pq, wq

    #Uniform grid of nodes on [-1,1]
    pu = np.linspace(-1, 1, N)

    #Initial guess
    y = np.cos((2.*np.arange(N) + 1)*np.pi/(2*N)) + \
        (0.27/N)*np.sin(np.pi*pu*(N-1)/(N+1))

    #Compute the zeros of the N+1 Legendre Polynomial using the recursion
    #   relation and the Newton-Raphson method

    #Legendre-Gauss Vandermonde Matrix
    LGV = np.zeros((N, N+1))
    LGV[:, 0] = 1         #Always 1

    #indices
    ks = np.arange(2, N+1).astype(int)

    #Iterate until new points are uniformly within epsilon of old points
    y0=2
    while max(abs(y-y0)) > np.spacing(1):

        #Update LGV matrix
        LGV[:,1] = y
        for k in ks:
            LGV[:,k] = ( (2*k - 1)*y*LGV[:,k-1] - (k - 1)*LGV[:,k-2] ) / k

        #Derivative matrix
        LGP = (N+1) * (LGV[:,N-1] - y*LGV[:,N]) / (1. - y**2)

        y0 = y
        y = y0 - LGV[:,N]/LGP

    #Linear map from [-1,1] to [a,b]
    pq = (a*(1-y) + b*(1+y))/2

    #Compute the weights
    wq = (b-a) / ((1-y**2)*LGP**2) * ((N+1)/N)**2

    #Cleanup
    del pu, LGV, LGP, ks

    return pq, wq
