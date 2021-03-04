import numpy as np
import matplotlib.pyplot as plt;plt.ion()
from fresnaq_functions import fresnaq_functions
import finufft
from scipy.special import erfc
from diffraq_grid import diffraq_grid

fresnaq = fresnaq_functions()

lambdaz = 9.
ximax = 14; ngrid = 1000
tol = 1e-9
verb = [False,True][1]
Aquad = ['g','u','a'][0]
design = ['disc', 'HG', 'erf', 'NI2'][0]

if design == 'disc':
    Np = 1; r0=5; r1=10
    Afunc = lambda r: 1+0*r
    n = 600; m = 80
elif design == 'erf':
    Np = 16; r0=7; r1=14
    beta = 3
    Afunc = lambda r: erfc(beta*(2*r-(r0+r1))/(r1-r0))/2
    n=30; m=100
elif design == 'HG':
    Np = 16; r0=7; r1=14
    A = lambda t: np.exp(-(t/0.6)**6)
    Afunc = lambda r: A((r-r0)/(r1-r0))
    n=30; m=80

#Get AQ
xq, yq, wq, bx, by = fresnaq.starshadequad(Np, Afunc, r0, r1, n, m, Aquad=Aquad,verb=verb)

if verb:
    plt.figure()
    plt.scatter(xq,yq,s=2,c=wq)
    plt.figure()
    # xi = -5; eta=-10
    xi=0;eta=0
    integ = np.exp((1j*np.pi/lambdaz) * ((xq-xi)**2 + (yq-eta)**2))
    plt.scatter(xq,yq,s=2,c=integ.real)

uu, xigrid  = diffraq_grid(xq, yq, wq, lambdaz, ximax, ngrid, tol)
it=0; jt=0
ut = uu[it,jt]
xi = xigrid[it]
eta = xigrid[jt]

uu = 1- uu

plt.figure()
plt.imshow(np.log10(np.abs(uu)**2))

breakpoint()
