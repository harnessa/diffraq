import numpy as np
import matplotlib.pyplot as plt;plt.ion()
from fresnaq_functions import fresnaq_functions
import finufft

fresnaq = fresnaq_functions()

def diffraq_pts(xq, yq, wq, lambdaz, xi, eta, tol):

    #cq will be input strengths to NUFFT...

    #premult by quadratic bit
    cq = np.exp(1j*np.pi/lambdaz*(xq**2 + yq**2)) * wq
    #scale factor to become FT
    sc = 2.*np.pi/lambdaz
    #Kirchhoff prefactor
    kirchfac = 1./(1j*lambdaz)

    #Do FINUFFT
    uu = finufft.nufft2d3(xq, yq, cq, sc*np.atleast_1d(xi).flatten(), \
        sc*np.atleast_1d(eta).flatten(), isign=-1, eps=tol)
    #Reshape
    uu = uu.reshape(np.atleast_1d(xi).shape)
    #post multiply bit
    uu *= kirchfac * np.exp((1j*np.pi/lambdaz)*(xi**2 + eta**2))

    return uu

def test_pts():
    #Fresnel number
    fresnum = 10.
    lambdaz = 1./fresnum
    #smooth radial function on [0,2pi]
    g = lambda t: 1 + 0.3*np.cos(3*t)
    #areal quadrature
    n=350; m=120
    xq, yq, wq, bx, by = fresnaq.polarareaquad(g,n,m)
    tol = 1e-9

    xi = -1.5
    eta = -1.5

    #compute
    uu = diffraq_pts(xq[:,0], yq[:,0], wq[:,0], lambdaz, xi, eta, tol)

    #check one target (grid SW corner)
    kirchfac = 1/(1j*lambdaz)
    ud = kirchfac * np.sum(np.exp((1j*np.pi/lambdaz)*((xq - xi)**2 + (yq - eta)**2))*wq)

    print(f'\nabs error vs direct Fresnel quadr at' + \
        f' ({xi:.3f},{eta:.3f}) = {float(abs(uu - ud)):.3e}\n')
    breakpoint()

if __name__ == '__main__':

    test_pts()
