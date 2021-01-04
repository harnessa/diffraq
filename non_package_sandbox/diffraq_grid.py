import numpy as np
import matplotlib.pyplot as plt;plt.ion()
from fresnaq_functions import fresnaq_functions
import finufft
from diffraq_pts import diffraq_pts

fresnaq = fresnaq_functions()

def diffraq_grid(xq, yq, wq, lambdaz, ximax, ngrid, tol):

    xigrid = ximax*(2*np.arange(ngrid)/ngrid - 1)
    #target grid spacing
    dxi = 2*ximax/ngrid
    #handle the odd case
    if ngrid % 2 == 1:
        xigrid += dxi/2
    #scale factor to become FT
    sc = 2.*np.pi/lambdaz
    #scaled grid spacing
    dk = sc*dxi
    #max NU coord
    maxNU = max(abs(dk*xq.max()), abs(dk*yq.max()))
    #premult by quadratic bit
    cq = np.exp(1j*np.pi/lambdaz*(xq**2 + yq**2)) * wq
    #only if needed
    if maxNU > 3*np.pi:
        #Wrap in case grid too coarse
        xq = np.mod(xq, 2*np.pi/dk)
        yq = np.mod(yq, 2*np.pi/dk)
    #Do FINUFFT
    uu = finufft.nufft2d1(dk*xq, dk*yq, cq, isign=-1, eps=tol, n_modes=(ngrid, ngrid))
    #post multiply by quadr bit (outer prod since separable grid)...
    uu *= np.exp((1j*np.pi/lambdaz)*xigrid**2) * np.exp((1j*np.pi/lambdaz)*xigrid[None].T**2)
    #Kirchhoff prefactor
    kirchfac = 1./(1j*lambdaz)
    uu *= kirchfac

    return uu, xigrid

def test_grid():
    #Fresnel number
    fresnum = 10.
    lambdaz = 1./fresnum
    #smooth radial function on [0,2pi]
    g = lambda t: 1 + 0.3*np.cos(3*t)
    #areal quadrature
    n=350; m=120
    xq, yq, wq, bx, by = fresnaq.polarareaquad(g,n,m)
    tol = 1e-9

    if [False,True][1]:
        #small grid for math test
        ximax, ngrid = 1.5, 100
    else:
        #big grid for speed tests
        ximax, ngrid = 1.5, 1000

    #compute
    uu, xigrid = diffraq_grid(xq[:,0], yq[:,0], wq[:,0], lambdaz, ximax, ngrid, tol)

    #recreate grid
    xi = np.tile(xigrid,(ngrid,1)).T
    eta = xi.copy().T

    #check one target (grid SW corner)
    # i=(10,50)
    i=(0,0)
    kirchfac = 1/(1j*lambdaz)
    ud = np.empty_like(uu)
    for j in range(uu.shape[0]):
        for k in range(uu.shape[1]):
            ud[j,k] =  kirchfac * np.sum(np.exp((1j*np.pi/lambdaz)*((xq - xi[j,k])**2 + (yq - eta[j,k])**2))*wq)
    ud0 = ud[i]
    # ud = kirchfac * np.sum(np.exp((1j*np.pi/lambdaz)*((xq - xi[i])**2 + (yq - eta[i])**2))*wq)
    uu0 = uu[i]
    print(f'\nabs error vs direct Fresnel quadr at' + \
        f' ({xi[i]:.3f},{eta[i]:.3f}) = {abs(uu0 - ud0):.3e}\n')

    upts = diffraq_pts(xq[:,0], yq[:,0], wq[:,0], lambdaz, xi, eta, tol)
    up0 = diffraq_pts(xq[:,0], yq[:,0], wq[:,0], lambdaz, xi[i], eta[i], tol)[0]
    fig, axes = plt.subplots(2,3)
    axes[0,0].imshow(np.abs(ud))
    axes[0,1].imshow(np.abs(uu))
    axes[0,2].imshow(np.abs(upts))
    axes[1,0].imshow(np.angle(ud))
    axes[1,1].imshow(np.angle(uu))
    axes[1,2].imshow(np.angle(upts))
    print(f'max abs err fresnaq_pts vs grid: {np.abs(upts-uu).max():.3e}\n')

    breakpoint()

if __name__ == '__main__':

    test_grid()
