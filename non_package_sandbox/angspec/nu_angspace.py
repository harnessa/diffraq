import numpy as np
import diffraq
import finufft
import matplotlib.pyplot as plt;plt.ion()

radius = 12.e-3
zz = 50.
# z0 = 27.5
z0 = 1e19
wave = 0.6e-6
n = 1000
m = 200            #Number of quadrature points
tol = 1e-12
num_pts = 512
tel_diameter = 5e-3

radius = 1
zz = 1
wave = 0.1
tel_diameter = 3

###################################
### Setup ###
###################################

#Get output grid
grid_pts = diffraq.utils.image_util.get_grid_points(num_pts, tel_diameter)

#Get input quadrature
xq0, yq0, wq0 = diffraq.quadrature.polar_quad(lambda t: \
    np.ones_like(t)*radius, m, n)

Dmax = 2*radius

#Get initial field
u0 = np.exp(1j*2*np.pi/(2*wave*z0)*(xq0**2 + yq0**2))

#Get sampling requirements
ngrid = len(grid_pts)
dgrid = grid_pts[1] - grid_pts[0]
D = (Dmax/2 + grid_pts.max())       #Maximum extent

#Get max alpha
amax = max(D/np.hypot(zz,D)/wave, Dmax/(wave*zz)) * wave
amax = min(amax, 1/np.sqrt(2))

#Get frequency quadrature
xqf, yqf, wqf = diffraq.quadrature.polar_quad(lambda t: \
    np.ones_like(t)*amax, m, n)

###################################
### Angular Spectrum ###
###################################

#Scale factor to become FT
sc = 2.*np.pi/wave

#Compute strengths
cq = u0 * wq0

#Do FINUFFT
aspec = finufft.nufft2d3(xq0, yq0, cq, sc*xqf, sc*yqf, isign=-1, eps=tol)


# over_sample = 4
# dangs = 1/(2*Dmax*over_sample) * wave
# nangs = int(np.ceil(2*amax/dangs/2)) * 2
# angs_pts = (np.arange(nangs) - nangs/2)*dangs
# angs_pts = np.tile(angs_pts, (nangs,1))
# xqf = angs_pts.flatten()
# yqf = angs_pts.T.flatten()
# wqf = dangs
# aspec = finufft.nufft2d1(sc*dangs*xq0, sc*dangs*yq0, cq, (nangs, nangs), isign=-1, eps=tol).flatten()

# plt.scatter(xqf, yqf, c=abs(aspec), s=1)
# plt.imshow(abs(aspec).reshape((nangs,nangs)))
# breakpoint()

###################################
### Diffraction ###
###################################

#Scaled grid spacing
dkg = sc * dgrid

#Max NU coord
maxNU = dkg * amax

#only if needed
if maxNU > 3*np.pi:
    print('too coarse diff')
    #Wrap in case grid too coarse
    xqf = np.mod(xqf, 2*np.pi/dkg)
    yqf = np.mod(yqf, 2*np.pi/dkg)

#Get propagation constant
kz = 1 - xqf**2 - yqf**2
k0inds = kz < 0
kz = np.exp(1j * 2*np.pi/wave * np.sqrt(np.abs(kz)) * zz)
kz[k0inds] *= 0

#Propagation kernel * aspec as strengths
cq = aspec * kz * wqf

#Cleanup
del kz, k0inds

#Do FINUFFT (inverse direction)
uu = finufft.nufft2d1(dkg*xqf, dkg*yqf, cq, (ngrid, ngrid), isign=1, eps=tol)

#Transpose to match visual representation
# uu = uu.T

#Normalize
uu /= wave**2

utru = diffraq.utils.solution_util.calculate_circle_solution(grid_pts, \
    wave, zz, z0, radius, False)


plt.imshow(abs(uu))

plt.figure()
plt.plot(abs(utru))
plt.plot(abs(uu)[len(uu)//2], '--')

breakpoint()
