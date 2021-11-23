import numpy as np
import diffraq
import finufft
import matplotlib.pyplot as plt;plt.ion()

radius = 25.e-3
zz = 50.
z0 = 27.5
# z0 = 1e19
wave = 0.6e-6
m0 = 400*3
n0 = 400*3
nf = 512
mf = 512
tol = 1e-12
num_pts = 512
tel_diameter = 5e-3

###################################
### Setup ###
###################################

#Get output grid
grid_pts = diffraq.utils.image_util.get_grid_points(num_pts, tel_diameter)

#Get input quadrature
xq0, yq0, wq0 = diffraq.quadrature.polar_quad(lambda t: \
    np.ones_like(t)*radius, m0, n0)

#Get initial field
u0 = np.exp(1j*2*np.pi/(2*wave*z0)*(xq0**2 + yq0**2))

# #Critical distance
# dx1 = 5e-6      #TODO: whata bout non-uniform
# zcrit = 2*num_pts*dx1**2/wave
#
# #Calculate bandwidth
# if zz < zcrit:
#     bf = 1/dx
# elif zz >= 3*zcrit:
#     bf = np.sqrt(2*num_pts/(wave*zz))
# else:
#     bf = 2*num_pts*dx/(wave*zz)

bf = np.sqrt(2*num_pts/(wave*zz))

hbf = bf/2

#Get frequency quadrature
xqf, yqf, wqf = diffraq.quadrature.polar_quad(lambda t: \
    np.ones_like(t), mf, nf)

###################################
### Angular Spectrum ###
###################################

#Scale factor to become FT
sc = 2.*np.pi * hbf

#Compute strengths
cq = u0 * wq0

#Do FINUFFT
aspec = finufft.nufft2d3(xq0, yq0, cq, sc*xqf, sc*yqf, isign=-1, eps=tol)

###################################
### Diffraction ###
###################################

#Get propagation constant
kz = 1 - (xqf*wave*hbf)**2 - (yqf*wave*hbf)**2
k0inds = kz < 0
# print(np.count_nonzero(k0inds))
Hn = np.exp(1j * 2*np.pi/wave * np.sqrt(np.abs(kz)) * zz)
Hn[k0inds] *= 0

#Propagation kernel * aspec as strengths
cq = aspec * Hn * wqf * hbf**2

#Cleanup
del Hn, kz, k0inds

ox = np.tile(grid_pts, (num_pts,1))
oy = ox.T.flatten()
ox = ox.flatten()

#Do FINUFFT (inverse direction)
uu = finufft.nufft2d3(sc*xqf, sc*yqf, cq, ox, oy, isign=1, eps=tol)
uu = uu.reshape((num_pts, num_pts))

#Transpose to match visual representation
# uu = uu.T

#Normalize
utru = diffraq.utils.solution_util.calculate_circle_solution(grid_pts, \
    wave, zz, z0, radius, False)


diff = abs(uu[len(uu)//2] - utru).max()
print(f'{radius**2/(zz*z0/(zz+z0)*wave):.1f}, {diff:.2e}')

plt.figure()
plt.imshow(abs(uu))

plt.figure()
plt.plot(abs(utru))
plt.plot(abs(uu)[len(uu)//2], '--')

plt.figure()
plt.plot(np.angle(utru))
plt.plot(np.angle(uu)[len(uu)//2], '--')

breakpoint()
