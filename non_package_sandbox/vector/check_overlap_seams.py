import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import diffraq

r0 = 8e-3
r1 = 9e-3
y0 = 12.e-6
num_pet = 12
m,n = 50, 50
seam_width = 5e-6


seam_width = y0*1.25

#Apod function
afunc = lambda r: y0*num_pet/(np.pi*r)
dfunc = lambda r: -y0*num_pet/(np.pi*r**2)

#Get seam quads
xq, yq, wq, dept_nodes, indt_values = diffraq.polarization.quadrature.seam_petal_quad( \
    lambda r: afunc(r)/num_pet, 1, r0, r1, m, n, seam_width)


####
rads = np.linspace(r0, r1, 1000)
xx = rads*np.cos(afunc(rads)*np.pi/num_pet)
yy = rads*np.sin(afunc(rads)*np.pi/num_pet)
###

#Get petal signs and angle to rotate
ones = np.ones(2*m, dtype=int)
pet_mul = np.concatenate((ones, -ones))
pet_add = 1

#Get function and derivative values at the parameter values
Aval = afunc(indt_values)
func = Aval*pet_mul + pet_add
diff = dfunc(indt_values)*pet_mul

#Get gaps widths
gw = (2*Aval*np.pi/num_pet*indt_values).ravel()

#Get cartesian function and derivative values at the parameter values
pang = np.pi/num_pet
cf = np.cos(func*pang)
sf = np.sin(func*pang)
cart_func = indt_values[:,None]*np.stack((cf, sf), func.ndim).squeeze()
cart_diff = np.stack((cf - indt_values*sf*diff*pang, sf + indt_values*cf*diff*pang), diff.ndim).squeeze()

#Calculate angle between normal and theta vector (orthogonal to position vector)
pos_angle = -(cart_func[...,0]*cart_diff[...,0] + \
    cart_func[...,1]*cart_diff[...,1]) / (np.hypot(cart_func[...,0], \
    cart_func[...,1]) * np.hypot(cart_diff[...,0], cart_diff[...,1]))

#Build normal angle
nq = np.arctan2(pet_mul*cart_diff[...,0], -pet_mul*cart_diff[...,1]).ravel()

#Get distances
dq = seam_width * (dept_nodes * pos_angle).ravel()

# plt.figure(); plt.colorbar(plt.scatter(xq, yq, c=pos_angle,s=1))
# plt.plot(xx, yy, 'r'); plt.plot(xx, -yy, 'r:')
# plt.figure(); plt.colorbar(plt.scatter(xq, yq, c=dq,s=1))
# plt.plot(xx, yy, 'r'); plt.plot(xx, -yy, 'r:')
# plt.figure(); plt.colorbar(plt.scatter(xq, yq, c=nq,s=1))
# plt.plot(xx, yy, 'r'); plt.plot(xx, -yy, 'r:')
# plt.figure(); plt.colorbar(plt.scatter(xq, yq, c=np.heaviside(dq, 1),s=1))
# plt.plot(xx, yy, 'r'); plt.plot(xx, -yy, 'r:')
# breakpoint()

#Build full gap widths
bigw = (gw[:,None] * np.ones_like(dept_nodes)).ravel()

#Find overlap
ovr_inds = dq >= bigw/2

#Zero out weights on overlap
wq[ovr_inds] = 0

#Compare areas (in open area only)
trua = (r1 - r0)*min(y0, seam_width)*2
area = (wq*np.heaviside(dq,1)).sum()

print(np.isclose(trua, area), f'{trua:.1e}, {area:.1e}')


plt.figure(); plt.colorbar(plt.scatter(xq, yq, c=wq, s=1))
plt.plot(xx, yy, 'r'); plt.plot(xx, -yy, 'r--')
plt.plot(xq[ovr_inds], yq[ovr_inds], 'x')

breakpoint()
