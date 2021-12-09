import numpy as np
import diffraq
import matplotlib.pyplot as plt;plt.ion()
from diffraq.quadrature import lgwt, polar_quad
from scipy.interpolate import InterpolatedUnivariateSpline
import h5py
import time

"""Verified it gives same diffraction results as with seams built with diffraq"""

###########################################################
###########################################################

seam_width = 25e-6
seam_radial_nodes = 240
seam_theta_nodes = 5

num_pet = 12

do_save = [False, True][1]

if do_save:
    seam_radial_nodes = 500
    seam_theta_nodes = 500

#############################

#Nominal petal
apod_dir = '/home/aharness/repos/diffraq/External_Data/Apodization_Profiles'
nom_file = f'{apod_dir}/bb_2017.h5'

#Add shifted petals as new apodization functions
sml_file = f'{apod_dir}/m12p6_7d5um.h5'
sml_pets = [3,4,11,12]

big_file = f'{apod_dir}/m12p6_10d5um.h5'
big_pets = [19,20]

tik = time.perf_counter()

###########################################################
###########################################################

#Petals width nodes and weights over [0,1]
pw, ww = lgwt(seam_theta_nodes, 0, 1)

#Combine nodes for positive and negative sides of edge (negative pw is positive side)
pw = np.concatenate((pw, -pw[::-1]))
ww = np.concatenate((ww,  ww[::-1]))

#Petals radius nodes and weights over [0,1]
pr0, wr0 = lgwt(seam_radial_nodes, 0, 1)

#Add axis
wr0 = wr0[:,None]
pr0 = pr0[:,None]

#############################

#Initiate
xq, yq, wq = np.array([]), np.array([]), np.array([])
nq, dq = np.array([]), np.array([])
loci = []

#Loop over half petals
for ip in range(2*num_pet):

    #Get edge data
    if ip in big_pets:
        efile = big_file
    elif ip in sml_pets:
        efile = sml_file
    else:
        efile = nom_file

    with h5py.File(efile, 'r') as f:
        edge_data = f['data'][()]

    #Get radii lims
    r0 = edge_data[:,0].min()
    r1 = edge_data[:,0].max()

    #Interpolate
    Afunc = InterpolatedUnivariateSpline(edge_data[:,0], edge_data[:,1], k=4, ext=3)

    #############################

    #Scale radius quadrature nodes to physical size
    pr = r0 + (r1 - r0) * pr0

    #Get rotation and side (trailing/leading) of petal
    pet_sgn = [-1,1][ip%2]
    pet_mul = np.pi/num_pet * pet_sgn
    pet_add = (2*np.pi/num_pet) * (ip//2)

    #For debug:
    xl = edge_data[:,0]*np.cos(edge_data[:,1]*pet_mul + pet_add)
    yl = edge_data[:,0]*np.sin(edge_data[:,1]*pet_mul + pet_add)
    loci.append(np.stack((xl, yl), 1))

    #Get theta value of edge
    Aval = Afunc(pr)
    tt = Aval * pet_mul + pet_add

    #Get cartesian coordinates of edge
    exx = pr * np.cos(tt)
    eyy = pr * np.sin(tt)

    #Get cartesian derivative of edge
    pdiff = Afunc.derivative(1)(pr) * pet_mul
    edx = np.cos(tt) - eyy*pdiff
    edy = np.sin(tt) + exx*pdiff

    #Build unit normal vectors (remember to switch x/y and make new x negative)
    evx = -edy / np.hypot(edx, edy)
    evy =  edx / np.hypot(edx, edy)

    #Get edge distance
    cd = np.ones_like(evx) * pw*seam_width * -pet_sgn

    #Build coordinates in seam
    cx = exx + evx*pw*seam_width
    cy = eyy + evy*pw*seam_width

    #Build normal angle (all get the same on given line)
    cn = np.ones_like(pw) * np.arctan2(pet_mul*edx, -pet_mul*edy)
    cn = np.ones_like(pw) * cn

    #Calculate cos(angle) between normal and theta vector (orthogonal to position vector) at edge
    pos_angle = -(exx*edx + eyy*edy) / (np.hypot(exx, eyy) * np.hypot(edx, edy))

    #dtheta
    wthe = np.abs(ww * seam_width/pr * pos_angle)

    #r*dr
    wrdr = (r1 - r0) * wr0 * pr

    #Weights (rdr * dtheta)
    cw = wrdr * wthe

    #Get gap widths
    cg = 2*Aval*abs(pet_mul)*pr

    #Find overlap (we only want to stop overlap on opposing screen, but we want data in gaps to add)
    ovr_inds = cd > cg

    #Zero out weights on overlap
    cw[ovr_inds] = 0

    #Append
    xq = np.concatenate((xq, cx.ravel()))
    yq = np.concatenate((yq, cy.ravel()))
    wq = np.concatenate((wq, cw.ravel()))
    nq = np.concatenate((nq, cn.ravel()))
    dq = np.concatenate((dq, cd.ravel()))

###########################################################
###########################################################

tok = time.perf_counter()
print(f'Time: {tok-tik:.3f} [s]')

if do_save:
    with h5py.File('./normal_seam_quad_m12p6.h5', 'w') as f:
        f.create_dataset('xq', data=xq)
        f.create_dataset('yq', data=yq)
        f.create_dataset('wq', data=wq)
        f.create_dataset('nq', data=nq)
        f.create_dataset('dq', data=dq)

else:

    plt.colorbar(plt.scatter(xq, yq, c=wq, cmap=plt.cm.jet, s=2))
    for ll in loci:
        plt.plot(*ll.T, 'k')
    plt.axis('equal')

    breakpoint()
