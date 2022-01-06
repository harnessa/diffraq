import numpy as np
import diffraq
import matplotlib.pyplot as plt;plt.ion()
from diffraq.quadrature import lgwt, polar_quad
from scipy.interpolate import InterpolatedUnivariateSpline
import h5py

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

#Rotation angle of occulter [radians]
occ_rot = np.radians(240)

#Initiate
xq, yq, wq = np.array([]), np.array([]), np.array([])
nq, dq = np.array([]), np.array([])

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

    r0 = edge_data[:,0].min()
    r1 = edge_data[:,0].max()

    #Interpolate
    Afunc = InterpolatedUnivariateSpline(edge_data[:,0], edge_data[:,1], k=4, ext=3)

    #############################

    #Scale radius quadrature nodes to physical size
    pr = r0 + (r1 - r0) * pr0

    #Turn seam_width into angle
    seam_width_angle = seam_width / pr

    #Get rotation and side (trailing/leading) of petal
    pet_mul = np.pi/num_pet
    pet_sgn = [-1,1][ip%2]
    pet_add = (2*np.pi/num_pet) * (ip//2)

    #Apodization value at nodes and weights
    Aval = Afunc(pr)
    tt = pet_mul*Aval + pw*seam_width_angle

    #Flip to side
    tt *= pet_sgn

    #Rotate to current angle
    tt += pet_add + occ_rot

    #r*dr
    wi = (r1 - r0) * wr0 * pr

    #Weights (rdr * dtheta)
    cw = (wi * ww * seam_width_angle).ravel()

    #Positions
    cx = (pr * np.cos(tt)).ravel()
    cy = (pr * np.sin(tt)).ravel()

    #Append
    wq = np.concatenate((wq, cw))
    xq = np.concatenate((xq, cx))
    yq = np.concatenate((yq, cy))

    #############################

    #Get gap widths
    cg = (2*Aval*pet_mul*pr).ravel()

    #Get polar function and derivative at edge
    pfunc = Aval*pet_mul*pet_sgn + pet_add + occ_rot
    pdiff = Afunc.derivative(1)(pr) * pet_mul*pet_sgn

    #Get cartesian function and derivative values at the edge
    cf = np.cos(pfunc)
    sf = np.sin(pfunc)
    cfunc = pr*np.hstack((cf, sf))
    cdiff = np.hstack((cf - pr*sf*pdiff, sf + pr*cf*pdiff))

    #Calculate cos(angle) between normal and theta vector (orthogonal to position vector)
    pos_angle = -(cfunc[:,0]*cdiff[:,0] + cfunc[:,1]*cdiff[:,1]) / \
        (np.hypot(*cfunc.T) * np.hypot(*cdiff.T))

    #Build normal angle
    cn = np.arctan2(pet_mul*pet_sgn*cdiff[:,0], -pet_mul*pet_sgn*cdiff[:,1])
    #Give all points same normal angle as the edge
    cn = (np.ones_like(pw) * cn[:,None]).ravel()

    #Build edge distances
    cd = seam_width * (pw * pos_angle[:,None]).ravel()

    #Append
    nq = np.concatenate((nq, cn))
    dq = np.concatenate((dq, cd))


    # # plt.plot(*cfunc.T,'x')
    # # plt.colorbar(plt.scatter(cx, cy, c=abs(cd), s=3))
    # plt.colorbar(plt.scatter(cx, cy, c=np.degrees(cn), s=3))
    #
    # breakpoint()

if do_save:
    with h5py.File(f'./seam_quad_m12p6_{np.degrees(occ_rot):.0f}.h5', 'w') as f:
        f.create_dataset('xq', data=xq)
        f.create_dataset('yq', data=yq)
        f.create_dataset('wq', data=wq)
        f.create_dataset('nq', data=nq)
        f.create_dataset('dq', data=dq)

else:

    plt.colorbar(plt.scatter(xq, yq, c=np.degrees(nq), s=2, cmap=plt.cm.jet))

    breakpoint()
