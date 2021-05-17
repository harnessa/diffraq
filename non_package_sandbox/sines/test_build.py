import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import diffraq

loci_dir = '/home/aharness/repos/Milestone_2/diffraq_analysis/modeling/M12P9/make_shape/Loci'

nom = np.genfromtxt(f'{loci_dir}/nompetal12p.txt', delimiter='  ')

use_inn = [False, True][1]

if use_inn:
    pet = np.genfromtxt(f'{loci_dir}/petal4_sine12p.txt', delimiter='  ')
    ncycles = 4
    pet_num = 3
else:
    pet = np.genfromtxt(f'{loci_dir}/petal10_sine12p.txt', delimiter='  ')
    ncycles = 5
    pet_num = -3

def rot(pet, num):
    ang = 2.*np.pi/12*num
    mat = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return pet.dot(mat)

nom[:,1] *= -1
pet[:,1] *= -1

rot_pet = rot(pet, pet_num)
old_nom = nom.copy()

dd = np.hypot(*(pet - nom).T)

inds = np.where(dd != 0)[0]
if use_inn:
    inds = inds[:-2]
    nquad = 784
else:
    inds = inds[:-7]
    nquad = 574

pet = pet[inds]
nom = nom[inds]
rads = np.hypot(*nom.T)

xy0 = rot_pet[dd != 0][0]
freq = ncycles/rads.ptp()

if use_inn:
    xy0 = np.array([-0.0992, -9.3391]) * 1e-3
    freq = 1379.94
else:
    xy0 = np.array([3.5509, 20.0539]) * 1e-3
    freq = 2155.24

# plt.plot(*rot_pet.T)
# plt.plot(*xy0, 'o')
# breakpoint()

############################################
params = {
    'radial_nodes':     1000,
    'theta_nodes':      100,
    'occulter_config':   f'{diffraq.occulter_dir}/bb_2017.cfg',
}
sim = diffraq.Simulator(params)
sine = {'kind':'sines', 'is_kluge':True, 'amplitude':2e-6, 'xy0':xy0, \
        'num_cycles':ncycles, 'frequency':freq, 'num_quad':nquad}

pert = diffraq.geometry.Sines(sim.occulter.shapes[0], **sine)
#Get location of perturbation
t0 = pert.get_start_point()

#Get radius start
t0, p0 = pert.parent.unpack_param(t0)[:2]
#Get change in parameter for one-half cycle
dt = 0.5/pert.frequency

#Get new edge at theta nodes
if [False, True][0]:
    old_edge, new_edge, pw, ww = pert.get_new_edge(t0, p0, dt)
else:
    ### Get new edge ###
    #Nodes along half-wave
    # pa, wa = diffraq.quadrature.lgwt(pert.num_quad, 0, 1)
    pa = np.linspace(0, 1, pert.num_quad, endpoint=False)[::-1]
    wa = np.ones_like(pa)/pert.num_quad

    #Add second half of wave (weights are negative on bottom half)
    pa = np.concatenate((pa[::-1], 1+pa[::-1]))
    wa = np.concatenate((wa[::-1],  -wa[::-1]))

    #Build all cycles
    pa = (pa + 2*np.arange(pert.num_cycles)[:,None]).ravel()
    wa = (wa * np.ones(pert.num_cycles)[:,None]).ravel()

    #Build parameters across all cycles
    ts = t0 + dt * pa

    #Normalize weights
    wa *= dt

    #Turn into parameter if petal
    ts_use = pert.parent.pack_param(ts, np.sign(p0)*1)   #rotate to first petal

    ### get_kluge_edge ###
    scale = 12/16

    #Build old edge
    old_edge = pert.parent.cart_func(ts_use[:,None])

    # old_edge = nom.copy()

    #Build new old edge w/ 16 petals
    a0 = np.arctan2(old_edge[:,1], old_edge[:,0]) * scale
    pet0 = np.hypot(old_edge[:,0], old_edge[:,1])[:,None] * \
        np.stack((np.cos(a0), np.sin(a0)),1)

    #Build normal from this edge
    normal = (np.roll(pet0, 1, axis=0) - pet0)[:,::-1] * np.array([-1,1])
    normal /= np.hypot(normal[:,0], normal[:,1])[:,None]
    #Fix normals on end pieces
    normal[0] = normal[1]
    normal[-1] = normal[-2]

    #Build sine wave
    sine = pert.amplitude * np.sin(np.pi*pa)[:,None]

    #Build new edge
    edge0 = pet0 + normal * sine

    #Convert to polar cooords
    anch = np.arctan2(edge0[:,1], edge0[:,0]) / scale

    #Scale and go back to cart coords
    nch = np.hypot(edge0[:,0], edge0[:,1])[:,None] * \
        np.stack((np.cos(anch), np.sin(anch)), 1)

    #New edge
    new_edge = scale*nch + old_edge*(1-scale)

    #Rotate back to original petal position
    rot_ang = 2*np.pi/pert.parent.num_petals*(abs(p0) - 1)
    rot_mat = pert.parent.parent.build_rot_matrix(rot_ang)
    old_edge2 = old_edge.dot(rot_mat)
    new_edge2 = new_edge.dot(rot_mat)

drad = np.hypot(*old_edge.T)

# drad = rads
############################################

# plt.figure()
# plt.plot(*nom.T)
# plt.plot(*nom[0], 'o')
# plt.plot(*old_edge.T, '--')
# plt.plot(*old_edge[0], 's')

#Plot
plt.figure()
plt.plot(rads, (pet - nom)[:,0], 'b')
plt.plot(rads, (pet - nom)[:,1], 'c')
plt.plot(drad, (new_edge - old_edge)[:,0], 'r--')
plt.plot(drad, (new_edge - old_edge)[:,1], 'k--')

# plt.figure()
# plt.plot(rads-drad)
#
# plt.figure()
# plt.plot(np.diff(rads), 'x')
# plt.plot(np.diff(drad), '+')

plt.figure()
plt.plot(*rot_pet.T)
plt.plot(*old_nom.T)
plt.plot(*old_edge.T, '--')
plt.plot(*new_edge.T, '--')
breakpoint()
