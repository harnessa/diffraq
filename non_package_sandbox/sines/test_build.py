import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import diffraq
import h5py
from scipy.interpolate import interp1d
from utilities import util
from scipy.optimize import newton

use_inn = [False, True][1]
use_raw = [False, True][1]

if use_inn:
    ncycles = 4
    pet_num = 3
else:
    ncycles = 5
    pet_num = -3

def rot(pet, num):
    ang = 2.*np.pi/12*num
    mat = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return pet.dot(mat)

###########################

if use_raw:
    loci_dir = '/home/aharness/repos/Milestone_2/diffraq_analysis/modeling/M12P9/make_shape/Loci'

    nom = np.genfromtxt(f'{loci_dir}/nompetal12p.txt', delimiter='  ')

    if use_inn:
        pet = np.genfromtxt(f'{loci_dir}/petal4_sine12p.txt', delimiter='  ')
    else:
        pet = np.genfromtxt(f'{loci_dir}/petal10_sine12p.txt', delimiter='  ')

    nom[:,1] *= -1
    pet[:,1] *= -1

    ###########################
else:
    apod = 'M12P9'
    loci_dir = '/home/aharness/repos/Milestone_2/diffraq_analysis/modeling/BDW_Compare/bdw_loci'
    deci = '1x'
    def get_bdw_petal(apod, petal):
        with h5py.File(f'{loci_dir}/M12P9_{deci}.h5', 'r') as f:
            num_pet = f['num_petals'][()]
            data = f[f'loci_{petal}'][()]

        return data

    if use_inn:
        bdw_pet = get_bdw_petal(apod, 9)
    else:
        bdw_pet = get_bdw_petal(apod, 3)

    nom = rot(get_bdw_petal(apod, 11), -1)
    pet = rot(bdw_pet, -pet_num)
    nom = nom[::-1]
    pet = pet[::-1]

###########################

rot_pet = rot(pet, pet_num)
old_nom = nom.copy()
old_pet = pet.copy()

dd = np.hypot(*(pet - nom).T)

if use_raw:
    inds = np.where(dd != 0)[0]
    if use_inn:
        inds = inds[:-2]
        nquad = 784
    else:
        inds = inds[:-7]
        nquad = 574
else:

    inds = np.where(dd > 1e-12)[0]
    if use_inn:
        inds = inds[:-4]
        nquad = 742
    else:
        inds = inds[:-9]
        nquad = 603

pet = pet[inds]
nom = nom[inds]
rads = np.hypot(*nom.T)

xy0 = rot_pet[inds][0]
freq = ncycles/rads.ptp()

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
    pa, wa = diffraq.quadrature.lgwt(pert.num_quad, 0, 1)
    # pa = np.linspace(0, 1, pert.num_quad, endpoint=False)[::-1]
    # wa = np.ones_like(pa)/pert.num_quad

    #Add second half of wave (weights are negative on bottom half)
    pa = np.concatenate((pa[::-1], 1+pa[::-1]))
    wa = np.concatenate((wa[::-1],  -wa[::-1]))

    #Build all cycles
    pa = (pa + 2*np.arange(pert.num_cycles)[:,None]).ravel()
    wa = (wa * np.ones(pert.num_cycles)[:,None]).ravel()

    #Normalize weights
    wa *= dt

    #Build parameters across all cycles
    ts = t0 + dt * pa

    #Turn into parameter if petal
    ts_use = pert.parent.pack_param(ts, np.sign(p0)*1)   #rotate to first petal

    ### get_kluge_edge ###
    scale = 12/16

    #I need a kluge to get radius points from original
    # ue = nom.copy() #Load this from file?
    # ur2 = np.hypot(*ue.T)    #Or this
    #FIXME: replace this ur with something generated without knowing loci

    #uniformly spaced x's in 16 petal space
    t1 = ts.max()
    theta = lambda r: pert.parent.outline.func(r)*np.pi/pert.parent.num_petals
    x0 = t0*np.cos(theta(t0)*scale)
    x1 = t1*np.cos(theta(t1)*scale)
    xs = np.linspace(x0, x1, len(ts))
    #turn xs into rs
    func = lambda r, x: r - x/np.cos(theta(r)*scale)
    ur = np.array([newton(func, ts[i], args=(xs[i],)) for i in range(len(xs))])
    # ur = xs/np.cos(theta(ts)*scale)

    # breakpoint()

    ##
    uxx = 2*ncycles*np.linspace(0, 1, len(ts))   #uniform cycle nodes
    pa = np.interp(ts, ur, uxx)                  #interpolated cycle nodes

    # ts = rads.copy()
    # t0 = rads.min()
    # ts2 = np.linspace(0,1, len(rads))
    # pa = 2*ts2*pert.num_cycles

    #Build old edge
    old_edge = pert.parent.cart_func(ts_use[:,None])

    # old_edge = nom.copy()

    # old_edge = interp1d(rads, nom, axis=0, bounds_error=False,\
    #     fill_value='extrapolate', kind='linear')(ts)
    # # breakpoint()

    #Build new old edge w/ 16 petals
    a0 = np.arctan2(old_edge[:,1], old_edge[:,0]) * scale
    pet0 = np.hypot(old_edge[:,0], old_edge[:,1])[:,None] * \
        np.stack((np.cos(a0), np.sin(a0)),1)

    #Build normal from this edge
    normal = (np.roll(pet0, 1, axis=0) - pet0)[:,::-1] * np.array([-1,1])
    # normal = util.compute_derivative(pet0, xaxis=ts)[:,::-1] * np.array([1,-1])
    normal /= np.hypot(normal[:,0], normal[:,1])[:,None]
    #Fix normals on end pieces
    normal[0] = normal[1]
    normal[-1] = normal[-2]

    #Build sine wave
    sine = pert.amplitude * np.sin(np.pi*pa)[:,None]

    # rr = np.hypot(*pet.T)
    # sine = pert.amplitude * np.sin(2*np.pi*(ts-ts[0])*pert.num_cycles/ts.ptp())[:,None]

    # plt.plot(rads, sine)
    # breakpoint()

    #Build new edge
    edge0 = pet0 + normal * sine

    #Convert to polar cooords
    anch = np.arctan2(edge0[:,1], edge0[:,0]) / scale

    #Scale and go back to cart coords
    nch = np.hypot(edge0[:,0], edge0[:,1])[:,None] * \
        np.stack((np.cos(anch), np.sin(anch)), 1)

    #New edge
    new_edge = scale*nch + old_edge*(1-scale)

    # #Rotate back to original petal position
    # rot_ang = 2*np.pi/pert.parent.num_petals*(abs(p0) - 1)
    # rot_mat = pert.parent.parent.build_rot_matrix(rot_ang)
    # old_edge2 = old_edge.dot(rot_mat)
    # new_edge2 = new_edge.dot(rot_mat)

drad = np.hypot(*old_edge.T)

# drad = rads

############################################
#reinterpolate with common x-axis
# new_edge_y = np.interp(old_edge[:,0], new_edge[:,0], new_edge[:,1])
pet_y = np.interp(nom[:,0], pet[:,0], pet[:,1])

ny2 = np.interp(nom[:,0], new_edge[:,0], new_edge[:,1])
oy2 = np.interp(nom[:,0], old_edge[:,0], old_edge[:,1])

plt.figure()
plt.plot(nom[:,0], pet_y - nom[:,1], 'b')
plt.plot(nom[:,0], ny2 - oy2, 'r--')

# plt.plot(rads, pet_y - nom[:,1], 'b')
# plt.plot(drad, new_edge_y - old_edge[:,1], 'r--')

plt.figure()
plt.plot(pet_y - ny2)

print(abs(pet_y-ny2).mean(), (pet_y-ny2).std())

# xx = nom[:,0]
# py3 = np.interp(xx, pet[:,0], pet[:,1])
# ny3 = np.interp(xx, new_edge[:,0], new_edge[:,1])
# oy3 = np.interp(xx, old_edge[:,0], old_edge[:,1])
# plt.figure()
# plt.plot(xx, py3 - nom[:,1], 'b')
# plt.plot(xx, ny3 - oy3, 'r--')

breakpoint()

############################################
# plt.figure()
# plt.plot(*nom.T)
# plt.plot(*nom[0], 'o')
# plt.plot(*old_edge.T, '--')
# plt.plot(*old_edge[0], 's')

#Plot
# plt.figure()
# plt.plot(rads, (pet - nom)[:,0], 'b')
# plt.plot(rads, (pet - nom)[:,1], 'c')
# plt.plot(drad, (new_edge - old_edge)[:,0], 'r--')
# plt.plot(drad, (new_edge - old_edge)[:,1], 'k--')

# plt.figure()
# plt.plot(rads-drad)
#
# plt.figure()
# plt.plot(np.diff(rads), 'x')
# plt.plot(np.diff(drad), '+')

# plt.figure()
# plt.plot(*rot_pet.T)
# plt.plot(*old_nom.T)
# plt.plot(*old_edge.T, '--')
# plt.plot(*new_edge.T, '--')
# breakpoint()
