import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import diffraq
import h5py

use_inn = [False, True][0]
use_raw = [False, True][0]

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
apod = 'M12P9'
bdw_dir = '/home/aharness/repos/Milestone_2/diffraq_analysis/modeling/BDW_Compare'
if use_raw:
    loci_dir = '/home/aharness/repos/Milestone_2/diffraq_analysis/modeling/M12P9/make_shape/Loci'

    bnom = np.genfromtxt(f'{loci_dir}/nompetal12p.txt', delimiter='  ')

    if use_inn:
        bpet = np.genfromtxt(f'{loci_dir}/petal4_sine12p.txt', delimiter='  ')
    else:
        bpet = np.genfromtxt(f'{loci_dir}/petal10_sine12p.txt', delimiter='  ')

    bnom[:,1] *= -1
    bpet[:,1] *= -1

    ###########################
else:
    loci_dir = f'{bdw_dir}/bdw_loci'
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

    bnom = rot(get_bdw_petal(apod, 11), -1)
    bpet = rot(bdw_pet, -pet_num)
    bnom = bnom[::-1]
    bpet = bpet[::-1]

###########################

params = {
    'radial_nodes':     5000,
    'occulter_config':  f'{bdw_dir}/plain_configs/{apod}.cfg',
}

sim = diffraq.Simulator(params)
sim.occulter.build_quadrature()
#Nominal
edge = sim.occulter.shapes[0].build_local_shape_edge()
angs = np.arctan2(edge[:,1], edge[:,0]) %(2.*np.pi)
dnom = edge[(angs < 3*np.pi/12) & (angs > np.pi/12)].copy()
dnom = rot(dnom, 1)
angs = np.arctan2(dnom[:,1] - dnom[:,1].mean(), dnom[:,0] - dnom[:,0].mean()) % (2.*np.pi)
dnom = dnom[np.argsort(angs)]

#Perturbed
if use_inn:
    dold, dnew = sim.occulter.shapes[0].pert_list[0].get_edge_points(doret=True)
    dpet = np.concatenate((dold[::-1], dnew))
    nrot = -3
else:
    dold, dnew = sim.occulter.shapes[0].pert_list[1].get_edge_points(doret=True)
    dpet = np.concatenate((dold[::-1], dnew))
    nrot = 3

dpet = rot(dpet, nrot)
dnew = rot(dnew, nrot)
dold = rot(dold, nrot)

del angs, edge

#Limit to bottom half
if use_inn:
    binds = (bnom[:,0] > 8.5e-3) & (bnom[:,0] < 13e-3) & (bnom[:,1] < 0)
    dinds = (dnom[:,0] > 8.5e-3) & (dnom[:,0] < 13e-3) & (dnom[:,1] < 0)
else:
    binds = (bnom[:,0] > 15e-3) & (bnom[:,0] < 24e-3) & (bnom[:,1] < 0)
    dinds = (dnom[:,0] > 15e-3) & (dnom[:,0] < 24e-3) & (dnom[:,1] < 0)

bnom = bnom[binds]
bnom = bnom[np.argsort(bnom[:,0])]
bpet = bpet[binds]
bpet = bpet[np.argsort(bpet[:,0])]


dnom = dnom[dinds]
dnom = dnom[np.argsort(dnom[:,0])]
dpet = dpet[np.argsort(dpet[:,0])]

###########################
#
# plt.plot(*bnom.T, label='Nom')
# # plt.plot(*dnom.T, '--')
# plt.plot(*bpet.T, label='BDW')
# plt.plot(*dpet.T, '--')
# plt.legend()

###########################

dpr = np.hypot(*dpet.T)
bpr = np.hypot(*bpet.T)
dnr = np.hypot(*dnew.T)

# bpy = np.interp(dpet[:,0], bpet[:,0], bpet[:,1])
# bpyn = np.interp(dnew[:,0], bpet[:,0], bpet[:,1])
bpxn = np.interp(dnr, bpr, bpet[:,0])
bpyn = np.interp(dnr, bpr, bpet[:,1])

plt.figure()
plt.plot(bpxn - dnew[:,0])
plt.plot(bpyn - dnew[:,1])
# plt.axis('equal')

plt.figure()
plt.plot(dnew[:,0], bpyn, 'x-')
plt.plot(*dnew.T, '+-')

print(abs(bpyn - dnew[:,1]).max(), abs(bpyn - dnew[:,1]).mean())


breakpoint()
