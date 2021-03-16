import diffraq
import numpy as np
import matplotlib.pyplot as plt;plt.ion()
from scipy.integrate import simps

ldir = '/home/aharness/repos/Milestone_2/analysis/modeling/M12P2/make_loci/Loci'
inn = np.genfromtxt(f'{ldir}/petal4_notch12p.txt', delimiter='  ')
out = np.genfromtxt(f'{ldir}/petal10_notch12p.txt', delimiter='  ')
nom = np.genfromtxt(f'{ldir}/nompetal12p.txt', delimiter='  ')

#Split to halves
inn = inn[:len(inn)//2-150]
out = out[:len(out)//2-150]
nom = nom[:len(nom)//2-150]

plt.plot(np.hypot(*nom.T), np.hypot(*(out - nom).T))
plt.plot(np.hypot(*nom.T), np.hypot(*(inn - nom).T))

breakpoint()
# Get areas
# inn_aa = np.trapz(*inn.T)
# out_aa = np.trapz(*out.T)
# nom_aa = np.trapz(*nom.T)

# inn_aa = simps(*inn[:,::-1].T)
# out_aa = simps(*out[:,::-1].T)
# nom_aa = simps(*nom[:,::-1].T)

# inn_da = np.abs(inn_aa - nom_aa)
# out_da = np.abs(out_aa - nom_aa)

indi = np.hypot(*(inn - nom).T) > 0
indo = np.hypot(*(out - nom).T) > 0

inn_da = np.abs( simps(*inn[indi][:,::-1].T) - simps(*nom[indi][:,::-1].T))
out_da = np.abs( simps(*out[indo][:,::-1].T) - simps(*nom[indo][:,::-1].T))

#Inner Perturbations
xyi = np.array([-2.5772, -11.5262]) * 1e-3
pert_inn = {'kind':'notch', 'xy0':xyi, 'height':2.47e-6, 'width':404e-6, \
    'direction':1, 'local_norm':True, 'num_quad':200}

#Outer Perturbations
xyo = np.array([ 2.9154, 20.7524]) * 1e-3
pert_out = {'kind':'notch', 'xy0':xyo, 'height':1.71e-6, 'width':516e-6, \
    'direction':1, 'local_norm':True, 'num_quad':200}

#Petal shape
num_pet = 12
apod_dir = '/home/aharness/repos/diffraq/External_Data/Apodization_Profiles'

starshade = {'kind':'starshade', 'is_opaque':False, 'num_petals':num_pet, \
    'edge_file':f'{apod_dir}/bb_2017.txt', 'has_center':False}

#Build simulator
sim = diffraq.Simulator( {'radial_nodes':6000, 'theta_nodes':300}, starshade)

#Build perturbations
pinn = diffraq.geometry.Notch(sim.occulter.shapes[0], **pert_inn)
ixp, iyp, iwp = pinn.get_quadrature()
ipa = np.abs(iwp.sum())

pout = diffraq.geometry.Notch(sim.occulter.shapes[0], **pert_out)
oxp, oyp, owp = pout.get_quadrature()
opa = np.abs(owp.sum())

#Rotate
def rot_mat(angle):
    return np.array([[ np.cos(angle), np.sin(angle)],
                     [-np.sin(angle), np.cos(angle)]])

ixp, iyp = np.stack((ixp, iyp), 1).dot(rot_mat(2*np.pi/num_pet*3)).T
oxp, oyp = np.stack((oxp, oyp), 1).dot(rot_mat(2*np.pi/num_pet*-3)).T
iyp *= -1
oyp *= -1

# plt.plot(*inn.T, 'x-')
# plt.plot(*out.T, '+-')
# # plt.plot(*nom.T, '+')
# plt.plot(ixp, iyp, '*')
# plt.plot(oxp, oyp, '*')
# plt.axis('equal')

print(f'\nTarget Inn: {inn_da:.3e}\t Calc Inn: {ipa:.3e}')
print(  f'Target Out: {out_da:.3e}\t Calc Out: {opa:.3e}\n')


breakpoint()
