import diffraq
import numpy as np
import matplotlib.pyplot as plt;plt.ion()
from bdw import BDW

num_occ_pts = 500

params = {
    'radial_nodes':     num_occ_pts,
    'theta_nodes':      20,
}

num_pet = 22
apod_dir = '/home/aharness/repos/diffraq/External_Data/Apodization_Profiles'
shape = {'kind':'starshade', 'is_opaque':True, 'num_petals':num_pet, \
    'edge_file':f'{apod_dir}/wfirst_ni2.h5', 'has_center':True, }

#Load simulator + build edge
sim = diffraq.Simulator(params, shape)
sim.occulter.build_edge()
# sim.occulter.build_quadrature()

#Get edge
dd = sim.occulter.edge.copy()

#Cleanup
# sim.clean_up()

#BDW
bdw_params = {
        'num_petals':       num_pet,
        'num_occ_pts':      num_occ_pts,
        'apod_name':        'wfirst',
        'is_connected':     True,
}
bdw = BDW(bdw_params)
bb = bdw.loci.copy()

#Roll to match
bb = np.roll(bb, -np.argmin(np.hypot(*(bb - dd[0]).T)), axis=0)

#Plot
plt.plot(*bb.T, 'x-')
plt.plot(*dd.T, '+')

plt.axis('equal')

diff = np.hypot(*(dd - bb).T)
plt.figure()
plt.plot(diff,'x')


breakpoint()
