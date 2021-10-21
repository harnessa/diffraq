import diffraq
import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import h5py

apod = ['M12P2', 'M12P6', 'M12P7', 'M12P8', 'M12P9', 'DW9'][1]

rn, tn = 200, 20
seam = 25e-6
sr, st = 80*3, 5

params = {
    'radial_nodes':     rn,
    'theta_nodes':      tn,
    'occulter_config':   f'./plain_configs/{apod}.cfg',
    'seam_radial_nodes':    sr,
    'seam_theta_nodes':     st,
    'seam_width':           seam,
    'do_run_vector':        True,
    'spin_angle':  0,
}

#Load simulator + build edge
sim = diffraq.Simulator(params)

#Get seam quad
sim.vector.build_quadrature()
xq, yq, wq, dq, nq = sim.vector.xq, sim.vector.yq, sim.vector.wq, sim.vector.dq, sim.vector.nq

# #Get regular quad
# sim.occulter.build_quadrature()
# oxq, oyq, owq = sim.occulter.xq.copy(), sim.occulter.yq.copy(), sim.occulter.wq.copy()

#get edge
sim.occulter.build_edge()
edge2 = sim.occulter.edge.copy()
sim.occulter.clean_up()

def get_bdw_edge(apod):
    loci_dir = '/home/aharness/repos/Milestone_2/diffraq_analysis/modeling/BDW_Compare/bdw_loci'
    deci = '4x'
    data = []
    with h5py.File(f'{loci_dir}/{apod}_{deci}.h5', 'r') as f:
        num_pet = f['num_petals'][()]
        for i in range(num_pet):
            data.append(sim.occulter.add_occulter_attitude(f[f'loci_{i}'][()]))

    return data
edge = get_bdw_edge(apod)

#Plot
# plt.colorbar(plt.scatter(xq, yq, c=np.degrees(nq), s=2, cmap=plt.cm.jet))
plt.colorbar(plt.scatter(xq, yq, c=dq, s=2, cmap=plt.cm.jet))

# for pt in sim.vector.seams[0].pert_list:
#     plt.plot(*pt.xy0, 'd')

# plt.plot(oxq, oyq, 'x')
# plt.plot(*edge2.T, '+')
for i in range(len(edge)):
    plt.plot(*edge[i].T)

plt.axis('equal')
print(wq.sum())
breakpoint()
