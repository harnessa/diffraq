import diffraq
import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import dial2

apod = ['M12P2', 'M12P6', 'M12P7', 'M12P8','DW9'][0]

def get_dif_edge(apod):

    rn, tn = 400, 20
    seam = 10e-6
    sr, st = 20, 20

    params = {
        'radial_nodes':     rn,
        'theta_nodes':      tn,
        'occulter_config':   f'./plain_configs/{apod}.cfg',
        'seam_radial_nodes':    sr,
        'seam_theta_nodes':     st,
        'seam_width':           seam,
        'do_run_vector':        True,
    }

    #Load simulator + build edge
    sim = diffraq.Simulator(params)

    xq, yq, wq, dq, nq, gw, n_nodes = sim.vector.seams[0].build_seam_quadrature(seam)

    return xq, yq, wq, dq, nq, gw, n_nodes, sim

xq, yq, wq, dq, nq, gw, n_nodes, sim = get_dif_edge(apod)

plt.colorbar(plt.scatter(xq, yq, c=wq, s=1))

for pt in sim.vector.seams[0].pert_list:
    plt.plot(*pt.xy0, 'd')

plt.axis('equal')
breakpoint()
