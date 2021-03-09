import diffraq
import numpy as np
import matplotlib.pyplot as plt;plt.ion()

def get_edge(apod):

    params = {
        'radial_nodes':     40,
        'theta_nodes':      20,
        'occulter_config':   f'{diffraq.occulter_dir}/{apod}.cfg',
    }

    #Load simulator + build edge
    sim = diffraq.Simulator(params)
    sim.occulter.build_quadrature()

    #Get edge
    xq = sim.occulter.xq.copy()
    yq = sim.occulter.yq.copy()
    wq = sim.occulter.wq.copy()

    #Cleanup
    sim.clean_up()

    return xq, yq, wq

apod = ['M12P2', 'bb_2017_circle'][1]

sx, sy, sw = get_edge(apod)
print(sx.size)

plt.scatter(sx, sy, c=sw, s=1)

breakpoint()
