import diffraq
import numpy as np
import matplotlib.pyplot as plt;plt.ion()

def get_edge(apod):

    params = {
        'radial_nodes':     4000,
        'theta_nodes':      200,
        'occulter_config':   f'{diffraq.occulter_dir}/{apod}.cfg',
    }

    #Load simulator + build edge
    sim = diffraq.Simulator(params)
    sim.occulter.build_edge()

    #Get edge
    edge = sim.occulter.edge.copy()

    #Cleanup
    sim.clean_up()

    return edge

apod = ['M12P2', 'bb_2017'][1]

sedg = get_edge(apod)
# jedg = get_edge(f'{apod}_joint')
jedg = get_edge(f'{apod}_strt')


if [False, True][1]:
    plt.plot(*jedg.T, 'x')
    plt.plot(*sedg.T, '+')
else:
    plt.plot(*jedg.T, '-')
    plt.plot(*sedg.T, '--')

print(len(sedg), len(jedg))
breakpoint()
