import diffraq
import numpy as np
import matplotlib.pyplot as plt;plt.ion()

def get_dif_edge(apod):

    params = {
        'radial_nodes':     1000,
        'theta_nodes':      20,
        'occulter_config':   f'{diffraq.occulter_dir}/{apod}.cfg',
    }

    #Load simulator + build edge
    sim = diffraq.Simulator(params)
    sim.occulter.build_edge()

    #Get edge
    edge = sim.occulter.edge.copy()

    #Cleanup
    sim.clean_up()

    return edge, sim

def get_bdw_edge(apod):
    edges = np.genfromtxt(f'./xtras/{apod}.dat', delimiter=',', comments='%')
    edges = edges[~np.isnan(edges[:,0])]
    return edges

apod = 'M12P6'

dedg, sim = get_dif_edge(apod)
bedg = get_bdw_edge(apod)

plt.plot(*bedg.T, '-')
plt.plot(*dedg.T, '+')

if apod == 'M12P6':

    the = np.linspace(0,2*np.pi,1000)
    r0 = np.hypot(*bedg.T).min()
    rf = sim.occulter.shapes[0].max_radius
    plt.plot(r0*np.sin(the), r0*np.cos(the), 'k')
    plt.plot(rf*np.sin(the), rf*np.cos(the), 'k')
    plt.plot((rf+7.5e-6)*np.sin(the), (rf+7.5e-6)*np.cos(the), 'k')

    pet_ang = np.pi/12
    for i in [1, 5, 9]:
        angs = pet_ang * np.array([2*(i-1), 2*i])
        plt.plot([0,3*r0*np.cos(angs.mean())], [0,3*r0*np.sin(angs.mean())])
        plt.plot([0,3*r0*np.cos(angs[0])], [0,3*r0*np.sin(angs[0])], '--')
        plt.plot([0,3*r0*np.cos(angs[1])], [0,3*r0*np.sin(angs[1])], '--')

# plt.xlim([12.189e-3,12.195e-3])
# plt.ylim([2.923e-3,2.928e-3])
plt.xlim([-8.631e-3, -8.6296e-3])
plt.ylim([9.0936e-3, 9.095e-3])
breakpoint()
