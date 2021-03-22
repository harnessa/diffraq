import diffraq
import numpy as np
import matplotlib.pyplot as plt;plt.ion()

def get_dif_edge(apod):

    params = {
        'radial_nodes':     1000,
        'theta_nodes':      20,
        # 'occulter_config':   f'{diffraq.occulter_dir}/{apod}.cfg',
    }

    num_pet = 12
    etch = 1e-6
    apod_dir = '/home/aharness/repos/diffraq/External_Data/Apodization_Profiles'
    shape = {'kind':'starshade', 'is_opaque':False, 'num_petals':num_pet, \
        'edge_file':f'{apod_dir}/{apod}.txt', 'has_center':False, 'etch_error':etch}


    #Load simulator + build edge
    sim = diffraq.Simulator(params, shape)
    sim.occulter.build_edge()

    #Get edge
    edge = sim.occulter.edge.copy()

    #Cleanup
    sim.clean_up()

    return edge, sim

bdw_ext = '_etch_p1'
def get_bdw_edge(apod):
    edges = np.genfromtxt(f'./xtras/{apod}_{bdw_ext}.dat', delimiter=',', comments='%')
    edges = edges[~np.isnan(edges[:,0])]
    angle = -2*np.pi/12
    # edges = edges.dot( np.array([[ np.cos(angle), np.sin(angle)],
                                 # [-np.sin(angle), np.cos(angle)]]) )
    return edges

apod = 'bb_2017'

dedg, sim = get_dif_edge(apod)
bedg = get_bdw_edge(apod)

plt.plot(*bedg.T, 'x-')
plt.plot(*dedg.T, '+--')

# sim.occulter.build_quadrature()
# plt.plot(sim.occulter.xq, sim.occulter.yq, '*')

if apod.startswith('M12P6'):

    the = np.linspace(0,2*np.pi,10000)
    r0 = np.hypot(*bedg.T).min()
    rf = sim.occulter.shapes[0].max_radius
    plt.plot(r0*np.cos(the), r0*np.sin(the), 'k:')
    plt.plot(rf*np.cos(the), rf*np.sin(the), 'k--')
    plt.plot(r0*np.cos(the) + 7.5e-6*np.cos(np.pi/12), r0*np.sin(the) + 7.5e-6*np.sin(np.pi/12), 'k-.')
    plt.plot(rf*np.cos(the) + 7.5e-6*np.cos(np.pi/12), rf*np.sin(the) + 7.5e-6*np.sin(np.pi/12), 'k-.')

    pet_ang = np.pi/12
    for i in [1, 5, 9]:
        angs = pet_ang * np.array([2*(i-1), 2*i])
        plt.plot([0,3*r0*np.cos(angs.mean())], [0,3*r0*np.sin(angs.mean())])
        plt.plot([0,3*r0*np.cos(angs[0])], [0,3*r0*np.sin(angs[0])], '--')
        plt.plot([0,3*r0*np.cos(angs[1])], [0,3*r0*np.sin(angs[1])], '--')

plt.axis('equal')
# plt.xlim([8.256e-3, 8.268e-3])
# plt.ylim([-2e-6, 2e-6])

# plt.xlim([12.189e-3,12.195e-3])
# plt.ylim([2.923e-3,2.928e-3])

# plt.xlim([-8.631e-3, -8.6296e-3])
# plt.ylim([9.0936e-3, 9.095e-3])
breakpoint()
