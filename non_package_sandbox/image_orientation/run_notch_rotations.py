import numpy as np
import diffraq
import matplotlib.pyplot as plt;plt.ion()

params = {

    'waves':            0.6e-6,
    'tel_diameter':     5e-3,
    'num_pts':          256,

    'zz':               50,
    'z0':               27.5,
    'focal_length':     0.498,
    'image_size':       128,

    'radial_nodes':     1000,
    'theta_nodes':      100,

    'do_save':          False,
    'free_on_end':      False,
    'verbose':          False,
}

circle_rad = 12e-3
height = 10e-3
width = 5e-3
cen_ang = width/2/circle_rad        #angle to move to center on defect
nrun = 12

fig, axes = plt.subplots(1,2,figsize=(9,9))
for i in range(nrun)[:]:

    notch_ang = 2*np.pi/nrun * i

    xy0 = circle_rad*np.array([np.cos(notch_ang + cen_ang), np.sin(notch_ang + cen_ang)])

    notch = {'kind':'notch', 'xy0':xy0, 'height':height, 'width':width, \
        'local_norm':False, 'direction':1, 'num_quad':50}

    #Build shape
    shapes = {'kind':'circle', 'max_radius':circle_rad, \
        'is_opaque':True, 'perturbations':notch}

    #Build simulator
    sim = diffraq.Simulator(params, shapes)
    sim.run_sim()
    sim.occulter.build_edge()

    axes[0].cla()
    axes[1].cla()
    axes[0].plot(*sim.occulter.edge.T,'x')
    axes[0].plot(*xy0, 'ro')
    axes[0].plot([0,circle_rad*np.cos(notch_ang)],[0,circle_rad*np.sin(notch_ang)], 'r')
    axes[0].plot([0,xy0[0]],[0,xy0[1]], 'k')
    axes[0].axis('equal')
    axes[1].imshow(sim.image[0])
    # axes[1].imshow(abs(sim.pupil[0]))
    fig.suptitle(f'Angle: {np.degrees(notch_ang):.0f}')

    sim.clean_up()


    breakpoint()
