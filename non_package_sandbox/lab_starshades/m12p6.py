import diffraq
import numpy as np

wave = 0.641e-6
aperture_diameter = 5e-3
z1 = 50.
z0 = 27.455
focal_length = 0.498
num_pts = 256
num_foc = 128

mm = 1000
nn = 200

#perturbations
perts = [
    ['shiftedPetal', {'petal':1, 'shift':2.500e-6, }],
]

breakpoint()
params = {

    ### World ###
    'num_pts':          num_pts,
    'zz':               z1,
    'z0':               z0,
    'tel_diameter':     aperture_diameter,
    'waves':            wave,
    'focal_length':     focal_length,
    'image_size':       num_foc,

    ### Numerics ###
    'radial_nodes':     mm,
    'theta_nodes':      nn,

    ### Occulter ###
    'occulter_shape':   'starshade',
    'is_babinet':       False,
    'num_petals':       12,
    'apod_file':        f'{diffraq.apod_dir}/bb_2017.txt',
    'perturbations':    perts,

    ### Saving ###
    'do_save':          True,
    'save_dir_base':    f'{diffraq.results_dir}/diffraq_test',
    'session':          'M12P6',
    'save_ext':         f'm{mm}__n{nn}',
}

sim = diffraq.Simulator(params)
# sim.run_sim()

# sim.occulter.build_quadrature()
occ = sim.occulter.get_edge_points()

import matplotlib.pyplot as plt;plt.ion()
plt.plot(*occ.T, 'x')
plt.plot(*xyi, 'o')
plt.plot(*xyo, 's')

breakpoint()
