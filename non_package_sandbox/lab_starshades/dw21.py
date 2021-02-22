import diffraq
import numpy as np

wave = 0.641e-6
aperture_diameter = 5e-3
z1 = 50.
z0 = 27.455
focal_length = 0.498
num_pts = 256
num_foc = 128

mm = 4000
nn = 200

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
    'occulter_file':   f'{diffraq.occulter_dir}/DW21.cfg',

    ### Saving ###
    'do_save':          True,
    'save_dir_base':    f'{diffraq.results_dir}/diffraq_test',
    'session':          'DW21',
    'save_ext':         f'm{mm}__n{nn}',
}

sim = diffraq.Simulator(params)
sim.run_sim()

# # sim.occulter.build_quadrature()
# sim.occulter.build_edge()
#
# import matplotlib.pyplot as plt;plt.ion()
# plt.plot(*sim.occulter.edge.T, 'x')
#
# breakpoint()
