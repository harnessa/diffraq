import diffraq
import numpy as np

wave = 0.641e-6
aperture_diameter = 5e-3
z1 = 50.
z0 = 27.455
focal_length = 0.498
num_pts = 256
num_foc = 128

mm = 2000
nn = 100

#perturbations
xyi = np.array([-2.576, -11.517]) * 1e-3
xyo = np.array([ 2.915,  20.752]) * 1e-3
perts = [
    ['notch', {'xy0':xyi, 'height':2.500e-6, 'width':413.817e-6, 'direction':1, 'local_norm':True}], #Inner
    ['notch', {'xy0':xyo, 'height':1.726e-6, 'width':530.944e-6, 'direction':1, 'local_norm':True}]  #Outer
]

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
    'session':          'M12P2',
    'save_ext':         f'm{mm}__n{nn}__2',
}

sim = diffraq.Simulator(params)
sim.run_sim()

# # sim.occulter.build_quadrature()
# occ = sim.occulter.get_edge_points()
#
# import matplotlib.pyplot as plt;plt.ion()
# plt.plot(*occ.T, 'x')
# plt.plot(*xyi, 'o')
# plt.plot(*xyo, 's')

# breakpoint()
