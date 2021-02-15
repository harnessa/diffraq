import diffraq
import numpy as np

wave = 0.641e-6
aperture_diameter = 5e-3
z1 = 50.
z0 = 27.455
focal_length = 0.498
num_pts = 256
num_foc = 128//2
num_pet = 12

mm = 2000
nn = 200

#perturbations
pet_ang = np.pi/num_pet
pet1_dir = np.array([np.cos(pet_ang), np.sin(pet_ang)])

perts = [
    ['ShiftedPetal', {'angles':pet_ang*np.array([0,2]), 'shift':10e-6, \
        'direction':pet1_dir, 'max_radius':12.5312e-3}],
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
    'num_petals':       num_pet,
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

sim.occulter.build_edge()

import matplotlib.pyplot as plt;plt.ion()
plt.plot(*sim.occulter.edge.T, 'x')

breakpoint()
