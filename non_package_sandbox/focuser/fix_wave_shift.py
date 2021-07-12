import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import diffraq
import h5py

waves = np.arange(635, 645, 1) * 1e-9

params = {

    ### World ###
    'num_pts':              256,
    'waves':                waves,
    'zz':                   49.986,
    'z0':                   27.455,
    'tel_diameter':         5e-3,
    'focal_length':         0.496,
    'defocus':              8e-4,
    'image_size':           74,

    ### Numerics ###
    'radial_nodes':         2000,
    'theta_nodes':          100,

    ### Saving ###
    'do_save':              False,
    'verbose':              False,
    'free_on_end':          False,

}

shapes = {'kind':'circle', 'is_opaque':True, 'max_radius':25.086e-3}

#Calculate pupil
sim = diffraq.Simulator(params, shapes)
pupil, grid_pts = sim.calc_pupil_field()

#Calculate image
sim.load_focuser()
image, image_pts = sim.focuser.calculate_image(pupil)

for i in range(image.shape[0]):
    true = sim.focuser.true_NN[i]
    targ = sim.focuser.targ_NN[i]

    print(f'{waves[i]*1e9:.0f}, {true}, {targ}, {true/targ}')
    plt.cla()
    plt.plot(image.shape[-1]/2, image.shape[-1]/2, 'r+')
    plt.imshow(image[i])
    breakpoint()

breakpoint()
