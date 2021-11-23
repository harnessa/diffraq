import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import diffraq
import h5py

waves = np.array([0.641e-6])

telD = 5e-3
focal_length = 0.5

z1 = 167.84e-3 + 0.5e-3
z2 = 59.85e-3 - 10e-3

lens_system = {
    'element_0': {'kind':'lens', 'lens_name':'AC508-150-A-ML', \
        'distance':z1, 'diameter':telD},
    'element_1': {'kind':'lens', 'lens_name':'AC064-015-A-ML', \
        'distance':z2, 'diameter':telD/2},
}

#Add wfe
ax1, ay1 = 0, 0
ax2, ay2 = 0, 0

fig, axes = plt.subplots(2, figsize=(5,8))

# for ax2 in np.logspace(-8+2, -6+2, 5):
for ax1 in [5e-7*0]:

    lens_system['element_0']['wfe_modes'] = [[-2,2,ax1], [2,2,ay1]]
    lens_system['element_1']['wfe_modes'] = [[-2,2,ax2], [2,2,ay2]]

    #Build simulator
    sim = diffraq.Simulator({'num_pts':512, 'waves':waves, 'tel_diameter':telD, \
        'image_size':74, 'focal_length':focal_length, 'lens_system':lens_system})
    sim.load_focuser()

    #Build uniform pupil image
    grid_pts = diffraq.utils.image_util.get_grid_points(sim.num_pts, sim.tel_diameter)
    pupil = np.ones((len(waves), sim.num_pts, sim.num_pts)) + 0j
    with h5py.File(f'/home/aharness/repos/Milestone_2/diffraq_analysis/final_run/Diffraq_Results__11_2_21/Mcal/pupil.h5', 'r') as f:
        pupil = f['field'][()]

    #Get images
    image, image_pts = sim.focuser.calculate_image(pupil, grid_pts)
    image = image[0]

    # plt.figure()
    # plt.imshow(np.log10(image))
    # breakpoint()

    axes[0].semilogy(image_pts, image[len(image)//2])
    axes[1].semilogy(image_pts, image[:,len(image)//2])
# axes[0].legend()
breakpoint()
