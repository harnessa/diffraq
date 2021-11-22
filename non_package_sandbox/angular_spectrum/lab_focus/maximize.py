import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import diffraq
import h5py
import time
from scipy.optimize import minimize

waves = np.array([641]) * 1e-9

telD = 5e-3
focal_length = 0.5

lens_system = {
    'element_0': {'kind':'lens', 'lens_name':'AC508-150-A-ML', \
        'distance':167.84e-3, 'diameter':telD},
    'element_1': {'kind':'lens', 'lens_name':'AC064-015-A-ML', \
        'distance':59.85e-3},
}

#Build simulator
sim = diffraq.Simulator({'num_pts':512, 'waves':waves, 'tel_diameter':telD, \
    'image_size':74, 'focal_length':focal_length, 'lens_system':lens_system})
sim.load_focuser()

#Build uniform pupil image
grid_pts = diffraq.utils.image_util.get_grid_points(sim.num_pts, sim.tel_diameter)
rr = np.hypot(grid_pts, grid_pts[:,None])
pupil = np.exp(1j*2.*np.pi/waves[:,None,None]*rr**2/(2*77.455))

z10 = 167.84e-3 + 0.5e-3
z20 = 59.85e-3 - 10e-3

def model(p0):

    lens_system['element_0']['distance'] = z10 + p0[0]
    lens_system['element_1']['distance'] = z20 + p0[1]

    #Set new lenses
    sim.focuser.lenses = diffraq.diffraction.Lens_System(lens_system, sim.focuser)

    #Get images
    image, image_pts = sim.focuser.calculate_image(pupil, grid_pts)

    return image

func = lambda p0: 1/model(p0).max()

out = minimize(func, [0, 0])#, constraints=con, method='SLSQP')

print(out.success)

z1 = z10 + out.x[0]
z2 = z20 + out.x[1]

img = model(out.x)[0]

plt.imshow(img)

breakpoint()
