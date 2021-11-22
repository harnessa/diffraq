import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import diffraq
import h5py
import time

waves = np.array([641, 660, 699, 725]) * 1e-9

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

z10 = 167.84e-3
z20 = 59.85e-3 - 20e-3

DZ = 20e-3
nz = 11
dzs = np.linspace(-DZ, DZ, nz + ((nz+1)%2))

maxs = np.zeros((nz,nz, len(waves)))

tik = time.perf_counter()

#Loop over distances
for i1 in range(nz):
    print(f'Running {i1}/{nz}')
    for i2 in range(nz):

        #Add new distance
        lens_system['element_0']['distance'] = z10 + dzs[i1]
        lens_system['element_1']['distance'] = z20 + dzs[i2]

        #Set new lenses
        sim.focuser.lenses = diffraq.diffraction.Lens_System(lens_system, sim.focuser)

        #Get images
        image, image_pts = sim.focuser.calculate_image(pupil, grid_pts)

        #Store max
        maxs[i1,i2] = image.max((1,2))
        breakpoint()

tok = time.perf_counter()
print(f'Time: {tok-tik:.2f}')

with h5py.File('./saves/lab_focus4.h5', 'w') as f:
    f.create_dataset('dzs', data=dzs)
    f.create_dataset('z10', data=z10)
    f.create_dataset('z20', data=z20)
    f.create_dataset('maxs', data=maxs)
