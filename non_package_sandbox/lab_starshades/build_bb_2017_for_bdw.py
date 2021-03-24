import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import diffraq
from utilities import util

num_pet = 12
max_apod = 0.9
etch = -1e-6
apod_dir = '/home/aharness/repos/diffraq/External_Data/Apodization_Profiles'

sgn = {1:'p', -1:'n'}[np.sign(etch)]
ext = f'__etch_{sgn}{abs(etch*1e6):.0f}'
out_file = f'bb_2017{ext}'

def rot_mat(angle):
    return np.array([[ np.cos(angle), np.sin(angle)],
                     [-np.sin(angle), np.cos(angle)]])

#Load data
rads, apod = np.genfromtxt(f'{apod_dir}/bb_2017.txt', delimiter=',', unpack=True)

#Convert to angles
apod *= np.pi/num_pet

#Convert to cartesian
xy = rads[:,None] * np.stack((np.cos(apod), np.sin(apod)), 1)

#Get normals
normal = util.compute_derivative(xy)
normal = np.vstack((-normal[:,1], normal[:,0])).T
normal /= np.hypot(*normal.T)[:,None]

#Add etch
xy += normal*etch

#Flip and build around y
xy = np.concatenate((xy, xy[::-1]*np.array([1, -1])))

#Loop through and build each petal
mask = []
for i in range(num_pet):

    #Current petal
    cur_pet = xy.copy().dot(rot_mat(i*2.*np.pi/num_pet))

    #append
    mask.append(cur_pet)

mask = np.array(mask)

plt.cla()
for i in range(len(mask)):
    plt.plot(*mask[i].T)

if [False, True][0]:
    with open(f'./{out_file}.dat', 'w') as f:
        for i in range(len(mask)):
            for j in range(len(mask[i])):
                f.write(f'{mask[i][j][0]}, {mask[i][j][1]}\n')
            f.write('**,**\n')


breakpoint()
