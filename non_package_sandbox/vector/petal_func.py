import numpy as np
import matplotlib.pyplot as plt;plt.ion()

num_pet = 12
r0 = 8
r1 = 16

m = -r0*np.tan(np.pi/num_pet) / (r1 - r0)

#Petal function
# func = lambda r: np.arctan(m)*np.ones_like(r)
func = lambda r: np.arctan(r0*np.sin(np.pi/num_pet)/(r1-r0*np.cos(np.pi/num_pet)) * \
    (r1/r - 1)) * num_pet/np.pi

rr = np.linspace(r0, r1, 20)

tt = func(rr)

xy = rr[:,None] * np.stack((np.cos(tt), np.sin(tt)),1)
xy = np.concatenate((xy, xy*np.array([1,-1])))


rot_mat = lambda ang: np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])

for i in range(num_pet):
    plt.plot(*xy.dot(rot_mat(2*np.pi/num_pet*i)).T, 'x')

    breakpoint()
