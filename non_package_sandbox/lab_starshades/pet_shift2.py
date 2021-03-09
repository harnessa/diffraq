import numpy as np
import matplotlib.pyplot as plt;plt.ion()

is_big = [False, True][0]

ldir = '/home/aharness/repos/Milestone_2/analysis/modeling/M12P6/make_loci/Loci'
pet0 = np.genfromtxt(f'{ldir}/petal_3.txt', delimiter='  ')

if is_big:
    pet1 = np.genfromtxt(f'{ldir}/petal_1.txt', delimiter='  ')
    pet2 = np.genfromtxt(f'{ldir}/petal_2.txt', delimiter='  ')
else:
    pet1 = np.genfromtxt(f'{ldir}/petal_6.txt', delimiter='  ')
    pet2 = np.genfromtxt(f'{ldir}/petal_7.txt', delimiter='  ')


pet_ang = 2*np.pi/12.

def rot_pet(pet, ang):
    newx =  pet[:,0]*np.cos(ang) + pet[:,1]*np.sin(ang)
    newy = -pet[:,0]*np.sin(ang) + pet[:,1]*np.cos(ang)
    new = np.stack((newx, newy),1)
    return new

if not is_big:
    pet1 = rot_pet(pet1, -5*pet_ang)
    pet2 = rot_pet(pet2, -5*pet_ang)

pet0 = rot_pet(pet0, -2*pet_ang)
pet0b = rot_pet(pet0.copy(), pet_ang)
shift = pet0[0][1] - pet1[0][1]
shift = 11e-6

plt.plot(*pet0[0],'rs')
plt.plot(*pet1[0],'rd')
print(shift*1e6)

# pet0 = rot_pet(pet0, -pet_ang/2)
# pet0b = rot_pet(pet0b, -pet_ang/2)
# pet1 = rot_pet(pet1, -pet_ang/2)
# pet2 = rot_pet(pet2, -pet_ang/2)


rad_cut = 0.012530074710099998
rad_min = np.hypot(*pet0.T).min()


plt.plot(*pet0.T, 'g+--')
# plt.plot(*pet0b.T, 'g+--')
plt.plot(*pet1.T, 'x-')
# plt.plot(*pet2.T, 'x-')

# plt.plot(pet0[:,0], pet0[:,1] - shift,'r:')
# plt.plot(pet0b[:,0], pet0b[:,1] - shift,'r:')

t0 = 3*np.pi/2.
dt = 2*np.pi/12. * 2
the = np.linspace(t0 - dt, t0+dt,5000)
plt.plot(rad_min*np.cos(the), rad_min* np.sin(the),'k--')
plt.plot((rad_min + shift)*np.cos(the), (rad_min + shift)* np.sin(the),'k')
plt.plot(rad_cut*np.cos(the), rad_cut* np.sin(the),'k--')
plt.plot((rad_cut + shift)*np.cos(the), (rad_cut + shift)* np.sin(the),'k')
plt.axis('equal')
breakpoint()
