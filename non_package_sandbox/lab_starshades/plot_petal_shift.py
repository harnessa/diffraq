import numpy as np
import matplotlib.pyplot as plt;plt.ion()

ldir = '/home/aharness/repos/Milestone_2/analysis/modeling/M12P6/make_loci/Loci'
nom = np.genfromtxt(f'{ldir}/petal_4.txt', delimiter='  ')
pet_big = np.genfromtxt(f'{ldir}/petal_1.txt', delimiter='  ')
pet_sml = np.genfromtxt(f'{ldir}/petal_10.txt', delimiter='  ')

def rot_pet(pet, ang):
    newx =  pet[:,0]*np.cos(ang) + pet[:,1]*np.sin(ang)
    newy = -pet[:,0]*np.sin(ang) + pet[:,1]*np.cos(ang)
    new = np.stack((newx, newy),1)
    return new

def shift(pet, angles, shft_dir, shift, rad_inn):
    angs = np.arctan2(pet[:,1], pet[:,0]) % (2.*np.pi)
    rads = np.hypot(*pet.T)

    # ang_inds = (angs >= angles[0]) & (angs <= angles[1])
    # good_half = pet[~ang_inds]
    # splt_half = pet[ang_inds]
    #
    # rads = np.hypot(*splt_half.T)
    # inn_half = splt_half[rads <=  rad_inn]
    # out_half = splt_half[rads > rad_inn]
    #
    # inn_half += shft_dir * shift
    #
    # out_rads = np.hypot(*out_half.T)
    #
    # # out_half = out_half[out_rads > rad_inn + shift]
    #
    # new = np.concatenate((good_half, inn_half, out_half))
    #
    # angs = np.arctan2(new[:,1]-new[:,1].mean(), new[:,0]-new[:,0].mean())
    # new = new[np.argsort(angs)]

    inds = (angs >= angles[0]) & (angs <= angles[1])
    inds = inds & (rads <= rad_inn)
    pet[inds] += shft_dir * shift
    bad_inds = (angs >= angles[0]) & (angs <= angles[1]) & \
        (rads <= rad_inn + shift) & (rads >= rad_inn)
    print(np.hypot(*pet[bad_inds].T).max())
    plt.plot(*pet[bad_inds].T, 'o')
    pet = pet[~bad_inds]

    return pet

nom = rot_pet(nom, -2.*np.pi/12*6)
pet_big = rot_pet(pet_big, -2.*np.pi/12*3)


#apod
apod = np.arctan2(nom[nom[:,1] >= 0][:,1], nom[nom[:,1] >= 0][:,0]) * 12/np.pi
rads = np.hypot(*nom[nom[:,1] >= 0].T)
# rad_inn = rads[apod >= 0.9].min()
# rad_inn = 12.5312e-3
rad_inn = 0.012530074710099998

big_dir = np.array([np.cos(-np.pi/12), np.sin(-np.pi/12)])
big_shift = 10.5e-6

sml_dir = np.array([np.cos(np.pi/12), np.sin(np.pi/12)])
sml_shift = 7.5e-6

new_big = nom.copy()
new_big = shift(new_big, [2*np.pi*(1. - 1/12), 2*np.pi], big_dir, big_shift, rad_inn)


# new_sml = nom.copy()
# new_sml = shift(new_sml, [0, 2*np.pi/12], sml_dir, sml_shift, rad_inn)


plt.plot(*nom.T)
plt.plot(*pet_big.T,'x-')
plt.plot(*new_big.T, '+--')
# plt.plot(*pet_sml.T)
# plt.plot(*new_sml.T, '--')
# plt.xlim([8.23e-3, 8.3e-3])
# plt.ylim([-3e-5, 3e-5])

the = np.linspace(0,2*np.pi,1000)
plt.plot(rad_inn*np.cos(the), rad_inn* np.sin(the),'k')
plt.plot((rad_inn + big_shift)*np.cos(the), (rad_inn + big_shift)* np.sin(the),'k')
plt.xlim([12.185e-3, 12.2e-3])
plt.ylim([-2.9290e-3, -2.9250e-3])

breakpoint()
