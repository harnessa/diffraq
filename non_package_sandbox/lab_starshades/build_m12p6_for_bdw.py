import numpy as np
import matplotlib.pyplot as plt;plt.ion()

num_pet = 12
max_apod = 0.9
apod_dir = '/home/aharness/repos/diffraq/External_Data/Apodization_Profiles'

out_file = 'M12P6_new'

def rot_mat(angle):
    return np.array([[ np.cos(angle), np.sin(angle)],
                     [-np.sin(angle), np.cos(angle)]])

#Load data
inn_rad, inn_apd = np.genfromtxt(f'{apod_dir}/bb_2017__inner.txt', delimiter=',', unpack=True)
out_rad, out_apd = np.genfromtxt(f'{apod_dir}/bb_2017__outer.txt', delimiter=',', unpack=True)
drho = np.diff(inn_rad)[0]

#Invert inner
inn_apd = 1 - inn_apd

#Convert to angles
inn_apd *= np.pi/num_pet
out_apd *= np.pi/num_pet

#Convert to cartesian
top_inn_xy = inn_rad[:,None] * np.stack((np.cos(inn_apd), np.sin(inn_apd)), 1)
top_out_xy = out_rad[:,None] * np.stack((np.cos(out_apd), np.sin(out_apd)), 1)

#Flip
bot_inn_xy = top_inn_xy.copy()[::-1] * np.array([1,-1])
bot_out_xy = top_out_xy.copy()[::-1] * np.array([1,-1])

#Shifted petals
petal_shifts = {1: 7.5e-6, 2: 7.5e-6, 5: 7.5e-6, 6: 7.5e-6, 9: 10.5e-6, 10: 10.5e-6}

pet_ang = 2.*np.pi/num_pet
strt_rmax = np.hypot(*top_out_xy.T).min()

#Loop through and build each petal
mask = []
for i in range(1, num_pet+1):

    #rotation matrix
    cur_rot_mat = rot_mat(i*pet_ang)

    #Copy over inner and rotate
    inn_top = top_inn_xy.copy().dot(cur_rot_mat)
    inn_bot = bot_inn_xy.copy().dot(cur_rot_mat)

    #Shifted petal
    if i in petal_shifts.keys():

        #Get shift
        shift = petal_shifts[i]

        #Shift
        if i % 2 == 0:
            inn_bot += shift * np.array([np.cos(i*pet_ang), np.sin(i*pet_ang)])
        else:
            inn_top += shift * np.array([np.cos(i*pet_ang), np.sin(i*pet_ang)])

    #Copy over outer and rotate
    out_top = top_out_xy.copy().dot(cur_rot_mat)
    out_bot = bot_out_xy.copy().dot(cur_rot_mat)

    #Build Struts
    str_top_rad = np.arange(np.hypot(*inn_top.T).max()+drho, strt_rmax, drho)
    str_top = str_top_rad[:,None] * np.array([np.cos(max_apod*np.pi/num_pet), \
                                              np.sin(max_apod*np.pi/num_pet)])
    str_top = str_top.dot(cur_rot_mat)

    str_bot_rad = np.arange(np.hypot(*inn_bot.T).max()+drho, strt_rmax, drho)
    str_bot = str_bot_rad[:,None] * np.array([np.cos(max_apod*np.pi/num_pet), \
                                              np.sin(max_apod*np.pi/num_pet)])
    str_bot = (str_bot * np.array([1, -1])).dot(cur_rot_mat)[::-1]

    #Build total petal
    petal = np.concatenate((inn_top, str_top, out_top, out_bot, str_bot, inn_bot))

    # plt.plot(*inn_bot.T, 'x')
    # plt.plot(*inn_top.T, 'x')
    # plt.plot(*out_bot.T, 'x')
    # plt.plot(*out_top.T, 'x')
    # plt.plot(*str_top.T, '+')
    # plt.plot(*str_bot.T, '+')
    # plt.plot(*petal.T)
    # plt.axis('equal')
    # breakpoint()

    #Append
    mask.append(petal)

#Write out
with open(f'./{out_file}.dat', 'w') as f:
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            f.write(f'{mask[i][j][0]}, {mask[i][j][1]}\n')
        f.write('**,**\n')

for i in range(len(mask)):
    plt.plot(*mask[i].T)


breakpoint()
