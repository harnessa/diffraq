import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import diffraq
from utilities import util

num_pet = 12
max_apod = 0.9
apod_dir = '/home/aharness/repos/diffraq/External_Data/Apodization_Profiles'

out_file = 'M12P2_new'

def rot_mat(angle):
    return np.array([[ np.cos(angle), np.sin(angle)],
                     [-np.sin(angle), np.cos(angle)]])

#Load data
rads, apod = np.genfromtxt(f'{apod_dir}/bb_2017.txt', delimiter=',', unpack=True)

#Convert to angles
apod *= np.pi/num_pet

#Convert to cartesian
xy = rads[:,None] * np.stack((np.cos(apod), np.sin(apod)), 1)

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

#Notches [petal, x0, y0, height, width]
old_inn_xy = np.array([-2.5761, -11.5170]) * 1e-3
inn_nch = [9, -2.5772e-3, -11.5262e-3, 2.47e-6, 413.82e-6 - 9.0e-6]

old_out_xy = np.array([ 2.9154,  20.7524]) * 1e-3
out_nch = [3, 2.9154e-3, 20.7524e-3, 1.71e-6, 530.94e-6 - 13.0e-6]
notches = [inn_nch, out_nch]

def make_line(r1, r2, num_pts):
    xline = np.linspace(r1[0], r2[0], num_pts)
    yline = np.linspace(r1[1], r2[1], num_pts)
    return np.stack((xline,yline),1)[1:-1]

#Add notches
new_petals = []
for nch in notches:

    #Get current petal and find start point
    cur_pet = mask[nch[0]].copy()
    ind0 = np.argmin(np.hypot(cur_pet[:,0] - nch[1], cur_pet[:,1] - nch[2]))
    p0 = cur_pet[ind0]

    old = cur_pet.copy()

    #Find finish point
    r0 = np.hypot(*p0)
    rr = np.hypot(*cur_pet.T)
    indf = np.argmin(1/(1+1e-9-np.heaviside(r0-rr, 0.5)) * \
        np.abs(np.hypot(cur_pet[:,0] - p0[0], cur_pet[:,1] - p0[1]) - nch[-1]))
    pf = cur_pet[indf]

    #make sure positive radius
    print(np.hypot(*pf) > np.hypot(*p0), indf > ind0)

    #Grab section between
    if indf > ind0:
        sec = cur_pet[ind0:indf+1].copy()
    else:
        sec = cur_pet[indf:ind0+1][::-1].copy()

    #Get edge normal
    normal = util.compute_derivative(sec)
    normal = np.vstack((-normal[:,1], normal[:,0])).T
    normal /= np.hypot(*normal.T)[:,None]

    #Build new edge
    new_edge = sec + normal * nch[3]

    #Flip back if necessary
    if not (indf > ind0):
        new_edge = new_edge[::-1]

    #Get lines
    nlne = max(10, int(nch[3] / abs(rr[1]-rr[0])))

    #Build new petal
    if indf > ind0:
        lne0 = make_line(cur_pet[ind0], new_edge[0], nlne)
        lnef = make_line(new_edge[-1], cur_pet[indf], nlne)
        new_petal = np.concatenate((cur_pet[:ind0+1], lne0, new_edge, lnef, cur_pet[indf:]))
    else:
        lne0 = make_line(cur_pet[indf], new_edge[0], nlne)
        lnef = make_line(new_edge[-1], cur_pet[ind0], nlne)
        new_petal = np.concatenate((cur_pet[:indf+1], lne0, new_edge, lnef, cur_pet[ind0:]))

    #Append
    new_petals.append([nch[0], new_petal])

    #check direction
    print(np.all(new_petal[0] == old[0]))

    #Load stuart's
    ldir = '/home/aharness/repos/Milestone_2/analysis/modeling/M12P2/make_loci/Loci'
    origname = {9:'petal4_notch12p', 3:'petal10_notch12p'}[nch[0]]
    orig = np.genfromtxt(f'{ldir}/{origname}.txt', delimiter='  ')
    orig[:,1] *= -1
    orig = orig.dot(rot_mat(2.*np.pi/num_pet*nch[0]))

    plt.cla()
    plt.plot(*old.T, '-')
    plt.plot(*orig.T, 'x-')
    plt.plot(*new_petal.T, '+-')
    plt.plot(*p0, 'o')
    plt.plot(*pf, 's')
    plt.axis('equal')
    win = 0.05e-3
    # cen = nch[1:3]
    cen = pf
    plt.xlim([cen[0]-win, cen[0]+win])
    plt.ylim([cen[1]-win, cen[1]+win])

    breakpoint()

#Build new mask
new_mask = []
for i in range(num_pet):
    if i == new_petals[0][0]:
        new_mask.append(new_petals[0][1])
    elif i == new_petals[1][0]:
        new_mask.append(new_petals[1][1])
    else:
        new_mask.append(mask[i])

plt.cla()
for i in range(len(mask)):
    plt.plot(*new_mask[i].T)

with open(f'./{out_file}.dat', 'w') as f:
    for i in range(len(new_mask)):
        for j in range(len(new_mask[i])):
            f.write(f'{new_mask[i][j][0]}, {new_mask[i][j][1]}\n')
        f.write('**,**\n')


breakpoint()
