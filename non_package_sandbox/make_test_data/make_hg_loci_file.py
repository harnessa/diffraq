import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import diffraq

sim = diffraq.Simulator()
ss_Afunc = lambda r: np.exp(-((r-sim.ss_rmin)/(sim.ss_rmax-sim.ss_rmin)/0.6)**6)

num_pts = 10000
npet = 16

rr = np.linspace(sim.ss_rmin, sim.ss_rmax, num_pts)
aa = ss_Afunc(rr)

#Trailing edge
xx = rr*np.cos(aa*np.pi/npet)
yy = rr*np.sin(aa*np.pi/npet)

#Leading edge
xx = np.concatenate((xx,  xx[::-1]))
yy = np.concatenate((yy, -yy[::-1]))

pet_ang = 2.*np.pi/npet

loci = []
for i in range(npet):
    xnew =  xx*np.cos(i*pet_ang) + yy*np.sin(i*pet_ang)
    ynew = -xx*np.sin(i*pet_ang) + yy*np.cos(i*pet_ang)
    loci.extend(np.stack((xnew, ynew),1))
loci = np.array(loci)

#Flip to run CCW
loci = loci[::-1]

plt.plot(*loci.T)


with open(f'{diffraq.int_data_dir}/Test_Data/hg_loci_file.txt', 'w') as f:
    f.write(f'#Test hypergaussian loci with {npet} petals and HG parameters: ' + \
        f'a={sim.ss_rmin}, b={(sim.ss_rmax-sim.ss_rmin)/0.6}, n=6\n')
    for i in range(len(loci)):
        f.write(f'{loci[i][0]},{loci[i][1]}\n')

breakpoint()
