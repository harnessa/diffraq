import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import diffraq

sim = diffraq.Simulator()
ss_Afunc = lambda r: np.exp(-((r-sim.ss_rmin)/(sim.ss_rmax-sim.ss_rmin)/0.6)**6)

num_pts = 10000

rr = np.linspace(sim.ss_rmin, sim.ss_rmax, num_pts)
aa = ss_Afunc(rr)

with open(f'{diffraq.int_data_dir}/Test_Data/test_apod_file.txt', 'w') as f:
    for i in range(len(rr)):
        f.write(f'{rr[i]},{aa[i]}\n')

plt.plot(rr, aa)
breakpoint()
