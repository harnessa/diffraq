import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import h5py
from scipy import integrate
from diffraq.quadrature import lgwt
from scipy.interpolate import interp1d

seam = 50e-6

wave = 641e-9

#Load data
maxwell_file = '/home/aharness/repos/diffraq/External_Data/Vector_Edges/DW9'
with h5py.File(f'{maxwell_file}.h5', 'r') as f:
    xx = f['xx'][()]
    sf = f[f'{wave*1e9:.0f}_s'][()]

nodes = np.arange(100, 2000, 100)
seams = [10e-6, 50e-6]

# sf = sf[xx > 1.5e-8]
# xx = xx[xx > 1.5e-8]

#Loop over node points
answers = []
true = []
for seam in seams[-1:]:
    tmp = []
    tru = abs(integrate.simpson(sf[abs(xx) <= seam], x=xx[abs(xx) <= seam]))
    true.append(tru)

    # sfx = interp1d(xx[abs(xx) <= seam], sf[abs(xx) <= seam], \
        # fill_value=(0j,0j), bounds_error=False,kind='cubic')

    for nw in nodes:

        if [False, True][0]:

            if [False, True][0]:
                pw, ww = lgwt(nw, -1, 1)

                pw *= seam
                ww *= seam
            else:
                pw, ww = lgwt(nw, 0, 1)
                pw = pw*xx.ptp()# + xx.min()
                ww *= xx.ptp()

            #TODO: why does adding xx screw things up?

            #Interpolate
            sfld = np.interp(pw, xx, sf, left=0j, right=0j)

            #Integrate
            ans = abs((sfld*ww).sum())

        else:

            if [False, True][0]:
                pw, ww = lgwt(nw, xx.min(), seam)
                sw = np.interp(pw, xx, sf, left=0j, right=0j)
                # sw = sfx(pw)
                ans = abs((sw*ww).sum())

            else:
                r0 = 0
                p1, w1 = lgwt(nw, r0, seam)
                p2, w2 = lgwt(nw, xx.min(), r0)

                pw = np.concatenate((p1, p2))
                ww = np.concatenate((w1, w2))

                # s1 = np.interp(p1, xx, sf, left=None, right=0j)
                # s2 = np.interp(p2, xx, sf, left=0j, right=None)

                # ans = abs( (s1*w1).sum() + (s2*w2).sum())

                sw = np.interp(pw, xx, sf, left=0j, right=0j)
                ans = abs((sw*ww).sum())


            # print(ans, true[0])
            # plt.cla()
            # plt.plot(p1, abs(s1),'x')
            # # plt.plot(p2, abs(s2),'+')
            # plt.plot(xx, abs(sf),'r')
            # breakpoint()

        #Append
        tmp.append(ans)
    answers.append(tmp)

answers = np.array(answers)
true = np.array(true)

#Differences
conv = abs(1 - answers/answers[:,-1][:,None]) * 100
diff = abs(1 - answers/true[:,None]) * 100

plt.figure()
for i in range(len(diff)):
    plt.semilogy(nodes, diff[i], 'o', label=f'True: {seams[i]*1e6:.0f}')
    plt.semilogy(nodes, conv[i], '*', label=f'Conv: {seams[i]*1e6:.0f}')

plt.axhline(1, color='k', linestyle=':')
plt.axhline(0.1, color='k', linestyle=':')
plt.legend()
plt.xlabel('Number of Nodes')
plt.ylabel('Percent Difference')

print(np.median(abs(diff),1), abs(diff).max(1))

breakpoint()
