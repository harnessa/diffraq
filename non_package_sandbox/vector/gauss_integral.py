import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import h5py
from scipy import integrate
from diffraq.quadrature import lgwt
from scipy.special import roots_legendre
from scipy.interpolate import interp1d

nw = 1600* 1

wave = 641e-9

maxwell_file = '/home/aharness/repos/diffraq/External_Data/Vector_Edges/DW9'
with h5py.File(f'{maxwell_file}.h5', 'r') as f:
    xx = f['xx'][()]
    sf = f[f'{wave*1e9:.0f}_s'][()]

sf = sf[abs(xx) <= 50e-6]
xx = xx[abs(xx) <= 50e-6]

a1 = integrate.simpson(sf, x=xx)
a2 = integrate.trapezoid(sf, x=xx)

# sws = range(5, int(xx.max()*1e6)+5, 5)
sws = range(5, int(xx.max()*1e6)+2, 1)
pw, ww = lgwt(nw, -1, 1)

# a = -1
# b = 1
#
# p1, ww = np.polynomial.legendre.leggauss(nw)
# # p1, ww = roots_legendre(nw)
# pw = (a*(1-p1) + b*(1+p1))/2
# ww *= (b-a)/2

# #cheby
# pw = np.cos(np.pi*(2*np.arange(1,nw+1) -1)/2/nw)
# beta = np.sqrt(1 - pw**2)

sfx = interp1d(xx, sf, fill_value=(0j,0j), bounds_error=False,kind='quadratic')

dds = []
anss = []
for seam in sws:

    a1cut = integrate.simpson(sf[abs(xx) <= seam*1e-6], x=xx[abs(xx) <= seam*1e-6])
    # diff = abs(1. - a1cut/a1)*100 * np.sign(abs(a1)-abs(a1cut))
    # print(f'{seam}, diff: {diff:.3f}')

    sfld = np.interp(pw*seam*1e-6, xx, sf, left=0j, right=0j)
    # sfld = sfx(pw*seam*1e-6)

    b1 = (sfld*ww).sum()*seam*1e-6

    #cheby
    # b1 = seam*1e-6 * np.pi/nw * (beta * sfld).sum()

    diff = abs(1. - b1/a1)*100 #* np.sign(abs(a1)-abs(b1))
    # diff = abs(1. - b1/a1cut)*100 * np.sign(abs(a1cut)-abs(b1))

    dds.append(diff)
    anss.append(b1)

    print(f'{seam}, diff: {diff:.3f}')
    # plt.plot(pw*seam*1e-6, abs(sfld))
    # breakpoint()

# print(f'\n{abs(a1):.5e}, {abs(a2):.5e}, {abs(a1cut):.5e}, {abs(b1):.5e}\n')
anss = np.array(anss)
print(np.mean(np.abs(dds)))

plt.plot(pw*seam*1e-6, abs(sfld), 'x')
plt.plot(xx, abs(sf),'-')
# plt.plot(xx, abs(sf),'+')
plt.axvline(-seam*1e-6, color='k', linestyle=':')
plt.axvline( seam*1e-6, color='k', linestyle=':')

plt.figure()
plt.plot(sws, dds, 'o')
plt.axhline(0, color='k')
plt.axhline(1, color='k', linestyle=':')
plt.axvline(xx.max()*1e6, color='k')
breakpoint()
