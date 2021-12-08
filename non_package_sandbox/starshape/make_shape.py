import numpy as np
import matplotlib.pyplot as plt;plt.ion()

x0 = 8e-3
x1 = 13e-3
num_pet = 12
gap = 22e-6

xx = np.linspace(x0, x1, 1000)
y0 = x0 * np.tan(np.pi/num_pet) - gap/2
y1 = gap/2
mm = (y0 - y1) / (x0 - x1)

yy = mm * (xx - x0) + y0

rr = np.hypot(xx, yy)
aa = num_pet/np.pi * np.arctan2(yy,xx)

plt.plot(rr, aa)

r2 = np.hypot(xx, yy)
a2 = num_pet/np.pi * np.arctan(x0/(x1-x0) * np.tan(np.pi/num_pet) * (x1/xx - 1))

plt.plot(r2, a2, '--')

xx = np.concatenate((xx,  xx[::-1]))
yy = np.concatenate((yy, -yy[::-1]))

plt.figure()

loci = []
for i in range(num_pet):
    cang = 2*np.pi/num_pet*i
    cx =  xx*np.cos(cang) + yy*np.sin(cang)
    cy = -xx*np.sin(cang) + yy*np.cos(cang)

    loci.append(np.stack((cx, cy), 1))

    plt.plot(cx, cy)

loci = np.array(loci)


breakpoint()
