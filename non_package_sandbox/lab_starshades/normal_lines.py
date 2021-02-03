import numpy as np
import matplotlib.pyplot as plt;plt.ion()

r0, r1 = 8, 15
hga, hgb = 8,4
hgn = 6
num_pet = 16

hypergauss = lambda r: np.exp(-((r - hga)/hgb)**hgn)

hypx = lambda r: r*np.cos(hypergauss(r)*np.pi/num_pet)
hypy = lambda r: r*np.sin(hypergauss(r)*np.pi/num_pet)

dadr = lambda r: hypergauss(r) * (-hgn/hgb)*((r-hga)/hgb)**(hgn-1)

rxy = lambda x, y: np.sqrt(x**2 + y**2)
dydx = lambda x, y: (y + np.pi/num_pet*dadr(rxy(x,y))*rxy(x,y)*x) / \
                    (x - np.pi/num_pet*dadr(rxy(x,y))*rxy(x,y)*y)

rads = np.linspace(r0, r1, 1000)
apod = hypergauss(rads)

# plt.plot(rads, apod)

x = hypx(rads)
y = hypy(rads)

newx = lambda x, y, d: x + d/np.sqrt(1. + 1./dydx(x,y)**2)
newy = lambda x, y, d: y + d/np.sqrt(1. + dydx(x,y)**2)

etch = 0.1
x2 = newx(x,y, etch)
y2 = newy(x,y, etch)

plt.figure()
plt.plot(x,y)
plt.plot(x2, y2)

# the = np.linspace(0,2*np.pi, 1000)
# rr = 10
#
# xc = rr*np.cos(the)
# yc = rr*np.sin(the)
#
# xc2 = xc + etch*10/np.sqrt(1. + (xc/yc)**2)
# yc2 = yc + etch*10/np.sqrt(1. + (yc/xc)**2)
#
# plt.figure()
# plt.plot(xc, yc)
# plt.plot(xc2, yc2)

breakpoint()
