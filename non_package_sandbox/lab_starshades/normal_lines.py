import numpy as np
import matplotlib.pyplot as plt;plt.ion()

if [False, True][0]:

    r0, r1 = 8, 15
    hga, hgb = 8,4
    hgn = 6
    num_pet = 16

    etch = -0.1

    #Apod functions
    afunc = lambda r: np.exp(-((r - hga)/hgb)**hgn)
    dadr = lambda r: afunc(r) * (-hgn/hgb)*((r-hga)/hgb)**(hgn-1)

    # hypergauss = lambda r: np.exp(-((r - hga)/hgb)**hgn)
    # dadr = lambda r: hypergauss(r) * (-hgn/hgb)*((r-hga)/hgb)**(hgn-1)

    #xy equations
    xc = lambda r: r*np.cos(afunc(r)*np.pi/num_pet)
    yc = lambda r: r*np.sin(afunc(r)*np.pi/num_pet)

    dx = lambda r: -etch*(np.sin(afunc(r)*np.pi/num_pet) + \
        np.pi/num_pet*r*dadr(r)*np.cos(afunc(r)*np.pi/num_pet)) / \
        np.sqrt(1. + (np.pi/num_pet*r*dadr(r))**2)

    dy = lambda r:  etch*(np.cos(afunc(r)*np.pi/num_pet) - \
        np.pi/num_pet*r*dadr(r)*np.sin(afunc(r)*np.pi/num_pet)) / \
        np.sqrt(1. + (np.pi/num_pet*r*dadr(r))**2)

    rad = np.linspace(r0, r1, 1000)

    x1 = xc(rad)
    y1 = yc(rad)

    x2 = xc(rad) + dx(rad)
    y2 = yc(rad) + dy(rad)

    plt.figure()
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    breakpoint()

if [False, True][1]:

    etch = 0.05*2
    the = np.linspace(0,2*np.pi, 1000)
    rr = 2

    if [False, True][0]:
        #Circle
        fr = lambda t: rr
        df = lambda t: 0.
    else:
        #Boomerang
        a=0.3
        fr = lambda t: 1 + a*np.cos(3*t)
        df = lambda t: -3*a*np.sin(3*t)

    #Cartesian Equations
    xc = lambda t: fr(t)*np.cos(t)
    yc = lambda t: fr(t)*np.sin(t)

    dx = lambda t: -etch*(df(t)*np.sin(t) + fr(t)*np.cos(t))/np.sqrt(df(t)**2 + fr(t)**2)
    dy = lambda t:  etch*(df(t)*np.cos(t) - fr(t)*np.sin(t))/np.sqrt(df(t)**2 + fr(t)**2)

    x1 = xc(the)
    y1 = yc(the)

    x2 = xc(the) + dx(the)
    y2 = yc(the) + dy(the)

    #Polar equations
    nf = lambda t: fr(t) - etch*fr(t)/np.sqrt(df(t)**2 + fr(t)**2)
    nt = lambda t: t + etch*df(t)/np.sqrt(df(t)**2 + fr(t)**2)

    nxc = lambda t: fr(t) * np.cos(t) * (1. - etch/np.sqrt(df(t)**2 + fr(t)**2)*(1. + df(t)/fr(t)*np.tan(t)))
    nyc = lambda t: fr(t) * np.sin(t) * (1. - etch/np.sqrt(df(t)**2 + fr(t)**2)*(1. - df(t)/fr(t)/np.tan(t)))

    # xr2, yr2 = nf(nt(the))*np.cos(nt(the)),  nf(nt(the))*np.sin(nt(the))
    # xr2, yr2 = nf(the)*np.cos(nt(the)),  nf(the)*np.sin(nt(the))
    xr2, yr2 = nxc(the), nyc(the)

    print(np.nanmax(np.hypot(xr2-x2, yr2-y2)))

    plt.figure()
    plt.plot(x1, y1)
    plt.plot(x2, y2)

    plt.plot(xr2, yr2, '--')
    plt.axis('equal')

breakpoint()
