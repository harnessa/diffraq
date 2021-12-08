"""
seam_rectangle_quad.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-15-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: quadrature over seam of edge of rectangle.

"""

from diffraq.quadrature import lgwt
import numpy as np

def seam_rectangle_quad(sides, m, n, seam_width):
    """
    xq, yq, wq = seam_rectangle_quad(sides, m, n, seam_width)

    Inputs:
        sides = sides of rectangle CCW: (bottom, left, top, right)
        m = # nodes over disc and over radial apodization [r0, r1]
        n = # nodes over petal width
        seam_width = seam half-width [meters]

    Outputs:
        xq, yq = numpy array of x,y coordinates of nodes [meters]
        wq = numpy array of weights
    """

    import matplotlib.pyplot as plt;plt.ion()

    #Theta nodes, constant weights
    pt, wt = lgwt(n, 0, 1)
    wt = wt[:,None]

    #Cartesian quadrature nodes, weights
    pr, wr = lgwt(m, 0, 1)

    #Combine nodes for positive and negative sides of edge
    pr = np.concatenate((pr, -pr[::-1]))
    wr = np.concatenate((wr,  wr[::-1]))

    #Loop through sides and build quads
    xq, yq, wq = np.array([]), np.array([]), np.array([])
    for i in range(len(sides)):

        #Decide which axes to take if horizontal or vertical
        if i % 2 == 0:                  #Top/bottom
            var = sides[i][:,0]         #Varying is horizontal
            con = sides[i][:,1][0]      #Constant is vertical
        else:
            var = sides[i][:,1]         #Varying is vertical
            con = sides[i][:,0][0]      #Constant is horizontal

        #Resample onto gauss nodes
        varq = np.interp(pt, np.linspace(0,1,len(var)), var)[:,None]
        conq = np.ones(n)[:,None] * con

        #Varying coords is just repeated    (x2 for both sides of seam)
        var2 = np.tile(varq, (1, m*2))

        #Constant coords gets put to seam   #Normal angles is sign of constant value
        con2 = conq + np.sign(con) * pr * seam_width

        #Build weights
        neww = seam_width * wt * wr * abs(varq)

        #Decide which axes to put back
        if i % 2 == 0:
            newx = var2
            newy = con2
        else:
            newy = var2
            newx = con2

        print(neww.sum(), seam_width*2*var.ptp())

        #Concatenate
        xq = np.concatenate((xq, newx.ravel()))
        yq = np.concatenate((yq, newy.ravel()))
        wq = np.concatenate((wq, neww.flatten()))

        # plt.cla()
        # plt.plot(con2.flatten(), var2.flatten(), 'x')
        # plt.plot(con*np.ones_like(var), var)

        plt.plot(newx.ravel(), newy.ravel(), 'x')
        plt.plot(sides[i][:,0], sides[i][:,1])

        breakpoint()
    #     #Get function values at all theta nodes
    #     ft = fxy(pt)[:,:,None]
    #     dt = dxy(pt)[:,:,None]
    #
    #     #Get normal angle at all theta nodes (flipped negative to point inward to match seam_polar_quad)
    #     norm = np.hypot(dt[:,0], dt[:,1])
    #     nx =  dt[:,1] / norm
    #     ny = -dt[:,0] / norm
    #
    #     #Nodes (Eq. 11 Barnett 2021)
    #     xq = ft[:,0] + nx*pr*seam_width
    #     yq = ft[:,1] + ny*pr*seam_width
    #
    # #Weights (Eq. 12 Barnett 2021) (nx and ny are switched and ny has negative sign)
    # wq = wt * seam_width * (wr * (xq * nx + yq * ny)).ravel()
    #
    # #Ravel
    # xq = xq.ravel()
    # yq = yq.ravel()

    #Cleanup
    # del wr, ft, dt, norm, nx, ny

    #Return nodes along primary axis (radius) and values along orthogonal axis (theta)
    return xq, yq, wq, pr, pt

    breakpoint()
