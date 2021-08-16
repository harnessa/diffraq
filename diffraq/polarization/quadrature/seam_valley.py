"""
seam_polar_quad.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 08-16-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: quadrature over seam of valley base

"""

from diffraq.quadrature import lgwt
import numpy as np

def seam_valley_quad(rmins, funcs, n_nodes, seam_width, num_petals, edge_keys):

    ###########################
    ### Pre calcs ###
    ###########################

    #Seam quadrature nodes, weights
    quad_n, quad_w = lgwt(n_nodes, 0, 1)

    #Combine nodes for positive and negative sides of edge
    pr = np.concatenate((quad_n, -quad_n[::-1]))
    wr = np.concatenate((quad_w,  quad_w[::-1]))

    #Add axis
    quad_n = quad_n[:,None]
    quad_w = quad_w[:,None]

    #Get normal angle at all theta nodes (flipped negative to point inward to match seam_polar_quad)
    dt = np.stack((np.zeros(n_nodes), -np.ones(n_nodes)),1)[:,:,None]

    #Build edge distances
    same_dq = -seam_width * (np.ones(len(dt))[:,None] * pr).ravel()

    #Build normal angles (x2 for each side of edge)
    same_nq = (np.ones_like(pr) * np.arctan2(dt[:,0], dt[:,1])).ravel()

    ###########################
    ###########################

    #Loop over unique petals and get quad
    allx, ally, allw = [], [], []
    for i in range(len(rmins)):

        #Get edge keys that match
        kinds = np.where(edge_keys == i)[0]

        #Theta nodes and weights
        T0 = float(funcs[i](rmins[i])) * np.pi/num_petals
        pw = quad_n.copy() * T0
        wt = quad_w.copy() * T0

        #Get old and new edges
        old_edge = rmins[i]*np.stack((np.cos(pw), np.sin(pw)),1)

        #Nodes (Eq. 11 Barnett 2021) (nx and ny are switched and ny has negative sign)
        xqv = old_edge[:,0] + dt[:,1]*pr*seam_width
        yqv = old_edge[:,1] - dt[:,0]*pr*seam_width

        #Weights (Eq. 12  2021)
        wqv = -seam_width * (wt * wr * (xqv * dt[:,1] - yqv * dt[:,0])).ravel()

        #Ravel xq, yq
        xqv = xqv.ravel()
        yqv = yqv.ravel()

        # #Check tru area
        # tru_area = T0 * rmins[i] * seam_width*2
        # print(tru_area, wqv.sum())

        #Store
        allx.append(xqv)
        ally.append(yqv)
        allw.append(wqv)

    #Cleanup
    del quad_n, quad_w, pw, wt, dt, pr, wr, old_edge, xqv, yqv, wqv

    ###########################
    ### Build for all petals ###
    ###########################

    xq, yq, wq, nq, dq = np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0)
    for i in range(num_petals*2):

        #Get index to unique petal
        uind = edge_keys[i]

        #Get rotation angle
        ang = 2*np.pi/num_petals * (i//2)

        #Flip y to negative if odd index
        ysgn = 2*(i%2) - 1

        #Rotate
        newx = allx[uind]*np.cos(ang) - ally[uind]*np.sin(ang) * ysgn
        newy = allx[uind]*np.sin(ang) + ally[uind]*np.cos(ang) * ysgn
        newn = ang - same_nq.copy()

        #Concatenate
        xq = np.concatenate((xq, newx))
        yq = np.concatenate((yq, newy))
        wq = np.concatenate((wq, allw[uind]))
        nq = np.concatenate((nq, newn))
        dq = np.concatenate((dq, same_dq.copy()))

    #Cleanup
    del newx, newy, newn, allx, ally, allw, same_dq, same_nq

    return xq, yq, wq, dq, nq

############################################
############################################

def seam_valley_edge(g, n, seam_width):
    #TODO
    pass
