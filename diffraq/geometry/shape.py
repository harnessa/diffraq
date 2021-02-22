"""
shape.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-15-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Base class of an occulter shape, used to generate quadrature points.

"""

import numpy as np
import diffraq.geometry as geometry

class Shape(object):

    def __init__(self, parent, **kwargs):
        """
        Keyword arguments (all not used by every shape):
            - kind:         kind of shape object
            - edge_func:    lambda function describing the shape's edge; f(theta)
            - edge_diff:    lambda function describing the derivative of the shapes edge; f'(theta)
            - loci_file:    filename that holds the (x,y) coordinates describing the occulter edge
            - edge_file:    filename that holds numerical apodization function as a function of radius. Supercedes apod_func
            - is_opaque:    described shape is opaque?
            - num_petals:   number of petals
            - has_center:   has central disk? (for petal/starshade only)
            - min_radius:   minimum radius
            - max_radius:   maximum radius
            - rotation:     angle to rotate
            - perturbations: List of dictionaries describing perturbations to be added to the shape
            - radial_nodes: number of radial quadrature nodes
            - theta_nodes:  number of azimuthal quadrature nodes
        """

        #Point to parent [occulter]
        self.parent = parent

        #Default parameters
        def_params = {'kind':'polar', 'edge_func':None, 'edge_diff':None, \
            'loci_file':None, 'edge_file':None, 'is_opaque':False, \
            'num_petals':16, 'min_radius':0, 'max_radius':12, 'rotation':0, \
            'has_center':True, 'perturbations':[], \
            'radial_nodes':self.parent.sim.radial_nodes, \
            'theta_nodes':self.parent.sim.theta_nodes,}

        #Set Default parameters
        for k,v in {**def_params, **kwargs}.items():
            setattr(self, k, v)

        #Load outline and perturbations
        self.load_outline_perturbations()

############################################
#####  Outline and Perturbations #####
############################################

    def load_outline_perturbations(self):

        #Set outline function
        self.set_outline()

        #Sign for perturbation directions
        self.opq_sign = [1, -1][self.is_opaque]

        #Turn perturbations into list
        if not isinstance(self.perturbations, list):
            self.perturbations = [self.perturbations]

        #Loop through and build perturbations
        self.pert_list = []
        for pert_dict in self.perturbations:
            #Get perturbation kind
            kind = pert_dict['kind'].capitalize()

            #Build perturbation
            pert = getattr(geometry, kind)(self, **pert_dict)

            #Add to list
            self.pert_list.append(pert)

############################################
############################################

############################################
#####  Quadrature and Edge Points #####
############################################

    def build_shape_quadrature(self):
        #Build main shape quadrature
        sxq, syq, swq = self.build_local_shape_quad()

        #Loop through perturbation list and add quadratures (which adds to current [sxq,syq,swq])
        for pert in self.pert_list:
            sxq, syq, swq = pert.build_quadrature(sxq, syq, swq)

        #Rotate (if applicable)
        if not np.isclose(self.rotation, 0):
            sxq, syq = self.rotate_points(self.rotation, sxq, syq).T

        return sxq, syq, swq

    def build_shape_edge(self):
        #Build main shape edge
        sedge = self.build_local_shape_edge()

        #Sort edge points
        sedge = self.sort_edge_points(sedge, self.num_petals)

        #Loop through perturbation list and add edge points (which adds to current sedge)
        for pert in self.pert_list:
            sedge = pert.build_edge_points(sedge)

        #Rotate (if applicable)
        if not np.isclose(self.rotation, 0):
            sedge = self.rotate_points(self.rotation, sedge.T)

        return sedge

############################################
############################################

############################################
#####   Misc Functions #####
############################################

    def rotate_points(self, rot_ang, vec, vec_xtra=None):
        rot_arr = np.array([[ np.cos(rot_ang), np.sin(rot_ang)],
                            [-np.sin(rot_ang), np.cos(rot_ang)]])

        if vec.ndim == 1:
            vec = np.stack((vec, vec_xtra), 0)

        return rot_arr.dot(vec).T

    def sort_edge_points(self, edge, num_petals):
        #Sort by angle across petals
        new_edge = np.empty((0,2))
        for i in range(num_petals):

            #Difference between angles
            dang = 2*np.pi/num_petals

            #Angle bisector (mean angle)
            mang = i*2*np.pi/num_petals

            #Get angle between points and bisector
            dot = edge[:,0]*np.cos(mang) + edge[:,1]*np.sin(mang)
            det = edge[:,0]*np.sin(mang) - edge[:,1]*np.cos(mang)
            diff_angs = np.abs(np.arctan2(det, dot))

            #Indices are where angles are <= dang/2 away
            inds = diff_angs <= dang/2

            #Grab points in this angle range
            cur_pts = edge[inds]

            if len(cur_pts) == 0:
                continue

            #Get new angles
            cur_ang = np.arctan2(cur_pts[:,1] - cur_pts[:,1].mean(), \
                cur_pts[:,0] - cur_pts[:,0].mean()) % (2.*np.pi)

            #Sort current section
            cur_pts = cur_pts[np.argsort(cur_ang)]

            #Rotate to minimum radius
            min_rad_ind = np.argmin(np.hypot(*cur_pts.T))
            cur_pts = np.roll(cur_pts, -min_rad_ind, axis=0)

            #Append
            new_edge = np.concatenate((new_edge, cur_pts))

        #Check the same
        if new_edge.size != edge.size:
            print('Bad Sort!')
            breakpoint()

        #Cleanup
        del dot, det, diff_angs, inds, cur_pts, cur_ang, edge

        return new_edge

############################################
############################################
