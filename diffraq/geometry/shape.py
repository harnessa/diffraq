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
from diffraq.utils import misc_util
import diffraq.geometry as geometry
from diffraq.utils import def_shape_params
from scipy.optimize import fmin, newton

class Shape(object):

    def __init__(self, parent, **kwargs):
        #Point to parent [Occulter]
        self.parent = parent

        #Set Default parameters
        misc_util.set_default_params(self, {**kwargs}, def_shape_params, skip_keys=['kind'])

        #Set nodes if fraction
        if self.radial_nodes is None:
            self.radial_nodes = self.parent.sim.radial_nodes
        elif self.radial_nodes < 1:
            self.radial_nodes = int(self.radial_nodes * self.parent.sim.radial_nodes)

        if self.theta_nodes is None:
            self.theta_nodes = self.parent.sim.theta_nodes
        elif self.theta_nodes < 1:
            self.theta_nodes = int(self.theta_nodes * self.parent.sim.theta_nodes)

        #Clocking matrix
        self.clock_angle = np.pi/self.num_petals
        self.clock_mat = self.build_rot_matrix(self.clock_angle)

        #Sign for perturbation and etch directions
        self.opq_sign = -(2*int(self.is_opaque) - 1)

        #Load outline and perturbations
        self.set_outline()
        self.set_perturbations()

############################################
#####  Outline and Perturbations #####
############################################

    def set_outline(self):
        #Set outline depending on input
        if self.edge_data is None and self.edge_file is None:
            #Use lambda functions
            self.outline = geometry.LambdaOutline(self, self.edge_func, \
                diff=self.edge_diff, etch_error=self.etch_error)

        else:
            #Use interpolation data (specified or from file)
            if self.edge_data is not None:
                edge_data = self.edge_data
            else:
                edge_data = self.load_edge_file(self.edge_file)

            #Build outline depending on if cartesian or not
            if self.kind == 'cartesian':
                Outline = geometry.Cart_InterpOutline
            else:
                Outline = geometry.InterpOutline
            self.outline = Outline(self, edge_data, etch_error=self.etch_error)

    def set_perturbations(self):

        #Turn perturbations into list
        if not isinstance(self.perturbations, list):
            self.perturbations = [self.perturbations]

        #Loop through and build perturbations
        self.pert_list = []
        for pert_dict in self.perturbations:
            #Get perturbation kind
            kind = pert_dict['kind'].title()

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

        #Clock (if applicable)
        if self.is_clocked:
            sxq, syq = np.stack((sxq,syq),1).dot(self.clock_mat).T

        #Rotate (if applicable)
        if not np.isclose(self.rotation, 0):
            rot_mat = self.build_rot_matrix(self.rotation)
            sxq, syq = np.stack((sxq,syq),1).dot(rot_mat).T

        return sxq, syq, swq

    def build_shape_edge(self):
        #Build main shape edge
        sedge = self.build_local_shape_edge()

        #Sort edge points
        sedge = self.sort_edge_points(sedge, self.num_petals)

        #Loop through perturbation list and add edge points (which adds to current sedge)
        for pert in self.pert_list:
            sedge = pert.build_edge_points(sedge)

        #Clock (if applicable)
        if self.is_clocked:
            sedge = sedge.dot(self.clock_mat)

        #Rotate (if applicable)
        if not np.isclose(self.rotation, 0):
            rot_mat = self.build_rot_matrix(self.rotation)
            sedge = sedge.dot(rot_mat)

        return sedge

############################################
############################################

############################################
#####  Perturbation Functions #####
############################################

    def find_closest_point(self, point):
        #Build minimizing function (derivative of distance equation)
        min_diff = lambda t: np.sum((self.cart_func(t) - point)*\
            self.cart_diff(np.array([t,t+1e-3])[:,None])[0])

        #Get initial guess
        x0 = np.arctan2(point[1], point[0])

        #Find root
        out, msg = newton(min_diff, x0, full_output=True)

        #Check
        if not msg.converged:
            print('\n!Closest point not Converged!\n')

        return out

    def find_width_point(self, t0, width):
        #Build distance = width equation
        dist = lambda t: np.hypot(*(self.cart_func(t) - \
            self.cart_func(t0))) - width

        #Solve (guess with nudge towards positive theta)
        t_guess = t0 - width/np.hypot(*self.cart_func(t0))/2
        out, msg = newton(dist, t_guess, full_output=True)

        #Check
        if not msg.converged:
            print('\n!Closest point (width) not Converged!\n')

        return out

############################################
############################################

############################################
#####   Misc Functions #####
############################################

    def load_edge_file(self, edge_file):
        #Load file
        data = np.genfromtxt(edge_file, delimiter=',')

        #Replace min/max radius
        self.min_radius = data[:,0].min()
        self.max_radius = data[:,0].max()

        return data

    def sort_edge_points(self, edge, num_petals):

        #FIXME: get better sort for lab occulter

        #Get angles
        angles = np.arctan2(edge[:,1] - edge[:,1].mean(), \
            edge[:,0] - edge[:,0].mean()) % (2.*np.pi)

        #Sort by angles
        edge = edge[np.argsort(angles)]

        return edge

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

    def build_rot_matrix(self, angle):
        return np.array([[ np.cos(angle), np.sin(angle)],
                         [-np.sin(angle), np.cos(angle)]])

############################################
############################################
