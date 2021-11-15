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
import h5py
from diffraq.utils import misc_util
import diffraq.geometry as geometry
from diffraq.utils import def_shape_params
from scipy.optimize import newton

class Shape(object):

    def __init__(self, parent, **kwargs):
        #Point to parent [Occulter]
        self.parent = parent

        #Set Default parameters
        misc_util.set_default_params(self, {**kwargs}, def_shape_params, skip_keys=['kind'])

        #Sign for perturbation and etch directions
        self.opq_sign = -(2*int(self.is_opaque) - 1)

        #Set nodes
        self.set_nodes()

        #Set rotation angle
        self.set_rotation_angle()

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

            #Get min/max radius
            self.get_min_max_radius()

        else:
            #Use interpolation data (specified or from file)
            if self.edge_data is not None:
                edge_data = self.edge_data
            else:
                edge_data = self.load_edge_file(self.edge_file)

            #Replace min/max radius
            self.min_radius = edge_data[:,0].min()
            self.max_radius = edge_data[:,0].max()

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

        #Rotate (w/ clocking; if applicable)
        if self.rot_mat is not None:
            sxq, syq = np.stack((sxq,syq),1).dot(self.rot_mat).T

        return sxq, syq, swq

    def build_shape_edge(self):
        #Build main shape edge
        sedge = self.build_local_shape_edge()

        #Sort edge points
        sedge = self.sort_edge_points(sedge, self.num_petals)

        #Loop through perturbation list and add edge points (which adds to current sedge)
        for pert in self.pert_list:
            sedge = pert.build_edge_points(sedge)

        #Rotate (w/ clocking; if applicable)
        if self.rot_mat is not None:
            sedge = sedge.dot(self.rot_mat)

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

    def set_nodes(self):
        #If nodes not specified, set equal to parents. If nodes < 1, use fraction of parents
        if self.radial_nodes is None:
            self.radial_nodes = self.parent.sim.radial_nodes
        elif self.radial_nodes < 1:
            self.radial_nodes = int(self.radial_nodes * self.parent.sim.radial_nodes)

        if self.theta_nodes is None:
            self.theta_nodes = self.parent.sim.theta_nodes
        elif self.theta_nodes < 1:
            self.theta_nodes = int(self.theta_nodes * self.parent.sim.theta_nodes)

    def set_rotation_angle(self):
        #Clocking angle
        self.clock_angle = np.pi/self.num_petals

        #Rotation matrix
        self.rot_angle = self.rotation
        if self.is_clocked:
            self.rot_angle += self.clock_angle
        if not np.isclose(self.rot_angle, 0):
            self.rot_mat = self.parent.build_rot_matrix(self.rot_angle)
        else:
            self.rot_mat = None

    ##########################

    def load_edge_file(self, edge_file):
        #Load file
        with h5py.File(edge_file, 'r') as f:
            data = f['data'][()]

        return data

    def sort_edge_points(self, edge, num_petals):

        #FIXME: get better sort for lab occulter

        #Get angles
        angles = np.arctan2(edge[:,1] - edge[:,1].mean(), \
            edge[:,0] - edge[:,0].mean()) % (2.*np.pi)

        #Sort by angles
        edge = edge[np.argsort(angles)]

        return edge

############################################
############################################
