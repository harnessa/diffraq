"""
seam.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 03-18-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class representing the narrow seam around the edge of an occulter/aperture
    that builds the area quadrature and edge distances allowing the calculation of
    non-scalar diffraction via the Braunbek method.

"""

import numpy as np
import diffraq.polarization as polar
import copy

class Seam(object):

    def __init__(self, shape, radial_nodes, theta_nodes):
        self.shape = shape
        self.radial_nodes = radial_nodes
        self.theta_nodes = theta_nodes

        #Set perturbations
        self.set_perturbations()

    def set_perturbations(self):
        #Loop through and build perturbations
        self.pert_list = []
        for pert_dict in self.shape.perturbations:
            #Get perturbation kind
            kind = pert_dict['kind'].title()

            #Build perturbation (need to copy dictionary b/c shared with shape)
            pert = getattr(polar, 'Seam_'+kind)(self.shape, **copy.deepcopy(pert_dict))

            #Add to list
            self.pert_list.append(pert)

############################################
#####  Build Quadrature and Edge distances #####
############################################

    def build_seam_quadrature(self, seam_width):
        #Get main shape quadrature
        xq, yq, wq, dq, nq, gw = self.build_shape_quadrature(seam_width)

        #Store number of pre-perturbation points
        n_nodes = len(xq)

        #Add perturbations
        for pert in self.pert_list:
            xq, yq, wq, dq, nq, gw = pert.build_quadrature(xq, yq, wq, dq, nq, gw)

        #Rotate with parent shape
        if self.shape.rot_mat is not None:
            xq, yq = np.stack((xq,yq),1).dot(self.shape.rot_mat).T
            nq += self.shape.rot_angle

        #Flip sign of distance and rotate normal angle by pi if opaque
        if self.shape.is_opaque:
            dq *= -1
            nq += np.pi

        return xq, yq, wq, dq, nq, gw, n_nodes

    ############################################

    def build_shape_quadrature(self, seam_width):
        #Get shape specific quadrature and nodes in dependent direction (r-polar, theta-petal) and
            #values in independent, i.e. parameter, direction (theta-polar, r-petal)
        xq, yq, wq, dept_nodes, indt_values = getattr(self, \
            f'get_quad_{self.shape.kind}')(seam_width)

        #Get normal and position angles and gap widths depending on shape
        if self.shape.kind == 'petal':
            pos_angle, nq, gw = self.get_normal_angles_petal(indt_values)
        else:
            pos_angle, nq, gw = self.get_normal_angles_polar(indt_values)

        #Build edge distances
        dq = seam_width * (dept_nodes * pos_angle).ravel()

        #Need to adjust where seams overlap (only used in petals)
        if gw is not None:

            #Build full gap widths
            bigw = (gw[:,None] * np.ones_like(dept_nodes)).ravel()

            #Find overlap
            ovr_inds = dq >= bigw/2

            #Zero out weights on overlap
            wq[ovr_inds] = 0

            #Clear gap widths if not running gaps
            if not self.shape.parent.sim.with_vector_gaps:
                gw = None

            #Cleanup
            del bigw, ovr_inds

        #Cleanup
        del dept_nodes, pos_angle

        return xq, yq, wq, dq, nq, gw

############################################
############################################

############################################
#####  Shape Specific Quadrature #####
############################################

    def get_quad_polar(self, seam_width):
        return polar.seam_polar_quad(self.shape.outline.func, \
            self.radial_nodes, self.theta_nodes, seam_width)

    def get_quad_cartesian(self, seam_width):
        return polar.seam_cartesian_quad(self.shape.outline.func,
            self.shape.outline.diff, self.radial_nodes, \
            self.theta_nodes, seam_width)

    def get_quad_petal(self, seam_width):
        return polar.seam_petal_quad(self.shape.outline.func, self.shape.num_petals, \
            self.shape.min_radius, self.shape.max_radius, self.radial_nodes, \
            self.theta_nodes, seam_width)

############################################
############################################

############################################
#####  Shape Specific Normals + Distances #####
############################################
    def get_normal_angles_petal(self, indt_values):

        #Get petal signs and angle to rotate
        ones = np.ones(2*self.theta_nodes, dtype=int)
        pet_mul = np.tile(np.concatenate((ones, -ones)), self.shape.num_petals)
        pet_add = 2*(np.repeat(np.roll(np.arange(self.shape.num_petals) + 1, -1), \
            4*self.theta_nodes) - 1)

        #Get function and derivative values at the parameter values
        #FIXME: for now, ignore overetch and assume all petals have the same normal
        if not isinstance(self.shape.outline.func, list):
            Aval = self.shape.outline.func(indt_values)
            func = Aval*pet_mul + pet_add
            diff = self.shape.outline.diff(indt_values)*pet_mul
        else:
            Aval = self.shape.outline.func[0](indt_values)
            func = Aval*pet_mul + pet_add
            diff = self.shape.outline.diff[0](indt_values)*pet_mul

        #Get gaps widths
        if self.shape.is_opaque:
            Aval = 1 - Aval
        gw = (2*Aval*np.pi/self.shape.num_petals*indt_values).ravel()

        #Get cartesian function and derivative values at the parameter values
        cart_func, cart_diff = self.shape.cart_func_diffs( \
            indt_values, func=func, diff=diff)

        #Cleanup
        del func, diff, pet_add, ones, Aval

        #Calculate angle between normal and theta vector (orthogonal to position vector)
        pos_angle = -(cart_func[...,0]*cart_diff[...,0] + cart_func[...,1]*cart_diff[...,1]) / \
            (np.hypot(cart_func[...,0], cart_func[...,1]) * np.hypot(cart_diff[...,0], cart_diff[...,1]))

        #Build normal angle
        nq = np.arctan2(pet_mul*cart_diff[...,0], -pet_mul*cart_diff[...,1]).ravel()

        #Cleanup
        del indt_values, cart_func, cart_diff, pet_mul

        return pos_angle, nq, gw

    # def get_normal_angles_petal(self, indt_values):
    #
    #     #Get petal signs and angle to rotate
    #     ones = np.ones(2*self.theta_nodes, dtype=int)
    #     pet_mul = np.tile(np.concatenate((ones, -ones)), self.shape.num_petals)
    #     pet_add = 2*(np.repeat(np.roll(np.arange(self.shape.num_petals) + 1, -1), \
    #         4*self.theta_nodes) - 1)
    #
    #     #Get function and derivative values at the parameter values
    #     if not isinstance(self.shape.outline.func, list):
    #         Aval = pet_mul*self.shape.outline.func(indt_values)
    #         func = np.tile(Aval + pet_add, (1, self.shape.num_petals))
    #         Aval = np.tile(Aval, (1, self.shape.num_petals))
    #         diff = np.tile(pet_mul*self.shape.outline.diff(indt_values), (1, self.shape.num_petals))
    #     else:
    #         Aval = [pet_mul*af(indt_values) for af in self.shape.outline.func]
    #         func = np.hstack(([Av + pet_add for Av in Aval]))
    #         Aval = np.hstack(([Av for Av in Aval]))
    #         diff = np.hstack(([pet_mul*df(indt_values) for df in self.shape.outline.diff]))
    #
    #     # Aval = self.shape.outline.func(indt_values)
    #     # func = Aval*pet_mul + pet_add
    #     # diff = self.shape.outline.diff(indt_values)*pet_mul
    #
    #     #Get gaps widths
    #     if self.shape.is_opaque:
    #         Aval = 1 - Aval
    #     gw = (2*Aval*np.pi/self.shape.num_petals*indt_values).ravel()
    #
    #     #Get cartesian function and derivative values at the parameter values
    #     cart_func, cart_diff = self.shape.cart_func_diffs( \
    #         indt_values, func=func, diff=diff)
    #
    #     #Cleanup
    #     # del func, diff, pet_add, ones, Aval
    #
    #     #Calculate angle between normal and theta vector (orthogonal to position vector)
    #     pos_angle = -(cart_func[...,0]*cart_diff[...,0] + cart_func[...,1]*cart_diff[...,1]) / \
    #         (np.hypot(cart_func[...,0], cart_func[...,1]) * np.hypot(cart_diff[...,0], cart_diff[...,1]))
    #     # breakpoint()
    #
    #     #Build normal angle
    #     nmul = pet_mul.shape[0]
    #     nq = np.hstack(([np.arctan2(pet_mul*cart_diff[:,nmul*i:nmul*(i+1),0], \
    #         -pet_mul*cart_diff[:,nmul*i:nmul*(i+1),1]) for i in range(self.shape.num_petals)]))
    #     nq = nq.ravel()
    #     breakpoint()
    #     # nq = np.arctan2(pet_mul*cart_diff[...,0], -pet_mul*cart_diff[...,1]).ravel()
    #
    #     #Cleanup
    #     del indt_values, cart_func, cart_diff, pet_mul
    #
    #     return pos_angle, nq, gw

    ############################################

    def get_normal_angles_polar(self, indt_values):
        #Get function and derivative values at the parameter values
        func, diff = self.shape.cart_func_diffs(indt_values)

        #Calculate angle between normal and radius vector (position vector)
        pos_angle = ((-func[:,0]*diff[:,1] + func[:,1]*diff[:,0]) / \
            (np.hypot(func[:,0],func[:,1]) * np.hypot(diff[:,0],diff[:,1])))[:,None]

        #Build normal angles (x2 for each side of edge)
        nq = (np.ones(2*self.radial_nodes) * \
            np.arctan2(diff[:,0], -diff[:,1])[:,None]).ravel()

        #Pass dummy gap widths
        gw = None

        #Cleanup
        del func, diff

        return pos_angle, nq, gw

############################################
############################################

############################################
#####  Build Edge Shape #####
############################################

    def build_seam_edge(self, npts=None):
        #TODO: add edge
        breakpoint()
        return edge

############################################
############################################
