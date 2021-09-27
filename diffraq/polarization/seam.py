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

        # #Add base valley seam     #TODO: is slow, doesnt add much
        # if self.shape.kind[:5] == 'petal':
        #     xq, yq, wq, dq, nq, gw = \
        #         self.get_valley_quad(xq, yq, wq, dq, nq, gw, seam_width)

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
        pos_angle, nq, gw = getattr(self, \
            f'get_normal_angles_{self.shape.kind}')(indt_values)

        #Build edge distances
        dq = seam_width * (dept_nodes * pos_angle).ravel()

        #Need to adjust where seams overlap (only used in petals)
        if gw is not None:

            #Handle unique petals differently
            if 'unique' in self.shape.kind:

                #Build gap widths to match k-inds
                bigw = np.array([])
                for ic in range(len(self.shape.outline.func)):

                    #Get edge keys that match
                    kinds = np.where(self.shape.edge_keys == ic)[0]

                    #Build gap widths
                    curw = gw[:,None] * np.ones(2*self.theta_nodes*len(kinds))

                    #Concatenate
                    bigw = np.concatenate((bigw, curw.ravel()))

            else:

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

    def get_quad_petal_unique(self, seam_width):
        return polar.seam_unique_petal_quad(self.shape.outline.func, self.shape.edge_keys, \
            self.shape.num_petals, self.shape.min_radius, self.shape.max_radius, \
            self.radial_nodes, self.theta_nodes, seam_width)

############################################
############################################

############################################
#####  Shape Specific Normals + Distances #####
############################################

    def get_normal_angles_petal(self, indt_values):

        #Get petal signs and angle to rotate
        ones = np.ones(2*self.theta_nodes, dtype=int)
        pet_mul = np.tile(np.concatenate((ones, -ones)), self.shape.num_petals)
        pet_add = 2*np.repeat(np.roll(np.arange(self.shape.num_petals), -1), \
            4*self.theta_nodes)

        #Get function and derivative values at the parameter values
        Aval = self.shape.outline.func(indt_values)
        func = Aval*pet_mul + pet_add
        diff = self.shape.outline.diff(indt_values)*pet_mul

        #Invert for gap width calc if opaque
        if self.shape.is_opaque:
            Aval = 1 - Aval

        #Get gap widths
        gw = (2*Aval*np.pi/self.shape.num_petals*indt_values).ravel()

        #Get cartesian function and derivative values at the parameter values
        cart_func, cart_diff = self.shape.cart_func_diffs( \
            indt_values, func=func, diff=diff)

        #Cleanup
        del func, diff, pet_add, ones, Aval

        #Calculate angle between normal and theta vector (orthogonal to position vector)
        pos_angle = -(cart_func[...,0]*cart_diff[...,0] + \
            cart_func[...,1]*cart_diff[...,1]) / (np.hypot(cart_func[...,0], \
            cart_func[...,1]) * np.hypot(cart_diff[...,0], cart_diff[...,1]))

        #Build normal angle
        nq = np.arctan2(pet_mul*cart_diff[...,0], -pet_mul*cart_diff[...,1]).ravel()

        #Cleanup
        del indt_values, cart_func, cart_diff, pet_mul

        return pos_angle, nq, gw

    ############################################

    def get_normal_angles_petal_unique(self, indt_values):

        #Get function and derivative values at the parameter values
        pos_angle, nq = np.empty(0), np.empty(0)
        for ic in range(len(indt_values)):

            #Get edge keys that match
            kinds = np.where(self.shape.edge_keys == ic)[0]

            #Get petal signs and angle to rotate
            pet_mul = np.repeat(np.array([-1,1])[kinds%2], 2*self.theta_nodes)
            pet_add = 2*np.repeat((kinds//2), 2*self.theta_nodes)

            #Get function and derivative
            Aval = self.shape.outline.func[ic](indt_values[ic])
            func = Aval*pet_mul + pet_add
            diff = self.shape.outline.diff[ic](indt_values[ic])*pet_mul

            #Invert for gap width calc if opaque
            if self.shape.is_opaque:
                Aval = 1 - Aval

            #Get gaps widths (force one half to be nominal petal)
            if ic == 0:
                oldA = self.shape.outline.func[0](indt_values[ic])
            gw = ((oldA + Aval)*np.pi/self.shape.num_petals*indt_values[ic]).ravel()

            #Get cartesian function and derivative values at the parameter values
            cart_func, cart_diff = self.shape.cart_func_diffs( \
                indt_values[ic], func=func, diff=diff)

            #Calculate angle between normal and theta vector (orthogonal to position vector)
            cur_pos_angle = -(cart_func[...,0]*cart_diff[...,0] + \
                cart_func[...,1]*cart_diff[...,1]) / (np.hypot(cart_func[...,0], \
                cart_func[...,1]) * np.hypot(cart_diff[...,0], cart_diff[...,1]))

            #Build normal angle
            cur_nq = np.arctan2(pet_mul*cart_diff[...,0], -pet_mul*cart_diff[...,1]).ravel()

            #Concatenate
            pos_angle = np.concatenate((pos_angle, cur_pos_angle.ravel()))
            nq = np.concatenate((nq, cur_nq))

        #Reshape pos_angle
        pos_angle = pos_angle.reshape(func.shape[0],-1)

        #Cleanup
        del kinds, pet_mul, pet_add, Aval, func, diff, cart_func, \
            cart_diff, cur_pos_angle, cur_nq, indt_values, oldA

        return pos_angle, nq, gw

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

    def get_normal_angles_cartesian(self, indt_values):
        #Just pass to polar
        return self.get_normal_angles_polar(indt_values)

############################################
############################################

############################################
#####  Valleys #####
############################################

    def get_valley_quad(self, xq, yq, wq, dq, nq, gw, seam_width):

        #Make 1D for normal petal
        rmins = np.atleast_1d(self.shape.min_radius)
        funcs = np.atleast_1d(self.shape.outline.func)

        #Get edge_keys
        if hasattr(self.shape, 'edge_keys'):
            edge_keys = self.shape.edge_keys
        else:
            edge_keys = np.zeros(self.shape.num_petal*2)

        #Get quad for valleys
        xv, yv, wv, dv, nv = polar.seam_valley_quad(rmins, funcs, self.theta_nodes, \
            seam_width, self.shape.num_petals, edge_keys)

        #Add to parents quad
        xq = np.concatenate((xq, xv))
        yq = np.concatenate((yq, yv))
        wq = np.concatenate((wq, wv))
        dq = np.concatenate((dq, dv))
        nq = np.concatenate((nq, nv))

        return xq, yq, wq, dq, nq, gw

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
