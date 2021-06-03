"""
unique_petal_shape.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 05-06-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Derived class of Shape with outline parameterized in radial coordinates,
    but allowing for unique petals.

"""

import numpy as np
import diffraq.quadrature as quad
from diffraq.geometry.shapes import PetalShape
from diffraq.geometry.outlines import Unique_InterpOutline
from scipy.optimize import fminbound

class UniquePetalShape(PetalShape):

    kind = 'petal_unique'
    param_code = 1000

############################################
#####  Setup #####
############################################

    def set_outline(self):
        #Load base interpolation data from file
        edge_data = [self.load_edge_file(self.edge_file)]

        #Set all keys to zero data index
        edge_data_keys = [0]*(2*self.num_petals)

        #Load unique edges from file
        if self.unique_edges is not None:
            for ef, ks in self.unique_edges.items():
                #Load data from file
                edge_data.append(self.load_edge_file(ef))
                #Redirect edge keys
                for k in ks:
                    edge_data_keys[k] = len(edge_data) - 1

        #Set None etch to zero
        if self.etch_error is None:
            self.etch_error = 0.

        #Give each petal an etch error
        etch_error = np.atleast_1d(self.etch_error)
        if etch_error.size == 1:
            etch_error = np.array([self.etch_error]*self.num_petals)

        #Give eatch edge an etch error (assume lab starshade and that each transmission region shares an error)
        etch_error = np.repeat(etch_error, 2)

        #Combos
        all_combos = np.stack((edge_data_keys, etch_error), 1)

        #Build unique edge combos
        combos, combo_keys = np.empty((0,2)), np.empty(0, dtype='int')
        for cur_combo in all_combos:
            #Only add if not alread in
            if not np.any(np.hypot(*(cur_combo - combos).T) == 0):
                combos = np.concatenate((combos, [cur_combo]))
            #Find key
            cur_key = np.argmin(np.hypot(*(cur_combo - combos).T))
            combo_keys = np.concatenate((combo_keys, [cur_key]))

        #Set min/max radii
        self.min_radius, self.max_radius = np.empty(0), np.empty(0)
        for dind in combos[:,0].astype(int):
            self.min_radius = np.concatenate((self.min_radius, [edge_data[dind][:,0].min()]))
            self.max_radius = np.concatenate((self.max_radius, [edge_data[dind][:,0].max()]))

        #Load outline
        self.outline = Unique_InterpOutline(self, edge_data, combos)

        #Store keys
        self.edge_keys = combo_keys.copy()

        #Clean up
        del edge_data, edge_data_keys, etch_error, all_combos, combos, combo_keys

############################################
############################################

############################################
#####  Main Shape #####
############################################

    def build_local_shape_quad(self):
        #Calculate petal quadrature
        xq, yq, wq = quad.unique_petal_quad(self.outline.func, self.edge_keys, self.num_petals, \
            self.min_radius, self.max_radius, self.radial_nodes, self.theta_nodes, \
            has_center=self.has_center)

        return xq, yq, wq

    def build_local_shape_edge(self, npts=None):
        #Radial nodes
        if npts is None:
            npts = self.radial_nodes

        #Calculate petal edge
        edge = quad.unique_petal_edge(self.outline.func, self.edge_keys, self.num_petals, \
            self.min_radius, self.max_radius, npts)

        return edge

############################################
############################################

############################################
#####  Wrappers for Cartesian coordinate systems #####
############################################

    def cart_func(self, r, func=None):
        #TODO: speed this up since I don't need to accomodate single petals

        #Grab function if not specified (usually by etch_error)
        if func is None:
            r, pet, pet_mul, pet_add = self.unpack_param(np.atleast_1d(r))
            func = np.empty((0,) + (1,)*(r.ndim-1))
            for p in np.unique(pet_add):
                ek = self.edge_keys[int(p)]
                func = np.concatenate((func, self.outline.func[ek](r) ))
            func = func*pet_mul + pet_add

        pang = np.pi/self.num_petals
        return r * np.stack((np.cos(func*pang), np.sin(func*pang)), func.ndim).squeeze()

    def cart_diff(self, r, func=None, diff=None):
        #Grab function and derivative if not specified (usually by etch_error)
        if func is None or diff is None:
            r, pet, pet_mul, pet_add = self.unpack_param(np.atleast_1d(r))
            func = np.empty((0,) + (1,)*(r.ndim-1))
            diff = np.empty_like(func)
            for p in np.unique(pet_add):
                ek = self.edge_keys[int(p)]
                func = np.concatenate((func, self.outline.func[ek](r) ))
                diff = np.concatenate((diff, self.outline.diff[ek](r) ))
            func = func*pet_mul + pet_add
            diff *= pet_mul

        pang = np.pi/self.num_petals
        cf = np.cos(func*pang)
        sf = np.sin(func*pang)
        ans = np.stack((cf - r*sf*diff*pang, sf + r*cf*diff*pang), diff.ndim).squeeze()
        del func, diff, cf, sf
        return ans

    def cart_diff_2nd(self, r, func=None, diff=None, diff_2nd=None):
        if func is None or diff is None:
            r, pet, pet_mul, pet_add = self.unpack_param(np.atleast_1d(r))
            func = np.empty((0,) + (1,)*(r.ndim-1))
            diff = np.empty_like(func)
            diff_2nd = np.empty_like(func)
            for p in np.unique(pet_add):
                ek = self.edge_keys[int(p)]
                func = np.concatenate((func, self.outline.func[ek](r) ))
                diff = np.concatenate((diff, self.outline.diff[ek](r) ))
                diff_2nd = np.concatenate((diff_2nd, self.outline.diff_2nd[ek](r) ))
            func = func*pet_mul + pet_add
            diff *= pet_mul
            diff_2nd *= pet_mul

        pang = np.pi/self.num_petals
        cf = np.cos(func*pang)
        sf = np.sin(func*pang)
        shr = (2*diff + r*diff_2nd)*pang
        ans = np.stack((-(diff*pang)**2*r*cf - sf*shr, -(diff*pang)**2*r*sf + cf*shr), func.ndim).squeeze()
        del func, diff, diff_2nd, cf, sf, shr
        return ans

    ############################################

    def cart_func_diffs(self, r, func=None, diff=None, diff_2nd=None, with_2nd=False):
        """Same functions as above, just calculate all at once to save time"""
        if func is None:
            r, pet, pet_mul, pet_add = self.unpack_param(np.atleast_1d(r))
            func = np.empty((0,) + (1,)*(r.ndim-1))
            diff = np.empty_like(func)
            diff_2nd = np.empty_like(func)
            for p in np.unique(pet_add):
                ek = self.edge_keys[int(p)]
                func = np.concatenate((func, self.outline.func[ek](r) ))
                diff = np.concatenate((diff, self.outline.diff[ek](r) ))
                if with_2nd:
                    diff_2nd = np.concatenate((diff_2nd, self.outline.diff_2nd[ek](r) ))
            func = func*pet_mul + pet_add
            diff *= pet_mul
            if with_2nd:
                diff_2nd *= pet_mul

        #Calculate intermediaries
        pang = np.pi/self.num_petals
        cf = np.cos(func*pang)
        sf = np.sin(func*pang)

        #Function
        func_ans = r[:,None]*np.stack((cf, sf), func.ndim).squeeze()

        #Derivative
        diff_ans = np.stack((cf - r*sf*diff*pang, sf + r*cf*diff*pang), diff.ndim).squeeze()

        #Second derivative
        if with_2nd:
            shr1 = -pang * diff**2 * r
            shr2 = 2*diff + r*diff_2nd
            diff_2nd_ans = pang*np.stack((shr1*cf - sf*shr2, shr1*sf + cf*shr2), func.ndim).squeeze()

            #Cleanup
            del func, diff, cf, sf, diff_2nd, shr1, shr2

            return func_ans, diff_ans, diff_2nd_ans

        else:

            #Cleanup
            del func, diff, cf, sf

            return func_ans, diff_ans

############################################
############################################
