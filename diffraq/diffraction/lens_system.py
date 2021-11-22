"""
lens_system.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 11-22-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class to hold full lens system angular spectrum
    propagation.

"""

import diffraq
import numpy as np

class Lens_System(object):

    def __init__(self, elements, sim):
        self.mm2m = 1e-3
        self.m2mm = 1e3
        self.sim = sim
        self.build_system(elements)

    def build_system(self, elements):

        #If elements not specified, build dummy system
        if elements is None:
            self.n_elements = 1
            pms = {'lens_name':'simple', 'diameter':self.sim.tel_diameter, \
                'distance':self.sim.focal_length}
            self.element_0 = diffraq.diffraction.Lens(pms, self.sim.num_pts)

        else:

            #Number of elements
            self.n_elements = len(elements)

            #Build element classes
            for i in range(self.n_elements):
                elem = Lens_Element(elements[f'element_{i}'], self.sim.num_pts)
                setattr(self, f'element_{i}', elem)

############################################
############################################

class Lens_Element(object):

    def __init__(self, params, num_pts):
        self.num_pts = num_pts
        self.set_parameters(params)
        self.build_element()

    def set_parameters(self, params):
        def_pms = {
            'kind':             '',
            'lens_name':        '',
            'input_diameter':   None,
            'input_dx':         None,
            'distance':         1e-12,
        }

        #Parameters
        for k, v in def_pms.items():
            setattr(self, k, v)

        for k, v in params.items():
            setattr(self, k, v)

    def build_element(self):

        #Build opd function
        if self.kind == 'lens':
            #Load lens
            self.lens = diffraq.diffraction.Lens(self.lens_name)
            self.opd_func = self.lens.opd_func
        else:
            self.opd_func = lambda r: np.zeros_like(r)

        #Get input diameter
        if self.input_diameter is not None:
            self.D1 = self.input_diameter
        else:
            #Get from lens
            self.D1 = self.lens.diameter

        #Determine scalings
        if self.input_dx is not None:
            self.dx = self.input_dx
        else:
            self.dx = self.D1 / self.num_pts
