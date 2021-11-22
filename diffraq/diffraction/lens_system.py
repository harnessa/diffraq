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

from diffraq.diffraction import Lens_Element
import numpy as np

class Lens_System(object):

    def __init__(self, elements, parent):
        self.parent = parent
        self.build_system(elements)

    def build_system(self, elements):

        #If elements not specified, build dummy system
        if elements is None:

            #Build parameters from sim
            pms = {'kind':'lens', 'lens_name':'simple',
                'focal_length':self.parent.sim.focal_length,\
                'diameter':self.parent.sim.tel_diameter, \
                'distance':self.parent.image_distance - self.parent.sim.defocus} #defocus will get added

            self.n_elements = 1
            self.element_0 = Lens_Element(pms, self.parent.num_pts, is_last=True)

        else:

            #Number of elements
            self.n_elements = len(elements)

            #Build element classes
            for i in range(self.n_elements):
                #Get parameters from dictionary
                pms = elements[f'element_{i}']

                #Build and store element
                elem = Lens_Element(pms, self.parent.num_pts, is_last=(i==self.n_elements-1))
                setattr(self, f'element_{i}', elem)
