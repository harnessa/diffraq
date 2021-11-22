"""
lens_element.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 11-20-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class to hold physical properties of a lens for angular spectrum
    propagation.

"""

import diffraq
import numpy as np

class Lens_Element(object):

    zemax_dir = f'{diffraq.ext_data_dir}/Lenses'
    mm2m = 1e-3
    m2mm = 1e3

    def __init__(self, params, num_pts, is_last=False):
        #Set parameters
        self.num_pts = num_pts
        self.set_parameters(params)

        #Is last element in system
        self.is_last = is_last

    def set_parameters(self, params):
        def_pms = {
            'kind':             '',
            'lens_name':        '',
            'diameter':         None,
            'distance':         1e-12,
            'focal_length':     1e-12,
        }

        #Parameters
        for k, v in def_pms.items():
            setattr(self, k, v)

        for k, v in params.items():
            setattr(self, k, v)

        #Build depending on kind
        if self.kind == 'aperture':
            self.load_aperture()

        elif self.lens_name == 'simple':
            self.load_simple_lens()

        else:
            self.load_zemax_lens()

        #Sampling
        self.dx = self.diameter / self.num_pts

############################################
#####  Simple elements #####
############################################

    def load_aperture(self):
        #OPD function
        self.opd_func = lambda r: np.zeros_like(r)

    ############################################

    def load_simple_lens(self):
        #OPD function
        self.opd_func = lambda r: -r**2/(2*self.focal_length)

############################################
############################################

############################################
#####  Zemax Lens #####
############################################

    def load_zemax_lens(self):

        #Get EFL from name
        self.focal_length = float(self.lens_name.split('-')[1])*1e-3

        #Loop through file and get surfaces
        surfaces = {}
        surf = Surface('dummy')
        in_surf = False
        with open(f'{self.zemax_dir}/{self.lens_name}.zmx', 'r') as f:
            #Loop through file
            for ln in f:
                #Start new surface
                if ln.startswith('SURF'):
                    #Store last surface
                    surfaces[surf.name] = surf
                    #Start new surface
                    surf = Surface(ln.split('\n')[0])
                    #Say we are inside surface
                    in_surf = True
                elif in_surf and ln.startswith('  '):
                    #Only Store a few keys
                    if ln.startswith('  CURV'):
                        curv = float(ln[7:].split(' ')[0])    #file is curvature, radius of curvature = 1/cruvature
                        if curv != 0:
                            surf.curvature = self.mm2m/curv
                        else:
                            surf.curvature = np.inf
                    elif ln.startswith('  DISZ'):
                        thk = ln[7:].split('\n')[0]
                        if thk == 'INFINITY':
                            thk = np.inf
                        surf.thickness = self.mm2m*float(thk)
                    elif ln.startswith('  GLAS'):
                        surf.glass = ln[7:].split(' ')[0].strip('\n')
                    elif ln.startswith('  DIAM'):
                        surf.diameter = self.mm2m*2*float(ln[7:].split(' ')[0])   #Files give radius
                else:
                    in_surf = False

        #Add last surface
        surfaces[surf.name] = surf

        #Pop out dummy
        surfaces.pop('dummy')

        #Store surfaces
        self.surfaces = surfaces

        #Take lens diameter if not given
        if self.diameter is None:
            self.diameter = self.surfaces['SURF 1'].diameter

        #Build OPD function
        R1 = abs(self.surfaces['SURF 1'].curvature)
        R2 = abs(self.surfaces['SURF 2'].curvature)
        R3 = abs(self.surfaces['SURF 3'].curvature)
        t12 = self.surfaces['SURF 1'].thickness
        t23 = self.surfaces['SURF 2'].thickness
        n1 = self.surfaces['SURF 1'].index
        n2 = self.surfaces['SURF 2'].index

        self.opd_func = lambda r: \
            (R1 - np.sqrt(R1**2 - r**2))*(1. - n1) + \
            (R2 - np.sqrt(R2**2 - r**2))*(n2 - n1) + \
            (R3 - np.sqrt(R3**2 - r**2))*(1. - n2) + \
            t12*n1 + t23*n2

############################################
############################################

############################################
#####  Surface Class #####
############################################

class Surface(object):

    def __init__(self, name):
        self.name = name
        self.curvature = 0
        self.thickness = 0
        self.glass = ''

        #Assuming wave = 680 nm  #from refractiveindex.info
        self.index_dict = {'':1, 'N-BK7':1.514 , 'N-BAF10':1.665, \
            'N-SF6HT':1.794, 'SF5':1.665, 'SF2':1.641, 'F2HT':1.614, 'N-FK5':1.485}

    @property
    def index(self):
        return self.index_dict[self.glass]

############################################
############################################
