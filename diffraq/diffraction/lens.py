"""
lens.py

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

class Lens(object):

    zemax_dir = f'{diffraq.ext_data_dir}/Lenses'

    def __init__(self, name, zemax_dir=None):
        self.mm2m = 1e-3
        self.m2mm = 1e3

        self.name = name

        #See if lens is simple lens
        if self.name.startswith('simple'):
            self.load_simple_lens()
        else:
            self.load_zemax_lens()

        #Build OPD
        self.build_OPD_function()

############################################
#####  Loading #####
############################################

    def load_zemax_lens(self):

        #Get EFL from name
        self.efl = float(self.name.split('-')[1])*1e-3

        #Loop through file and get surfaces
        surfaces = {}
        surf = Surface('dummy')
        in_surf = False
        with open(f'{self.zemax_dir}/{self.name}.zmx', 'r') as f:
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

        #Count number of true surfaces
        self.num_surf = len(self.surfaces.keys()) - 2

        #Check diameters are the same
        if not np.isclose(0, np.std([self.surfaces[f'SURF {i+1}'].diameter for i in range(self.num_surf)])):
            print('All diameters not the same')

        #Get diameter + thickness
        self.diameter = self.surfaces['SURF 1'].diameter
        self.thickness = self.surfaces['SURF 1'].thickness + self.surfaces['SURF 2'].thickness

        #Simple lens flag
        self.is_simple = False

    ############################################

    def load_simple_lens(self):

        #Get focal length from name
        self.focal_length = float(self.name.split('simple_')[-1]) * 1e-3

        self.diameter = 0
        self.thickness = 0

        #Simple lens flag
        self.is_simple = True

############################################
############################################

############################################
#####  OPD #####
############################################

    def build_OPD_function(self):

        if self.is_simple:
            #Simple lens
            self.opd_func = lambda r: -r**2/(2*self.focal_length)

        else:
            #Zemax lens
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
