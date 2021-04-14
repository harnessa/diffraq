"""
dual_analyzer.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 04-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class to analyze results from two separate DIFFRAQ simulations.
"""

import numpy as np
from diffraq.analysis import Analyzer
import copy
import matplotlib.pyplot as plt

class Dual_Analyzer(object):

    def __init__(self, params={}):
        #Get split parameters
        pms1, pms2 = self.set_parameters(params)

        #Load analyzers
        self.alz1 = Analyzer(pms1)
        self.alz2 = Analyzer(pms2)

    def set_parameters(self, params):
        def_duo = {
            'load_dir_base_1':      None,
            'session_1':            '',
            'load_ext_1':           '',
            'load_dir_base_2':      None,
            'session_2':            None,
            'load_ext_2':           '',
            'show_images':          False,
        }

        #Get full parameter set
        params = {**def_duo, **params}

        #Build new params for the analyzers
        pms1 = copy.deepcopy(params)
        pms2 = copy.deepcopy(params)

        for k in ['load_dir_base', 'session', 'load_ext']:
            #Copy over with number stripped
            pms1[k] = params[f'{k}_1']

            #If None for #2, use #1 value
            if params[f'{k}_2'] is None:
                pms2[k] = params[f'{k}_1']
            else:
                pms2[k] = params[f'{k}_2']

        #Look for duo params
        for k in list(params):
            if k in def_duo.keys():
                #Delete duo params
                del pms1[k]
                del pms2[k]

                #Set as attribute
                setattr(self, k, params[k])

        return pms1, pms2

############################################
####	Main Script ####
############################################

    def show_results(self):

        plt.ion()
        fig, axes = plt.subplots(1,2, figsize=(8,5), sharex=True, sharey=True)

        for i in range(2):
            axes[i].imshow(getattr(self, f'alz{i+1}').image)
            axes[i].set_title(getattr(self, f'load_ext_{i+1}'))


    def clean_up(self):
        self.alz1.clean_up()
        self.alz2.clean_up()

############################################
############################################
