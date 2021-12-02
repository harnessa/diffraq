"""
simulator.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Main class to control simulations for the DIFFRAQ python package.
    #TODO: ADD MORE

"""

import diffraq
import numpy as np

class Simulator(object):

    def __init__(self, params={}, shapes=[]):
        #Set parameters
        self.set_parameters(params)

        #Load children
        self.load_children(shapes)

############################################
####	Initialize  ####
############################################

    def set_parameters(self, params):
        #Set parameters
        diffraq.utils.misc_util.set_default_params(self, params, diffraq.utils.def_sim_params)
        self.params = diffraq.utils.misc_util.deepcopy(params)

        ### Derived ###
        if self.save_dir_base is None:
            self.save_dir_base = f"{diffraq.pkg_home_dir}/Results"

        self.waves = np.atleast_1d(self.waves)
        self.z_scl = self.z0 / (self.z0 + self.zz)

        #Number of polarization dimensions
        if self.do_run_vector:
            self.npol = 3
        else:
            self.npol = 1

    def load_children(self, shapes):
        #Logging + Saving
        self.logger = diffraq.utils.Logger(self)

        #Occulter
        self.occulter = diffraq.geometry.Occulter(self, shapes)

        #Braunbek seam for vector calculation
        if self.do_run_vector:
            self.vector = diffraq.polarization.VectorSim(self, self.occulter.shapes)
        else:
            self.vector = None

        #Open flag
        self.shop_is_open = False

    def load_focuser(self):
        #Load focuser child
        self.focuser = diffraq.diffraction.Focuser(self)

############################################
############################################

############################################
##### Open/Close Up Shop #####
############################################

    def open_up_shop(self):
        #Run start ups
        self.logger.start_up()

        #Print
        self.logger.print_start_message()

        #Open flag
        self.shop_is_open = True

    def close_up_shop(self):
        if not self.shop_is_open:
            return

        #Print
        self.logger.print_end_message()

        #Run close ups
        self.logger.close_up()

        #Empty trash
        self.clean_up()

        #Close flag
        self.shop_is_open = False

    def clean_up(self):
        #Return if not freeing
        if not self.free_on_end:
            return

        #Delete trash
        trash_list = ['pupil', 'image', 'vec_pupil', 'image_pts']
        for att in trash_list:
            if hasattr(self, att):
                delattr(self, att)

        #Cleanup children
        self.occulter.clean_up()
        if self.vector is not None:
            self.vector.clean_up()

############################################
############################################

############################################
####	Main Script  ####
############################################

    def run_sim(self):

        #Open shop
        self.open_up_shop()

        #Calculate pupil field
        self.run_pupil_field()

        #Calculate image field
        self.run_image_field()

        #Close shop
        self.close_up_shop()

############################################
############################################

############################################
####	Pupil Field  ####
############################################

    def run_pupil_field(self):
        #Load pupil instead of calculating?
        pupil_loaded = self.load_pupil_field()

        #If pupil loaded, return
        if pupil_loaded:
            return

        #Calculate pupil field
        self.pupil, self.grid_pts = self.calc_pupil_field()

        #Save pupil field
        self.logger.save_pupil_field(self.pupil, self.grid_pts, \
            self.vec_pupil, self.vec_comps)

    ###########################

    def load_pupil_field(self):
        #Return immediately if not loading
        if not self.do_load_pupil:
            return False

        #Check if pupil file exists
        if not self.logger.pupil_file_exists():
            return False

        #Load pupil
        self.pupil, self.grid_pts, self.vec_pupil, self.vec_comps, is_polarized = \
            self.logger.load_pupil_field()

        #Return True
        return True

    ###########################

    def calc_pupil_field(self):

        #Build target
        grid_pts = diffraq.utils.image_util.get_grid_points(self.num_pts, self.tel_diameter)

        #Run scalar diffraction calculation over the occulter
        pupil = self.scalar_diffraction_calculation(grid_pts)

        #If vector calculation, calculate diffraction over Braunbek seam
        if self.do_run_vector:
            #Cleanup occulter to free memory
            self.occulter.clean_up()

            #Run calculation (Save, but don't return)
            self.vec_pupil = self.vector_diffraction_calculation(grid_pts)

            #Get incident polarization components
            self.vec_comps = np.array([self.vector.Ex_comp, self.vector.Ey_comp])

        else:
            self.vec_pupil, self.vec_comps = None, None

        #Get target points
        grid_pts = self.get_target_points(grid_pts)

        return pupil, grid_pts

############################################
############################################

############################################
####	Diffraction Calculations  ####
############################################

    def scalar_diffraction_calculation(self, grid_pts):
        """Calculate the scalar diffraction of the occulter's quadrature x+y and
            quadrature weights, over the supplied grid. """

        #Build Area Quadrature
        self.occulter.build_quadrature()

        #Create empty pupil field array
        pupil = np.empty((len(self.waves), self.num_pts, self.num_pts)) + 0j

        #Adjust occulter values if off_axis (shift doesn't go into beam function)
        xq, yq, xoff = self.get_offaxis_points(self.occulter.xq, self.occulter.yq, grid_pts)

        #Run diffraction calculation over wavelength
        for iw in range(len(self.waves)):

            #lambda * z
            lamzz = self.waves[iw] * self.zz
            lamz0 = self.waves[iw] * self.z0

            #Apply input beam function
            if self.beam_function is not None:
                wq = self.occulter.wq * self.beam_function(self.occulter.xq, \
                    self.occulter.yq, self.waves[iw])
            else:
                wq = self.occulter.wq

            #Calculate diffraction
            uu = diffraq.diffraction.diffract_grid(xq, yq, wq, lamzz, grid_pts, \
                self.fft_tol, lamz0=lamz0, is_babinet=self.occulter.is_babinet)

            #Account for extra phase added by off_axis
            uu *= np.exp(1j*np.pi/lamz0*self.z_scl * xoff)

            #Multiply by plane wave
            uu *= np.exp(1j * 2*np.pi/self.waves[iw] * self.zz)

            #Store
            pupil[iw] = uu

        #Cleanup
        del uu, xq, yq, wq

        return pupil

    ############################################
    ############################################

    def vector_diffraction_calculation(self, grid_pts, is_babinet=False):
        """Calculate the scalar diffraction of the vector seam's quadrature x+y,
            quadrature weights, and additive vector field, over the supplied grid. """

        #Build Area Quadrature
        self.vector.build_quadrature()

        #Create empty pupil field array
        pupil = np.empty((len(self.waves), self.npol, self.num_pts, self.num_pts)) + 0j

        #Adjust occulter values if off_axis (shift doesn't go into beam function)
        xq, yq, xoff = self.get_offaxis_points(self.vector.xq, self.vector.yq, grid_pts)

        #Get edge normal components (relative to horizontal -- differs from Harness OE 2020)
        cosa = np.cos(self.vector.nq)
        sina = np.sin(self.vector.nq)
        del self.vector.nq

        #Run diffraction calculation over wavelength
        for iw in range(len(self.waves)):

            #lambda * z
            lamzz = self.waves[iw] * self.zz
            lamz0 = self.waves[iw] * self.z0

            #Get incident field
            sfld, pfld = self.vector.screen.get_edge_field(self.vector.dq, \
                self.vector.gw, self.waves[iw])

            #Apply input beam function
            if self.beam_function is not None:
                w_beam = self.beam_function(self.vector.xq, self.vector.yq, self.waves[iw])
                sfld *= w_beam
                pfld *= w_beam
            else:
                w_beam = None       #for cleanup

            #Get electric field components
            Ex = self.vector.Ex_comp * (sfld*sina**2 + pfld*cosa**2) + \
                self.vector.Ey_comp * (sina*cosa * (pfld - sfld))

            Ey = self.vector.Ey_comp * (sfld*cosa**2 + pfld*sina**2) + \
                self.vector.Ex_comp * (sina*cosa * (pfld - sfld))

            #Loop over horizontal and vertical polarizations
            for ip in range(self.npol):

                #Build quadrature weights * incident field
                if ip == 0:
                    wu0 = self.vector.wq * Ex
                elif ip == 1:
                    wu0 = self.vector.wq * Ey
                else:
                    wu0 = (-1/self.zz) * self.vector.wq * (xq*Ex + yq*Ey)

                #Calculate diffraction
                uu = diffraq.diffraction.diffract_grid(xq, yq, wu0, lamzz, \
                    grid_pts, self.fft_tol, lamz0=lamz0, is_babinet=is_babinet)

                #Account for extra phase added by off_axis
                uu *= np.exp(1j*np.pi/lamz0*self.z_scl * xoff)

                #Multiply by plane wave
                uu *= np.exp(1j * 2*np.pi/self.waves[iw] * self.zz)

                #Store
                pupil[iw,ip] = uu

        #Cleanup
        del wu0, uu, sfld, pfld, cosa, sina, xq, yq, w_beam, Ex, Ey
        if self.free_on_end:
            self.vector.clean_up()

        return pupil

    ############################################
    ############################################

    def get_offaxis_points(self, xx, yy, grid_pts):
        #Return early if not off-axis
        if np.isclose(0, np.hypot(*self.target_center)):
            return xx, yy, 0

        #Adjust occulter/vector values if off_axis (shift doesn't go into beam function)
        newx = xx - self.target_center[0] * self.z_scl
        newy = yy - self.target_center[1] * self.z_scl
        xoff = 2*(grid_pts*self.target_center[0] + grid_pts[:,None]*self.target_center[1])
        xoff += np.hypot(*self.target_center)**2

        return newx, newy, xoff

    ############################################

    def get_target_points(self, grid_pts):
        #Return early if not off-axis
        if np.isclose(0, np.hypot(*self.target_center)):
            return grid_pts

        grid_2D = np.tile(grid_pts, (len(grid_pts),1)).T
        xi = grid_2D.flatten() + self.target_center[1]
        eta = grid_2D.T.flatten() + self.target_center[0]
        del grid_2D
        return xi, eta

############################################
############################################

############################################
####	Image Field  ####
############################################

    def run_image_field(self):
        #Return immediately if running pupil only
        if self.skip_image:
            return

        #Load focuser child
        self.load_focuser()

        #If vector calculation, calculate image for each polarization component
        if self.do_run_vector:

            #Build total field for each polarization component
            pupil = self.vec_pupil.copy()
            for i in range(len(self.vec_comps)):
                pupil[:,i] += self.pupil * self.vec_comps[i]

            #Cacluate image
            self.image, self.image_pts = self.focuser.calculate_image(pupil, self.grid_pts)

        else:

            #Calculate scalar image
            self.image, self.image_pts = \
                self.focuser.calculate_image(self.pupil[:,None], self.grid_pts)

        #Save image
        self.logger.save_image_field(self.image, self.image_pts)

############################################
############################################
