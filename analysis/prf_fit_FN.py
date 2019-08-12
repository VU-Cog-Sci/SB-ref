
# adaptation on popeyes RF fitting, but then I can have the hrf sampled to eyetracking acq rate into account
# only done for gaussian fit


import ctypes
import numpy as np
import scipy as sp
import platform
from math import *
import os
import glob
import json

# MRI analysis imports
import nibabel as nb
import popeye.utilities as utils
from popeye.visual_stimulus import VisualStimulus
import popeye.css as css
import popeye.og as og
import cifti
from joblib import Parallel, delayed
from scipy import signal

from scipy.signal import fftconvolve#, savgol_filter
from scipy.stats import linregress
from popeye.base import PopulationModel
from popeye.spinach import generate_og_receptive_field, generate_rf_timeseries_nomask

from tqdm import tqdm


class GaussianModel(PopulationModel):
    
    r"""
    A Gaussian Spatial Summation population receptive field model class
    
    """
    
    def __init__(self, stimulus, hrf, cached_model_path=None, nuisance=None, nr_TRs=103):
                
        PopulationModel.__init__(self, stimulus, hrf, nuisance,nr_TRs)

    # main method for deriving model time-series
    def generate_prediction(self, x, y, sigma, beta, baseline,hrf,nr_TRs):
        
        # generate the RF
        rf = generate_og_receptive_field(
            x, y, sigma, self.stimulus.deg_x, self.stimulus.deg_y)
        
        # normalize by the integral
        rf /= ((2 * np.pi * sigma**2) * 1 /
               np.diff(self.stimulus.deg_x[0, 0:2])**2)
        
        # extract the stimulus time-series
        response = generate_rf_timeseries_nomask(self.stimulus.stim_arr, rf)

        # convolve HRF with the stimulus
        model = fftconvolve(response, hrf)[0:len(response)]
        
        # resample to TR (because hrf and stim in sample frequency)
        model = signal.resample(model, num=nr_TRs, axis=0)
        
        # units
        model /= np.max(model)
        
        # offset
        model += baseline
        
        # scale it by beta
        model *= beta

        return model



class FN_fit(object):
    
    def __init__(self, data, fit_model, visual_design, screen_distance, screen_width, 
                 scale_factor, tr, bound_grids, grid_steps, bound_fits, n_jobs, hrf,
                 sg_filter_window_length=210, sg_filter_polyorder=3, sg_filter_deriv=0,nr_TRs = 103):

        # immediately convert nans to nums
        self.data = np.nan_to_num(data)
        self.data_var = self.data.var(axis=-1)

        self.n_units = self.data.shape[0]
        self.n_timepoints = self.data.shape[1]

        self.hrf = hrf
        self.nr_TRs = nr_TRs

        self.stimulus = VisualStimulus( stim_arr = visual_design,
                                        viewing_distance = screen_distance, 
                                        screen_width = screen_width,
                                        scale_factor = scale_factor,
                                        tr_length = tr,
                                        dtype = np.short)

        #assert self.n_timepoints == self.stimulus.run_length, \
        #    "Data and design matrix do not have the same nr of timepoints, %i vs %i!"%(self.n_timepoints, self.stimulus.run_length)

        if fit_model == 'gauss':
            self.model_func = GaussianModel(stimulus = self.stimulus, 
                                            hrf = self.hrf,
                                           nr_TRs = self.nr_TRs)    

        self.model_func.hrf_delay = 0
        self.predictions = None      
        self.fit_model =  fit_model
        self.bound_grids = bound_grids
        self.grid_steps = grid_steps
        self.bound_fits = bound_fits
        self.n_jobs = n_jobs
        
    def make_grid(self):
        prf_xs = np.linspace(self.bound_grids[0][0],self.bound_grids[0][1],self.grid_steps)
        prf_ys = np.linspace(self.bound_grids[1][0],self.bound_grids[1][1],self.grid_steps)
        prf_sigma = np.linspace(self.bound_grids[2][0],self.bound_grids[2][1],self.grid_steps)
        
        if self.fit_model == 'gauss':
            self.prf_xs, self.prf_ys, self.prf_sigma = np.meshgrid(prf_xs, prf_ys, prf_sigma)
    
    def make_predictions(self, out_file=None):
        if not hasattr(self, 'prf_xs'):
            self.make_grid()
        self.predictions = np.zeros(list(self.prf_xs.shape) + [self.nr_TRs])#[self.stimulus.run_length])
        self.predictions = self.predictions.reshape(-1, self.predictions.shape[-1]).T
        print(' predictions %s' %str(self.predictions.shape))
        if self.fit_model == 'gauss':
            for i, (x, y, s) in tqdm(enumerate(zip(self.prf_xs.ravel(), self.prf_ys.ravel(), self.prf_sigma.ravel()))):
                #print('%.6f %.6f %.6f' %(x,y,s))
                self.predictions[:, i] = self.model_func.generate_prediction(x, y, s, 1, 0, self.hrf,self.nr_TRs)
        self.predictions = np.nan_to_num(self.predictions)
        if out_file != None:
            np.save(out_file, self.predictions)

    def load_grid_predictions(self, prediction_file):
        self.make_grid()
        self.predictions = np.load(prediction_file)

    def grid_fit(self):
        
        if self.fit_model == 'gauss':
            prediction_params = np.ones((self.n_units, 6))*np.nan

        # set up book-keeping to minimize memory usage.
        self.gridsearch_r2 = np.zeros(self.n_units)
        self.best_fitting_prediction_thus_far = np.zeros(self.n_units, dtype=int)
        self.best_fitting_beta_thus_far = np.zeros(self.n_units, dtype=float)
        self.best_fitting_baseline_thus_far = np.zeros(self.n_units, dtype=float)

        for prediction_num in tqdm(range(self.predictions.shape[1])):
            # scipy implementation?
            # slope, intercept, rs, p_values, std_errs = linregress(self.predictions[:,prediction_num], self.data)
            # rsqs = rs**2
            # numpy implementation is slower?
            dm = np.vstack([np.ones_like(self.predictions[:,prediction_num]),self.predictions[:,prediction_num]]).T
            (intercept, slope), residual, _, _ = sp.linalg.lstsq(dm, self.data.T, check_finite=False) #  , lapack_driver='gelsy')
<<<<<<< HEAD
            
            if bool(residual)==True: #if residual not empty
                rsqs = ((1 - residual / (self.n_timepoints * self.data_var)))

                improved_fits = rsqs > self.gridsearch_r2
                # fill in the improvements
                self.best_fitting_prediction_thus_far[improved_fits] = prediction_num
                self.gridsearch_r2[improved_fits] = rsqs[improved_fits]
                self.best_fitting_baseline_thus_far[improved_fits] = intercept[improved_fits]
                self.best_fitting_beta_thus_far[improved_fits] = slope[improved_fits]
=======
            rsqs = ((1 - residual / (self.n_timepoints * self.data_var)))

            improved_fits = rsqs > self.gridsearch_r2
            # fill in the improvements
            self.best_fitting_prediction_thus_far[improved_fits] = prediction_num
            self.gridsearch_r2[improved_fits] = rsqs[improved_fits]
            self.best_fitting_baseline_thus_far[improved_fits] = intercept[improved_fits]
            self.best_fitting_beta_thus_far[improved_fits] = slope[improved_fits]
>>>>>>> 661796100b2750fff8f24312b8d2147201abdd32

        if self.fit_model == 'gauss':
            self.gridsearch_params = np.array([ self.prf_xs.ravel()[self.best_fitting_prediction_thus_far],
                                                    self.prf_ys.ravel()[self.best_fitting_prediction_thus_far],
                                                    self.prf_sigma.ravel()[self.best_fitting_prediction_thus_far],
                                                    self.best_fitting_beta_thus_far,
                                                    self.best_fitting_baseline_thus_far
                                                ])
