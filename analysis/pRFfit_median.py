#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Th Jun 6 11:13:11 2019

@author: inesverissimo

Do pRF fit on median run and save outputs
"""

import os, json
import sys, glob
import re

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
import scipy as sp
import scipy.stats as stats
import nibabel as nb
from nilearn.image import mean_img

from nilearn import surface

from utils import * #import script to use relevante functions
from prf_fit_lyon import * #import script to use relevante functions

# define participant number and open json parameter file
if len(sys.argv)<2:	
    raise NameError('Please add subject number (ex:01) '	
                    'as 1st argument in the command line!')	
   	 
else:	
    sj = str(sys.argv[1]).zfill(2) #fill subject number with 0 in case user forgets	

    with open('analysis_params.json','r') as json_file:	
            analysis_params = json.load(json_file)	
    
# use smoothed data?        
with_smooth = analysis_params['with_smooth']
    
# define paths and list of files
filepath = glob.glob(os.path.join(analysis_params['post_fmriprep_outdir'],'prf','sub-{sj}'.format(sj=sj),'*'))

# changes depending on data used
if with_smooth=='True':
    # list of functional files
    filename = [run for run in filepath if 'prf' in run and 'fsaverage' in run and run.endswith('_sg_smooth5.mgz')]
    # compute median run, per hemifield
    median_path = os.path.join(analysis_params['pRF_outdir'],'sub-{sj}'.format(sj=sj),'run-median','smooth')
else:
    # list of functional files
    filename = [run for run in filepath if 'prf' in run and 'fsaverage' in run and run.endswith('_sg.mgz')]
    # compute median run, per hemifield
    median_path = os.path.join(analysis_params['pRF_outdir'],'sub-{sj}'.format(sj=sj),'run-median')
    
filename.sort()
if not os.path.exists(median_path): # check if path to save median run exist
        os.makedirs(median_path) 


med_gii=[]
for field in ['hemi-L','hemi-R']:
    hemi = [h for h in filename if field in h]
    
    #if file doesn't exist
    abs_file = os.path.join(median_path,re.sub('run-\d{2}_','run-median_',os.path.split(hemi[0])[-1]))
    abs_file = re.sub('smooth5.mgz','smooth5',abs_file)
    if not os.path.exists(abs_file): 
        med_gii.append(median_mgz(hemi,median_path)) #create it
        print('computed %s' %(med_gii))
    else:
        med_gii.append(abs_file)
        print('median file %s already exists, skipping' %(med_gii))
        
        
# create/load design matrix 
png_filename = [os.path.join(analysis_params['imgs_dir'],png) for png in os.listdir(analysis_params['imgs_dir'])] 
png_filename.sort()

dm_filename = os.path.join(os.getcwd(),'prf_dm.npy')

if not os.path.exists(dm_filename): #if not exists
        screenshot2DM(png_filename,0.1,analysis_params['screenRes'],dm_filename) #create it
        print('computed %s' %(dm_filename))

prf_dm = np.load(dm_filename)
prf_dm = prf_dm.T #swap axis for popeye (x,y,time)
    

# define model params
fit_model = analysis_params["fit_model"]

# exception for these 2 subjects, TR was different
for string in ['sub-01_ses-01', 'sub-03_ses-01']:
    if re.search(string, filename[0]):
        TR = 1.5
    else:
        TR = analysis_params["TR"]

# Fit: define search grids
x_grid_bound = (-analysis_params["max_eccen"], analysis_params["max_eccen"])
y_grid_bound = (-analysis_params["max_eccen"], analysis_params["max_eccen"])
sigma_grid_bound = (analysis_params["min_size"], analysis_params["max_size"])
n_grid_bound = (analysis_params["min_n"], analysis_params["max_n"])
grid_steps = analysis_params["grid_steps"]

# Fit: define search bounds
x_fit_bound = (-analysis_params["max_eccen"]*2, analysis_params["max_eccen"]*2)
y_fit_bound = (-analysis_params["max_eccen"]*2, analysis_params["max_eccen"]*2)
sigma_fit_bound = (1e-6, 1e2)
n_fit_bound = (1e-6, 2)
beta_fit_bound = (-1e6, 1e6)
baseline_fit_bound = (-1e6, 1e6)

if fit_model == 'gauss' or fit_model == 'gauss_sg':
    bound_grids  = (x_grid_bound, y_grid_bound, sigma_grid_bound)
    bound_fits = (x_fit_bound, y_fit_bound, sigma_fit_bound, beta_fit_bound, baseline_fit_bound)
elif fit_model == 'css' or fit_model == 'css_sg':
    bound_grids  = (x_grid_bound, y_grid_bound, sigma_grid_bound, n_grid_bound)
    bound_fits = (x_fit_bound, y_fit_bound, sigma_fit_bound, n_fit_bound, beta_fit_bound, baseline_fit_bound)      
  

# load median data and fit each hemisphere at a time
for gii_file in med_gii: 
    print('loading data from %s' %gii_file)
    data = np.array(surface.load_surf_data(gii_file))
    
    # intitialize prf analysis
    prf = PRF_fit(data = data.T,
                fit_model = fit_model, 
                visual_design = prf_dm, 
                screen_distance = analysis_params["screen_distance"],
                screen_width = analysis_params["screen_width"],
                scale_factor = 1/2.0, 
                tr =  TR,
                bound_grids = bound_grids,
                grid_steps = grid_steps,
                bound_fits = bound_fits,
                n_jobs = analysis_params['N_PROCS'],
                sg_filter_window_length = analysis_params["sg_filt_window_length"],
                sg_filter_polyorder = analysis_params["sg_filt_polyorder"],
                sg_filter_deriv = analysis_params["sg_filt_deriv"], 
                )


    # make/load predictions
    pred_out = gii_file.replace('.npy','_predictions.npy')

    if not os.path.exists(pred_out): # if file doesn't exist

        print('making predictions for %s' %pred_out) #create it
        prf.make_predictions(out_file=pred_out)

    else:
        print('loading predictions %s' %pred_out)
        prf.load_grid_predictions(prediction_file=pred_out)

    prf.grid_fit() # do grid fit

    # save outputs
    rsq_output = prf.gridsearch_r2
    params_output = prf.gridsearch_params.T

    #in estimates file
    estimates_out = gii_file.replace('.npy','_estimates.npz')
    np.savez(estimates_out, 
             x=params_output[...,0],
             y=params_output[...,1],
             size=params_output[...,2],
             baseline=params_output[...,3],
             betas=params_output[...,4],
             r2=rsq_output)
