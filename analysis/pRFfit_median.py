#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Th Jun 6 11:13:11 2019

@author: inesverissimo

Do pRF fit on median run and save outputs
"""

import os
import json
import sys
import glob
import re

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
import scipy as sp
import scipy.stats as stats
import nibabel as nb
from nilearn.image import mean_img

from nilearn import surface

from utils import *  # import script to use relevante functions
from prf_fit_lyon import *  # import script to use relevante functions

# define participant number and open json parameter file
if len(sys.argv) < 2:
    raise NameError('Please add subject number (ex:1) '
                    'as 1st argument in the command line!')

elif len(sys.argv) < 3:
    raise NameError('Please select server being used (ex: aeneas or cartesius) '
                    'as 2nd argument in the command line!')

else:
    # fill subject number with 0 in case user forgets
    sj = str(sys.argv[1]).zfill(2)


json_dir = '/home/knapen/SB-ref/scripts/analysis_params.json' if str(
    sys.argv[2]) == 'cartesius' else 'analysis_params.json'

with open(json_dir, 'r') as json_file:
    analysis_params = json.load(json_file)

# use smoothed data?
with_smooth = analysis_params['with_smooth']


# define paths and list of files
if str(sys.argv[2]) == 'cartesius':
    filepath = glob.glob(os.path.join(
        analysis_params['post_fmriprep_outdir_cartesius'], 'prf', 'sub-{sj}'.format(sj=sj), '*'))
    print('functional files from %s' % os.path.split(filepath[0])[0])
    out_dir = analysis_params['pRF_outdir_cartesius']

elif str(sys.argv[2]) == 'aeneas':
    print(os.path.join(
        analysis_params['post_fmriprep_outdir'], 'prf', 'sub-{sj}'.format(sj=sj), '*'))
    filepath = glob.glob(os.path.join(
        analysis_params['post_fmriprep_outdir'], 'prf', 'sub-{sj}'.format(sj=sj), '*'))
    print('functional files from %s' % os.path.split(filepath[0])[0])
    out_dir = analysis_params['pRF_outdir']

# changes depending on data used
if with_smooth == 'True':
    # last part of filename to use
    file_extension = '_sg_psc_smooth%d.func.gii' % analysis_params['smooth_fwhm']
    # compute median run, per hemifield
    median_path = os.path.join(
        out_dir, 'sub-{sj}'.format(sj=sj), 'run-median', 'smooth%d' % analysis_params['smooth_fwhm'])
else:
    # last part of filename to use
    file_extension = '_sg_psc.func.gii'
    # compute median run, per hemifield
    median_path = os.path.join(out_dir, 'sub-{sj}'.format(sj=sj), 'run-median')

# list of functional files
filename = [run for run in filepath if 'prf' in run and 'fsaverage' in run and run.endswith(
    file_extension)]
filename.sort()
if not os.path.exists(median_path):  # check if path to save median run exist
    os.makedirs(median_path)


med_gii = []
for field in ['hemi-L', 'hemi-R']:
    hemi = [h for h in filename if field in h]

    # set name for median run (now numpy array)
    med_file = os.path.join(median_path, re.sub(
        'run-\d{2}_', 'run-median_', os.path.split(hemi[0])[-1]))
    # if file doesn't exist
    if not os.path.exists(med_file):
        med_gii.append(median_gii(hemi, median_path))  # create it
        print('computed %s' % (med_gii))
    else:
        med_gii.append(med_file)
        print('median file %s already exists, skipping' % (med_gii))


# create/load design matrix
png_path = '/home/inesv/SB-ref/scripts/imgs/' if str(
    sys.argv[2]) == 'cartesius' else analysis_params['imgs_dir']
png_filename = [os.path.join(png_path, png) for png in os.listdir(png_path)]
png_filename.sort()

dm_filename = os.path.join(os.getcwd(), 'prf_dm.npy')

if not os.path.exists(dm_filename):  # if not exists
    screenshot2DM(png_filename, 0.1,
                  analysis_params['screenRes'], dm_filename)  # create it
    print('computed %s' % (dm_filename))

else:
    print('loading %s' % dm_filename)

prf_dm = np.load(dm_filename)
prf_dm = prf_dm.T  # swap axis for popeye (x,y,time)


# define model params
fit_model = analysis_params["fit_model"]

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
    bound_grids = (x_grid_bound, y_grid_bound, sigma_grid_bound)
    bound_fits = (x_fit_bound, y_fit_bound, sigma_fit_bound,
                  beta_fit_bound, baseline_fit_bound)
elif fit_model == 'css' or fit_model == 'css_sg':
    bound_grids = (x_grid_bound, y_grid_bound, sigma_grid_bound, n_grid_bound)
    bound_fits = (x_fit_bound, y_fit_bound, sigma_fit_bound,
                  n_fit_bound, beta_fit_bound, baseline_fit_bound)


# load median data and fit each hemisphere at a time
for gii_file in med_gii:
    print('loading data from %s' % gii_file)
    data = np.array(surface.load_surf_data(gii_file))

    # intitialize prf analysis
    prf = PRF_fit(data=data,
                  fit_model=fit_model,
                  visual_design=prf_dm,
                  screen_distance=analysis_params["screen_distance"],
                  screen_width=analysis_params["screen_width"],
                  scale_factor=1/2.0,
                  tr=TR,
                  bound_grids=bound_grids,
                  grid_steps=grid_steps,
                  bound_fits=bound_fits,
                  n_jobs=analysis_params['N_PROCS'],
                  sg_filter_window_length=analysis_params["sg_filt_window_length"],
                  sg_filter_polyorder=analysis_params["sg_filt_polyorder"],
                  sg_filter_deriv=analysis_params["sg_filt_deriv"],
                  )

    # make/load predictions
    pred_out = gii_file.replace('.func.gii', '_predictions.npy')

    if not os.path.exists(pred_out):  # if file doesn't exist

        print('making predictions for %s' % pred_out)  # create it
        prf.make_predictions(out_file=pred_out)

    else:
        print('loading predictions %s' % pred_out)
        prf.load_grid_predictions(prediction_file=pred_out)

    # prf.grid_fit()  # do grid fit
    # in estimates file
    # estimates_out_filename = gii_file.replace('.func.gii', '_estimates.npz')

    # np.savez(estimates_out_filename,
    #          x=params_output[..., 0],
    #          y=params_output[..., 1],
    #          size=params_output[..., 2],
    #          baseline=params_output[..., 3],
    #          betas=params_output[..., 4],
    #          r2=rsq_output)

    loaded_gf_pars = np.load(gii_file.replace('.func.gii', '_estimates.npz'))

    prf.gridsearch_params = np.array(
        [loaded_gf_pars[par] for par in ['x', 'y', 'size', 'betas', 'baseline']])
    prf.gridsearch_r2 = loaded_gf_pars['r2']

    prf.iterative_fit()  # do iterative fit

    # save outputs
    rsq_output = prf.gridsearch_r2
    params_output = prf.gridsearch_params.T

    iterative_out = gii_file.replace('.func.gii', '_iterative_output.npz')
    np.savez(iterative_out,
             fit_output=prf.fit_output)
