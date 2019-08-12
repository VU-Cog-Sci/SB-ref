import os, json
import sys, glob
import re 

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import imageio
from skimage import color
import cv2
from skimage.transform import rescale
from skimage.filters import threshold_triangle

from scipy import ndimage
from nipy.modalities.fmri import hrf as nipy_hrf

from nilearn import surface

from utils import * #import script to use relevante functions
from prf_fit_FN import * #import script to use relevante functions


# define participant number and open json parameter file
if len(sys.argv)<2: 
    raise NameError('Please add subject number (ex:1) ' 
                    'as 1st argument in the command line!') 

elif len(sys.argv)<3:
    raise NameError('Please select server being used (ex: aeneas or cartesius) ' 
                    'as 2nd argument in the command line!') 
else:   
    sj = int(sys.argv[1])
    ses = 1 # it's always in the first session #int(sys.argv[2])
    print('FN data will be loaded for sub-%d ses-%d'%(sj,ses))



with open('analysis_params.json','r') as json_file: 
        analysis_params = json.load(json_file) 

# for linux computer
screen = [1920, 1080]
task='fn'

# alter screen res for laptop subjects
laptop_subs = [(1,1),(3,1),(4,1),(5,1),(5,2),(6,1),(7,1),(8,1),(9,1)]
for _,data in enumerate(laptop_subs):
    if (sj,ses)==data:
        screen = [1680,1050]

# scaling factor of screenshots
fig_sfactor = 0.1  

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

bound_grids  = (x_grid_bound, y_grid_bound, sigma_grid_bound)
bound_fits = (x_fit_bound, y_fit_bound, sigma_fit_bound, beta_fit_bound, baseline_fit_bound)

               
# set data paths  
if str(sys.argv[2]) == 'cartesius':
    filepath = glob.glob(os.path.join(analysis_params['post_fmriprep_outdir_cartesius'],'fn','sub-{sj}'.format(sj=sj),'*'))
    print('functional files from %s' %os.path.split(filepath[0])[0])
    data_dir = os.path.join(analysis_params['eyetrack_dir_cartesius'],'fn','sub-{sj}'.format(sj=str(sj).zfill(2)))
    print('eyetracking files from %s' %data_dir)

elif str(sys.argv[2]) == 'aeneas':
    filepath = glob.glob(os.path.join(analysis_params['post_fmriprep_outdir'],'fn','sub-{sj}'.format(sj=sj),'*'))
    print('functional files from %s' %os.path.split(filepath[0])[0])
    data_dir = os.path.join(analysis_params['eyetrack_dir'],'fn','sub-{sj}'.format(sj=str(sj).zfill(2)))
    print('eyetracking files from %s' %data_dir)

# use smoothed data?        
with_smooth = analysis_params['with_smooth']
    
# define list of files
# changes depending on data used
if with_smooth=='True':
    # list of functional files
    func_filename = [run for run in filepath if 'fn' in run and 'fsaverage' in run and run.endswith('smooth5.mgz')]
else:
    # list of functional files
    func_filename = [run for run in filepath if 'fn' in run and 'fsaverage' in run and run.endswith('_sg_conf.mgz')]
    
func_filename.sort()

# set hrf to be in same sample ratio
hrf_timepoints = np.arange(0, 32, 1/analysis_params['eyetrack_subsmp_freq']) # 32 randomly chosen, tail of hrf
hrf = nipy_hrf.spm_hrf_compat(t=hrf_timepoints)

# do loop for all runs
num_runs = np.arange(1,11)

for run in num_runs:
    try:
        # load saccade files
        eye_filename = os.path.join(data_dir,'sacc4dm_run-{run}.npz'.format(run=str(run).zfill(2)))
        fileinfo = os.path.join(data_dir,'gaze_timings_run-{run}.npz'.format(run=str(run).zfill(2)))

        # absolute path to save binary image numpy array
        img_filename = os.path.join(data_dir,'fn_bin_sub-{sj}_run-{run}.npy'.format(sj=str(sj).zfill(2),run=str(run).zfill(2)))

        if not os.path.exists(img_filename): #if not exists
            # get "long" DM(subsampled but not to TR)
            sacc2longDM(eye_filename,fileinfo,img_filename,smp_freq=analysis_params['eyetrack_smp_freq'],subsmp_freq=analysis_params['eyetrack_subsmp_freq'],
                        nrTR=analysis_params['FN_TRs'],TR=analysis_params['TR'],fig_sfactor=fig_sfactor,screen=screen)
        else:
            print('loading %s' %img_filename)

            fn_dm = np.load(img_filename)
            #fn_dm = fn_dm.T #swap axis for popeye (x,y,time)

            # make output folders, for each run
            # set data paths  
            if str(sys.argv[2]) == 'cartesius':
                output_path = os.path.join(analysis_params['fn_outdir_cartesius'],'sub-{sj}'.format(sj=str(sj).zfill(2)),'run-{run}'.format(run=str(run).zfill(2)))
                print('files will be saved in %s' %output_path)

            elif str(sys.argv[2]) == 'aeneas':
                output_path = os.path.join(analysis_params['fn_outdir'],'sub-{sj}'.format(sj=str(sj).zfill(2)),'run-{run}'.format(run=str(run).zfill(2)))
                print('files will be saved in %s' %output_path)

            if not os.path.exists(output_path): # check if path to save median run exist
                    os.makedirs(output_path) 

            # load data for run and fit each hemisphere at a time
            run_gii = [file for file in func_filename if 'run-{run}'.format(run=str(run).zfill(2)) in file]; run_gii.sort()

            for gii_file in run_gii: 

                print('loading data from %s' %gii_file)
                data = surface.load_surf_data(gii_file)

                # intitialize prf analysis
                FN = FN_fit(data = data.T,
                            fit_model = fit_model, 
                            visual_design = fn_dm, 
                            screen_distance = analysis_params["screen_distance"],
                            screen_width = analysis_params["screen_width"],
                            scale_factor = 1/2.0, 
                            tr =  TR,
                            bound_grids = bound_grids,
                            grid_steps = grid_steps,
                            bound_fits = bound_fits,
                            n_jobs = analysis_params['N_PROCS'],
                            hrf = hrf,
                            nr_TRs = analysis_params['FN_TRs'])
                
                
                # make/load predictions
                pred_out = gii_file.replace('.mgz','_predictions.npy')
                pred_out = os.path.join(output_path,os.path.split(pred_out)[-1])

                if not os.path.exists(pred_out): # if file doesn't exist

                    print('making predictions for %s' %pred_out) #create it
                    FN.make_predictions(out_file=pred_out)
                    
                else:
                    print('loading predictions %s' %pred_out)
                    FN.load_grid_predictions(prediction_file=pred_out)

                FN.grid_fit() # do grid fit

                # save outputs
                rsq_output = FN.gridsearch_r2
                params_output = FN.gridsearch_params.T

                #in estimates file
                estimates_out = pred_out.replace('_predictions.npy','_estimates.npz')
                np.savez(estimates_out, 
                         x=params_output[...,0],
                         y=params_output[...,1],
                         size=params_output[...,2],
                         baseline=params_output[...,3],
                         betas=params_output[...,4],
                         r2=rsq_output)


            
    except: 
        print('skipping run-%s' %run)
        pass






