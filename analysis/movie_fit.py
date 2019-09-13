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

# change number of threads used (cartesius can go up to 24) to speed up CPU
import mkl
mkl.set_num_threads(24)

# define participant number and open json parameter file
if len(sys.argv)<2: 
    raise NameError('Please add subject number (ex:1) ' 
                    'as 1st argument in the command line!') 

elif len(sys.argv)<3:
    raise NameError('Please select server being used (ex: aeneas or cartesius) ' 
                    'as 2nd argument in the command line!')

elif len(sys.argv)<4:
    raise NameError('Please state if fitting all runs (all) ' 
                    'or single runs (single)')
    
else:   
    sj = str(sys.argv[1]).zfill(2)
    ses = 1 # it's always in the first session #int(sys.argv[2])
    print('FN data will be loaded for sub-%d ses-%d'%(sj,ses))

fit_runs = str(sys.argv[3]) # string with type of fitting to do (all concatenated vs single-run)
json_dir = '/home/inesv/SB-ref/scripts/analysis_params.json' if str(sys.argv[2]) == 'cartesius' else 'analysis_params.json'

with open(json_dir,'r') as json_file: 
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
    filepath = glob.glob(os.path.join(analysis_params['post_fmriprep_outdir_cartesius'],'fn','sub-{sj}'.format(sj=str(sj).zfill(2)),'*'))
    print('functional files from %s' %os.path.split(filepath[0])[0])
    data_dir = os.path.join(analysis_params['eyetrack_dir_cartesius'],'fn','sub-{sj}'.format(sj=str(sj).zfill(2)))
    print('eyetracking files from %s' %data_dir)

elif str(sys.argv[2]) == 'aeneas':
    filepath = glob.glob(os.path.join(analysis_params['post_fmriprep_outdir'],'fn','sub-{sj}'.format(sj=str(sj).zfill(2)),'*'))
    print('functional files from %s' %os.path.split(filepath[0])[0])
    data_dir = os.path.join(analysis_params['eyetrack_dir'],'fn','sub-{sj}'.format(sj=str(sj).zfill(2)))
    print('eyetracking files from %s' %data_dir)

# use smoothed data?        
with_smooth = analysis_params['with_smooth']
    
# define list of files
# changes depending on data used
file_extension = '_sg_psc_smooth5.func.gii' if with_smooth=='True' else '_sg_psc.func.gii'
# list of functional files
func_filename = [run for run in filepath if 'fn' in run and 'fsaverage' in run and run.endswith(file_extension)]
func_filename.sort()

# set hrf timepoints according to subsample frequency
hrf_timepoints = np.arange(0, 32, 1/analysis_params['eyetrack_subsmp_freq']) # 32 randomly chosen, tail of hrf

# list all eyetracking files for subject
eye_filename = glob.glob(os.path.join(data_dir,'sacc4dm_run-*.npz')); eye_filename.sort()
fileinfo = glob.glob(os.path.join(data_dir,'gaze_timings_run-*.npz')); fileinfo.sort()
  
# list of strings with all run numbers that have eyetracking data (not necessarily 10 runs always)
all_runs = [os.path.splitext(eye_filename[r])[0][-6::] for r,_ in enumerate(eye_filename)]

# stack all DMs
fn_dm = []
for _,runstr in enumerate(all_runs):
    # absolute path to save binary image numpy array
    img_filename = os.path.join(data_dir,'fn_bin_sub-{sj}_'.format(sj=str(sj).zfill(2))+runstr+'.npy')

    if not os.path.exists(img_filename): #if not exists
        # get "long" DM(subsampled but not to TR) for that run
        eye_file_run = [w for _,w in enumerate(eye_filename) if runstr in w][0]
        info_file_run = [w for _,w in enumerate(fileinfo) if runstr in w][0]
        sacc2longDM(eye_file_run,info_file_run,img_filename,smp_freq=analysis_params['eyetrack_smp_freq'],
                    subsmp_freq=analysis_params['eyetrack_subsmp_freq'],
                    nrTR=analysis_params['FN_TRs'],TR=analysis_params['TR'],fig_sfactor=fig_sfactor,screen=screen)
    else:
        print('loading %s' %img_filename)
    
    # then stack
    fn_dm.append(np.load(img_filename).T) #swap axis for popeye (x,y,time)

# fit both hemispheres
hemi_label = ['hemi-L']#['hemi-L','hemi-R']

for _,hemi in enumerate(hemi_label):
    
    print('Fitting %s' %hemi)
    # absolute path of all runs of that hemisphere to be fitted
    hemi_file = [file for file in func_filename if any(runstr in file for runstr in all_runs) and hemi in file]
    
    if fit_runs == 'all':
        print('loading data from %s and concatenating' %all_runs)
        data_hemi_all = [np.concatenate([np.array(surface.load_surf_data(gii_file)) for _,gii_file in enumerate(hemi_file)], axis=-1)]
        
        # and concatenate hrf too because I need to use it for new concatenated time
        hrf = []
        for i in range(len(all_runs)):
            hrf.append(nipy_hrf.spm_hrf_compat(t=hrf_timepoints))
        hrf = np.concatenate(hrf,axis=-1)
        
        fn_dm = [np.concatenate(fn_dm,axis=-1)] # concatenate along time
        
        nr_TRs = analysis_params['FN_TRs']*len(all_runs)

    elif fit_runs == 'single':
        print('loading data from %s' %all_runs)
        data_hemi_all = np.stack([np.array(surface.load_surf_data(gii_file)) for _,gii_file in enumerate(hemi_file)], axis=0)
        
        hrf = nipy_hrf.spm_hrf_compat(t=hrf_timepoints)
        
        nr_TRs = analysis_params['FN_TRs']
         
    # now actually fit it
    for ind in range(len(data_hemi_all)):
        
        # make output folders, for each run
        folder = 'run-all' if fit_runs=='all' else all_runs[ind] # name of folder to save outputs
        
        ## set data paths  
        #if str(sys.argv[2]) == 'cartesius':
        #    output_path = os.path.join(analysis_params['fn_outdir_cartesius'],'sub-{sj}'.format(sj=str(sj).zfill(2)),folder)
        #elif str(sys.argv[2]) == 'aeneas':
        #    output_path = os.path.join(analysis_params['fn_outdir'],'sub-{sj}'.format(sj=str(sj).zfill(2)),folder)
        output_path = os.path.join(analysis_params['fn_outdir'],'sub-{sj}'.format(sj=str(sj).zfill(2)),folder)

        print('files will be saved in %s' %output_path)

        if not os.path.exists(output_path): # check if path to save run exist
            os.makedirs(output_path) 
            
        # intitialize prf analysis
        FN = FN_fit(data = data_hemi_all[ind].astype(np.float32), #to make the fitting faster
                    fit_model = fit_model, 
                    visual_design = fn_dm[ind], 
                    screen_distance = analysis_params["screen_distance"],
                    screen_width = analysis_params["screen_width"],
                    scale_factor = 1/2.0, 
                    tr =  TR,
                    bound_grids = bound_grids,
                    grid_steps = grid_steps,
                    bound_fits = bound_fits,
                    n_jobs = analysis_params['N_PROCS'],
                    hrf = hrf,
                    nr_TRs = nr_TRs)
        
        # make/load predictions
        pred_out = os.path.join(output_path,re.sub('run-\d{2}',folder,os.path.split(hemi_file[0])[-1])).replace('.func.gii','_predictions.npy')

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





