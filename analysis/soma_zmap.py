#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Th Jun 6 11:13:11 2019

@author: inesverissimo

Do SOMA contrasts and save outputs
"""

import os, json
import sys, glob

import numpy as np
import pandas as pd


from nilearn.signal import clean
from nistats.design_matrix import make_first_level_design_matrix
from nistats.first_level_model import run_glm
from nistats.contrasts import compute_contrast

from utils import * #import script to use relevante functions


# define participant number and open json parameter file
if len(sys.argv)<2:	
    raise NameError('Please add subject number (ex:01) '	
                    'as 1st argument in the command line!')	
   	 
else:	
    sj = str(sys.argv[1]).zfill(2) #fill subject number with 0 in case user forgets	

    with open('analysis_params.json','r') as json_file:	
            analysis_params = json.load(json_file)	
    

# define paths and list of files
filepath = glob.glob(os.path.join(analysis_params['fmriprep_dir'],'sub-{sj}'.format(sj=sj),'*','func/*'))
eventpath = glob.glob(os.path.join(analysis_params['sourcedata_dir'],'sub-{sj}'.format(sj=sj),'*','func/*'))

# list of functional files
filename = [run for run in filepath if 'soma' in run and 'fsaverage' in run and run.endswith('.func.gii')]
filename.sort()
# list of confounds
confounds = [run for run in filepath if 'soma' in run and run.endswith('_desc-confounds_regressors.tsv')]
confounds.sort()
# list of stimulus onsets
events = [run for run in eventpath if 'soma' in run and run.endswith('events.tsv')]
events.sort()


# high pass filter all runs

# exception for these 2 subjects, TR was different
TR = 1.5 if 'sub-01_ses-01' or 'sub-03_ses-01' in filename[0] else params["TR"]
# soma out path
soma_out = os.path.join(analysis_params['soma_outdir'],'sub-{sj}'.format(sj=sj))

# savgol filter giis and confounds
filt_gii = highpass_gii(filename,analysis_params['sg_filt_polyorder'],analysis_params['sg_filt_deriv'],
         analysis_params['sg_filt_window_length'],soma_out)

filt_conf = highpass_confounds(confounds,analysis_params['nuisance_columns'],analysis_params['sg_filt_polyorder'],analysis_params['sg_filt_deriv'],
                   analysis_params['sg_filt_window_length'],TR,soma_out)


# regress the confounds from signal
data = []

for idx,file in enumerate(filt_conf):
    confs = pd.read_csv(file, sep='\t', na_values='n/a')
    d = clean(filt_gii[idx], confounds=confs.values, standardize=False)
    
    data.append(d)

data = np.array(data) # cleaned data

# compute median soma file
median_data = np.median(data,axis=0)

# Append all events in same dataframe
print('Loading events')

all_events = []
for _,val in enumerate(events):
    
    events_pd = pd.read_csv(val,sep = '\t')

    new_events = []
    
    for ev in events_pd.iterrows():
        row = ev[1]   
        if row['trial_type'][0] == 'b': # if both hand/leg then add right and left events with same timings
            new_events.append([row['onset'],row['duration'],'l'+row['trial_type'][1:]])
            new_events.append([row['onset'],row['duration'],'r'+row['trial_type'][1:]])
        else:
            new_events.append([row['onset'],row['duration'],row['trial_type']])
   
    df = pd.DataFrame(new_events, columns=['onset','duration','trial_type'])  #make sure only relevant columns present
    all_events.append(df)


# specifying the timing of fMRI frames
frame_times = TR * (np.arange(median_data.shape[0]))

# Create the design matrix, hrf model containing Glover model 
design_matrix = make_first_level_design_matrix(frame_times,
                                               events=all_events[0],
                                               hrf_model='glover'
                                               )

# Setup and fit GLM, estimates contains the parameter estimates
labels, estimates = run_glm(median_data, design_matrix.values)


# Compute z-score of contrasts

print('Computing contrasts')

all_contrasts = {'upper_limb':['lhand_fing1','lhand_fing2','lhand_fing3','lhand_fing4','lhand_fing5',
             'rhand_fing1','rhand_fing2','rhand_fing3','rhand_fing4','rhand_fing5'],
             'lower_limb':['lleg','rleg'],
             'face':['eyes','eyebrows','tongue','mouth']}

for index, (contrast_id, contrast_val) in enumerate(all_contrasts.items()):
    contrast = np.zeros(len(design_matrix.columns)) # array of zeros with len = num predictors
    for i in range(len(contrast)):
        if design_matrix.columns[i] in contrast_val:
            contrast[i] = 1
    
    print('contrast %s is %s' %(contrast_id,contrast))
    # compute contrast-related statistics
    contrast_val = compute_contrast(labels, estimates, contrast, contrast_type='t') 

    z_map = contrast_val.z_score()
    z_map = np.array(z_map)

    zscore_file = os.path.join(soma_out,'z_%s_contrast.npy' %contrast_id)
    np.save(zscore_file,z_map)


# compare each finger with the others of same hand
bhand_label = ['lhand','rhand']
for j,lbl in enumerate(bhand_label):
    
    hand_label = [s for s in all_contrasts['upper_limb'] if lbl in s]
    
    for index, label in enumerate(hand_label):

        contrast = np.zeros(len(design_matrix.columns)) # array of zeros with len = num predictors
    
        for i in range(len(contrast)):
            if design_matrix.columns[i]==label: 
                contrast[i] = 1
            elif lbl in design_matrix.columns[i]: # -1 to other fingers of same hand
                contrast[i] = -1/4.0
        
        print('contrast %s %s is %s' %(label,lbl,contrast))       
        # compute contrast-related statistics
        contrast_val = compute_contrast(labels, estimates, contrast, contrast_type='t') 

        z_map = contrast_val.z_score()
        z_map = np.array(z_map)

        zscore_file = os.path.join(soma_out,'z_%s-all_%s_contrast.npy' %(label,lbl))
        np.save(zscore_file,z_map)
    

#compare left vs right
rl_limb = ['hand','leg']

for j in range(len(rl_limb)):
    contrast = np.zeros(len(design_matrix.columns)) # array of zeros with len = num predictors
    for i in range(len(contrast)):
        if 'r'+rl_limb[j] in design_matrix.columns[i]:
            contrast[i] = 1
        elif 'l'+rl_limb[j] in design_matrix.columns[i]:
            contrast[i] = -1
           
    
    print('contrast %s is %s' %('z_right-left_'+rl_limb[j],contrast))
    # compute contrast-related statistics
    contrast_val = compute_contrast(labels, estimates, contrast, contrast_type='t') 

    z_map = contrast_val.z_score()
    z_map = np.array(z_map)

    zscore_file = os.path.join(soma_out,'z_right-left_'+rl_limb[j]+'_contrast.npy')
    np.save(zscore_file,z_map)

print('Success!')










