#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Th Jun 6 11:13:11 2019

@author: inesverissimo

Do SOMA contrasts and save outputs
"""

import os, json
import sys, glob
import re 

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
for string in ['sub-01_ses-01', 'sub-03_ses-01']:
    if re.search(string, filename[0]):
        TR = 1.5
    else:
        TR = analysis_params["TR"]

# soma out path
soma_out = os.path.join(analysis_params['soma_outdir'],'sub-{sj}'.format(sj=sj),'run-median')

if not os.path.exists(soma_out): # check if path to save median run exist
    os.makedirs(soma_out) 

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

print('Computing simple contrasts')

zmaps_all = {} # save all computed z_maps, don't need to load again

reg_keys = list(analysis_params['all_contrasts'].keys()); reg_keys.sort() # list of key names (of different body regions)
loo_keys = leave_one_out_lists(reg_keys) # loo for keys 

for index,region in enumerate(reg_keys):
      
    print('contrast for %s ' %region)
    # list of other contrasts
    other_contr = np.append(analysis_params['all_contrasts'][loo_keys[index][0]],analysis_params['all_contrasts'][loo_keys[index][1]])
    
    contrast = make_contrast(design_matrix.columns,[analysis_params['all_contrasts'][str(region)],other_contr],[1,-1/len(other_contr)],num_cond=2)
    
    # compute contrast-related statistics
    contrast_val = compute_contrast(labels, estimates, contrast, contrast_type='t') 

    z_map = contrast_val.z_score()
    z_map = np.array(z_map)
    zmaps_all[str(region)]=z_map
    
    zscore_file = os.path.join(soma_out,'z_%s_contrast.npy' %region)
    np.save(zscore_file,z_map)

z_threshold = analysis_params['z_threshold']

print('Using z-score of %d as threshold for localizer' %z_threshold)

# compute masked data for R+L hands and for face
# to be used in more detailed contrasts
data_upmask = mask_data(median_data,zmaps_all['upper_limb'],z_threshold)
data_facemask = mask_data(median_data,zmaps_all['face'],z_threshold)
data_lowmask = mask_data(median_data,zmaps_all['lower_limb'],z_threshold)

# Setup and fit GLM for masked data, estimates contains the parameter estimates
labels_upmask, estimates_upmask = run_glm(data_upmask, design_matrix.values)
labels_facemask, estimates_facemask = run_glm(data_facemask, design_matrix.values)
labels_lowmask, estimates_lowmask = run_glm(data_lowmask, design_matrix.values)

print('Right vs Left contrasts')

limbs = [['hand',analysis_params['all_contrasts']['upper_limb']],['leg',analysis_params['all_contrasts']['lower_limb']]]
         
for _,key in enumerate(limbs):
    print('For %s' %key[0])
    
    rtask = [s for s in key[1] if 'r'+key[0] in s]
    ltask = [s for s in key[1] if 'l'+key[0] in s]
    tasks = [rtask,ltask] # list with right and left elements
    
    contrast = make_contrast(design_matrix.columns,tasks,[1,-1],num_cond=2)
    # compute contrast-related statistics
    if key[0]=='leg':
        masklbl = labels_lowmask
        maskestm = estimates_lowmask
    else:
        masklbl = labels_upmask
        maskestm = estimates_upmask
        
    contrast_val = compute_contrast(masklbl, maskestm, contrast, contrast_type='t') 

    z_map = contrast_val.z_score()
    z_map = np.array(z_map)

    zscore_file = os.path.join(soma_out,'z_right-left_'+key[0]+'_contrast.npy')
    np.save(zscore_file,z_map)  

# compare each finger with the others of same hand
print('Contrast one finger vs all others of same hand')

bhand_label = ['lhand','rhand']

for j,lbl in enumerate(bhand_label):
    
    print('For %s' %lbl)
    
    hand_label = [s for s in analysis_params['all_contrasts']['upper_limb'] if lbl in s] #list of all fingers in one hand  
    otherfings = leave_one_out_lists(hand_label) # list of lists with other fingers to contrast 
    
    for i,fing in enumerate(hand_label):
        contrast = make_contrast(design_matrix.columns,[[fing],otherfings[i]],[1,-1/4.0],num_cond=2)
        # compute contrast-related statistics
        contrast_val = compute_contrast(labels_upmask, estimates_upmask, contrast, contrast_type='t') 

        z_map = contrast_val.z_score()
        z_map = np.array(z_map)

        zscore_file = os.path.join(soma_out,'z_%s-all_%s_contrast.npy' %(fing,lbl))
        np.save(zscore_file,z_map)
 
# compare each finger with the others of same hand
print('Contrast one face part vs all others within face')
    
face = analysis_params['all_contrasts']['face']
otherface = leave_one_out_lists(face) # list of lists with other fingers to contrast 

for i,part in enumerate(face):
    contrast = make_contrast(design_matrix.columns,[[part],otherface[i]],[1,-1/3.0],num_cond=2)
    # compute contrast-related statistics
    contrast_val = compute_contrast(labels_facemask, estimates_facemask, contrast, contrast_type='t') 

    z_map = contrast_val.z_score()
    z_map = np.array(z_map)

    zscore_file = os.path.join(soma_out,'z_%s-other_face_areas_contrast.npy' %(part))
    np.save(zscore_file,z_map)        

print('Success!')










