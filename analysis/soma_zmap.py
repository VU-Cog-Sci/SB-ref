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

import nibabel as nb
from nilearn import surface

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
    

# use smoothed data?        
with_smooth = analysis_params['with_smooth']

if sj == 'median':
    allsubdir = glob.glob(os.path.join(analysis_params['post_fmriprep_outdir'],'soma','sub-*/'))
    alleventdir = glob.glob(os.path.join(analysis_params['sourcedata_dir'],'sub-*/'))
    
else: # if individual subject
    allsubdir = glob.glob(os.path.join(analysis_params['post_fmriprep_outdir'],'soma','sub-{sj}'.format(sj=sj)))
    alleventdir = glob.glob(os.path.join(analysis_params['sourcedata_dir'],'sub-{sj}'.format(sj=sj)))
        
allsubdir.sort()
alleventdir.sort()

onsets_allsubs = []
durations_allsubs = []
    
for idx,subdir in enumerate(allsubdir): #loop over all subjects in defined list
    print('functional files from %s'%allsubdir[idx])
    print('event files from %s'%alleventdir[idx])

    # define paths and list of files
    filepath = glob.glob(os.path.join(subdir,'*'))
    eventpath = glob.glob(os.path.join(alleventdir[idx],'*','func/*'))

    # changes depending on data used
    if with_smooth=='True':
        # soma out path
        soma_out = os.path.join(analysis_params['soma_outdir'],'sub-{sj}'.format(sj=sj),'run-median','smooth%d'%analysis_params['smooth_fwhm'])
        # last part of filename to use
        file_extension = 'sg_psc_smooth%d.func.gii'%analysis_params['smooth_fwhm']
    else:
        # soma out path
        soma_out = os.path.join(analysis_params['soma_outdir'],'sub-{sj}'.format(sj=sj),'run-median')
        # last part of filename to use
        file_extension = 'sg_psc.func.gii'

    # list of functional files
    filename = [run for run in filepath if 'soma' in run and 'fsaverage' in run and run.endswith(file_extension)]
    filename.sort()

    if not os.path.exists(soma_out): # check if path to save median run exist
        os.makedirs(soma_out) 
        
    # list of stimulus onsets
    events = [run for run in eventpath if 'soma' in run and run.endswith('events.tsv')]
    events.sort()

    TR = analysis_params["TR"]
    
    # load and stack median run for subject
    data_both=[]
    for hemi_label in ['hemi-L','hemi-R']:

        filestring = os.path.join(subdir,'{sj}_ses-*_task-soma_run-median_space-fsaverage_{hemi}_{ext}'.format(sj=os.path.split(os.path.split(subdir)[0])[1],
                                                                                            hemi=hemi_label,
                                                                                            ext=file_extension))
        absfile = glob.glob(filestring) #absolute filename for median run

        if not absfile: #if list is empty (no median run)
            print('%s doesn\'t exist' %(filestring))
            # list with absolute files to make median over

            run_files = [os.path.join(subdir,file) for _,file in enumerate(os.listdir(subdir)) 
                        if 'sub-{sj}'.format(sj=str(sj).zfill(2)) in file and
                        '_{hemi}'.format(hemi=hemi_label) in file and 
                         '_{ext}'.format(ext=file_extension) in file]
            run_files.sort()

            #compute and save median run 
            file_hemi = median_gii(run_files,subdir) 
            print('averaged %d runs, computed %s' %(len(run_files),file_hemi))

            # load surface data from path and append both hemi in array
            data_both.append(surface.load_surf_data(file_hemi).T)
            print('loading %s' %file_hemi)
        else:
            # load surface data from path and append both hemi in array
            data_both.append(surface.load_surf_data(absfile[0]).T)
            print('loading %s' %absfile[0])

    # stack them to get 2D array
    median_data = np.hstack(data_both)
    
    if idx == 0:
        median_sub = median_data[np.newaxis,:,:]
    else:
        median_sub = np.vstack((median_sub,median_data[np.newaxis,:,:]))
        
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

    # make median event dataframe
    onsets = []
    durations = []
    for w in range(len(all_events)):
        onsets.append(all_events[w]['onset'])
        durations.append(all_events[w]['duration'])

    onsets_allsubs.append(np.median(np.array(onsets),axis=0)) #append average onset of all runs
    durations_allsubs.append(np.median(np.array(durations),axis=0))


# all subjects in one array, use this to compute contrasts
median_data_all = np.median(median_sub,axis=0)
events_avg = pd.DataFrame({'onset':np.median(np.array(onsets_allsubs),axis=0),'duration':np.median(np.array(durations_allsubs),axis=0),'trial_type':all_events[0]['trial_type']})

# specifying the timing of fMRI frames
frame_times = TR * (np.arange(median_data_all.shape[0]))

# Create the design matrix, hrf model containing Glover model 
design_matrix = make_first_level_design_matrix(frame_times,
                                               events=events_avg,
                                               hrf_model='glover'
                                               )

# Setup and fit GLM, estimates contains the parameter estimates
labels, estimates = run_glm(median_data_all, design_matrix.values)

print('Computing simple contrasts')

zmaps_all = {} # save all computed z_maps, don't need to load again

reg_keys = list(analysis_params['all_contrasts'].keys()); reg_keys.sort() # list of key names (of different body regions)
loo_keys = leave_one_out_lists(reg_keys) # loo for keys 

for index,region in enumerate(reg_keys):
      
    print('contrast for %s ' %region)
    # list of other contrasts
    other_contr = np.append(analysis_params['all_contrasts'][loo_keys[index][0]],analysis_params['all_contrasts'][loo_keys[index][1]])
    
    contrast = make_contrast(design_matrix.columns,[analysis_params['all_contrasts'][str(region)],other_contr],[1,-len(analysis_params['all_contrasts'][str(region)])/len(other_contr)],num_cond=2)
    
    # compute contrast-related statistics
    contrast_val = compute_contrast(labels, estimates, contrast, contrast_type='t') 

    z_map = contrast_val.z_score()
    z_map = np.array(z_map)
    zmaps_all[str(region)]=z_map
    
    zscore_file = os.path.join(soma_out,'z_%s_contrast.npy' %(region))
    np.save(zscore_file,z_map)

z_threshold = analysis_params['z_threshold']

print('Using z-score of %0.2f as threshold for localizer' %z_threshold)

# compute masked data for R+L hands and for face
# to be used in more detailed contrasts
data_upmask = mask_data(median_data,zmaps_all['upper_limb'],threshold=z_threshold,side='above')
data_facemask = mask_data(median_data,zmaps_all['face'],threshold=z_threshold,side='above')
data_lowmask = mask_data(median_data,zmaps_all['lower_limb'],threshold=z_threshold,side='above')

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
    zmaps_all['RL_'+key[0]]=z_map

    zscore_file = os.path.join(soma_out,'z_right-left_'+key[0]+'_contrast_thresh-%0.2f.npy'%z_threshold)
    np.save(zscore_file,z_map)  

# compare each finger with the others of same hand
print('Contrast one finger vs all others of same hand')

bhand_label = ['lhand','rhand']

for j,lbl in enumerate(bhand_label):
    
    print('For %s' %lbl)

    if lbl == 'lhand': # estou aqui, pensar melhor nisto
        data_RLmask = mask_data(median_data,zmaps_all['RL_hand'],side='below')
    elif lbl == 'rhand':
        data_RLmask = mask_data(median_data,zmaps_all['RL_hand'],side='above')

    labels_RLmask, estimates_RLmask = run_glm(data_RLmask, design_matrix.values)
    
    hand_label = [s for s in analysis_params['all_contrasts']['upper_limb'] if lbl in s] #list of all fingers in one hand  
    otherfings = leave_one_out_lists(hand_label) # list of lists with other fingers to contrast 
    
    for i,fing in enumerate(hand_label):
        contrast = make_contrast(design_matrix.columns,[[fing],otherfings[i]],[1,-1/4.0],num_cond=2)
        # compute contrast-related statistics
        contrast_val = compute_contrast(labels_RLmask, estimates_RLmask, contrast, contrast_type='t') 

        z_map = contrast_val.z_score()
        z_map = np.array(z_map)

        zscore_file = os.path.join(soma_out,'z_%s-all_%s_contrast_thresh-%0.2f.npy' %(fing,lbl,z_threshold))
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

    zscore_file = os.path.join(soma_out,'z_%s-other_face_areas_contrast_thresh-%0.2f.npy' %(part,z_threshold))
    np.save(zscore_file,z_map)        

print('Success!')





