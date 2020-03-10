# make new script to make contrasts that will be saved in
# soma_fit/new_fits/...
# script not using nilearn functions but glm defined in utils
# in this way I can also save other stats and model predictions per voxel


import os, json
import sys, glob
import re 

import numpy as np
import pandas as pd

import nibabel as nb
from nilearn import surface

from nistats.design_matrix import make_first_level_design_matrix
from nistats.contrasts import compute_contrast

from utils import * #import script to use relevante functions

from joblib import Parallel, delayed
from nistats.reporting import plot_design_matrix

import shutil


# define participant number and open json parameter file
if len(sys.argv)<2:	
    raise NameError('Please add subject number (ex:01) '	
                    'as 1st argument in the command line!')	

else:
    sj = str(sys.argv[1]).zfill(2) #fill subject number with 0 in case user forgets	

    with open('analysis_params.json','r') as json_file:	
            analysis_params = json.load(json_file)	
 

# define paths and variables
rsq_threshold = 0 #0.5 
z_threshold = analysis_params['z_threshold']

TR = analysis_params["TR"]

# last part of filename to use
file_extension = 'sg_psc.func.gii'

# make list with subject number (or all subject number if we want median contrast)
if sj == 'median':
    all_subs = ['01','02','03','04','05','08','09','11','12','13']
    all_contrasts = {}
else:
    all_subs = [sj]


# fit GLM 
for _,sub in enumerate (all_subs): # for sub or all subs

    # soma out path (where to save contrasts)
    out_dir = os.path.join(analysis_params['soma_outdir'],'new_fits','sub-{sj}'.format(sj=sub))

    # path to save fits, for testing
    if not os.path.exists(out_dir):  # check if path exists
        os.makedirs(out_dir)

    print('files will be saved in %s'%out_dir)
            
    # path to events
    eventdir = os.path.join(analysis_params['sourcedata_dir'],'sub-{sj}'.format(sj=str(sub).zfill(2)),'ses-01','func')
    print('event files from %s' % eventdir)
    
    # make average event file
    events_avg = make_median_soma_events(sub,eventdir)

    ## average data file
    # path to functional files
    filepath = os.path.join(analysis_params['post_fmriprep_outdir'], 'soma', 'sub-{sj}'.format(sj=str(sub).zfill(2)))
    
    # load data array for sub    
    data = make_median_soma_sub(sub,filepath,out_dir,file_extension=file_extension)
    
    # specifying the timing of fMRI frames
    frame_times = TR * (np.arange(data.shape[-1]))

    # Create the design matrix, hrf model containing Glover model 
    design_matrix = make_first_level_design_matrix(frame_times,
                                                   events=events_avg,
                                                   hrf_model='glover'
                                                   )
    
    # plot design matrix and save just to check if everything fine
    plot = plot_design_matrix(design_matrix)
    fig = plot.get_figure()
    fig.savefig(os.path.join(out_dir,'design_matrix.svg'), dpi=100,bbox_inches = 'tight')
    
    # set estimates filename
    estimates_filename = os.path.join(out_dir,'sub-{sj}_ses-01_task-soma_run-median_space-fsaverage_hemi-both_{ext}'.format(sj=sub,ext=file_extension))
    estimates_filename = estimates_filename.replace('.func.gii','_estimates.npz')

    if not os.path.isfile(estimates_filename): # if doesn't exist already
        print('fitting GLM to %d vertices'%data.shape[0])
        soma_params = Parallel(n_jobs=16)(delayed(fit_glm)(vert, design_matrix.values) for _,vert in enumerate(data))
        soma_params = np.vstack(soma_params)
        
        np.savez(estimates_filename,
                  model = soma_params[..., 0],
                  betas = soma_params[..., 1],
                  r2 = soma_params[..., 2],
                  mse = soma_params[...,3])
        
    # load betas
    soma_estimates = np.load(estimates_filename,allow_pickle=True)
    betas = soma_estimates['betas']

    # mask the data
    # only use data where rsq of fit higher than X% 
    print('masking data with %.2f rsq threshold'%rsq_threshold)
    data_masked = data.copy()
    alpha_mask = np.array([True if val<= rsq_threshold or np.isnan(val) else False for _,val in enumerate(soma_estimates['r2'])])
    data_masked[alpha_mask]=np.nan
    
    # now make simple contrasts
    print('Computing simple contrasts')

    zmaps_all = {} # save all computed z_maps, don't need to load again

    reg_keys = list(analysis_params['all_contrasts'].keys()); reg_keys.sort() # list of key names (of different body regions)
    loo_keys = leave_one_out_lists(reg_keys) # loo for keys 

    for index,region in enumerate(reg_keys): # one broader region vs all the others

        print('contrast for %s ' %region)
        # list of other contrasts
        other_contr = np.append(analysis_params['all_contrasts'][loo_keys[index][0]],analysis_params['all_contrasts'][loo_keys[index][1]])

        contrast = make_contrast(design_matrix.columns,[analysis_params['all_contrasts'][str(region)],other_contr],[1,-len(analysis_params['all_contrasts'][str(region)])/len(other_contr)],num_cond=2)

        # save estimates in dir 
        stats_filename = os.path.join(out_dir,'z_%s_contrast_stats_rsq-%.2f.npz' %(region,rsq_threshold))
        zscore_file = os.path.join(out_dir,'z_%s_contrast_rsq-%.2f.npy' %(region,rsq_threshold))
        
        if not os.path.isfile(stats_filename): # if doesn't exist already
            # compute contrast-related statistics
            soma_stats = Parallel(n_jobs=16)(delayed(compute_stats)(vert, design_matrix.values,contrast,betas[w]) for w,vert in enumerate(data_masked))
            soma_stats = np.vstack(soma_stats) # t_val,p_val,zscore
            
            np.savez(stats_filename,
                      t_val = soma_stats[..., 0],
                      p_val = soma_stats[..., 1],
                      zscore = soma_stats[..., 2])

            zmaps_all[str(region)]=soma_stats[..., 2]
            np.save(zscore_file,soma_stats[..., 2])

        else:
            zmaps_all[str(region)] = np.load(zscore_file,allow_pickle=True)

    # now do rest of the contrasts within region
    # do contrasts only in regions that are relevant (finger contrast only within hand region)

    print('Using z-score of %0.2f as threshold for localizer' %z_threshold)

    # compute masked data for R+L hands and for face
    # to be used in more detailed contrasts
    data_upmask = mask_data(data_masked.T,zmaps_all['upper_limb'],threshold=z_threshold,side='above')
    data_facemask = mask_data(data_masked.T,zmaps_all['face'],threshold=z_threshold,side='above')
    data_lowmask = mask_data(data_masked.T,zmaps_all['lower_limb'],threshold=z_threshold,side='above')
    
    # compare left and right
    print('Right vs Left contrasts')

    limbs = [['hand',analysis_params['all_contrasts']['upper_limb']],['leg',analysis_params['all_contrasts']['lower_limb']]]

    for _,key in enumerate(limbs):
        print('For %s' %key[0])

        rtask = [s for s in key[1] if 'r'+key[0] in s]
        ltask = [s for s in key[1] if 'l'+key[0] in s]
        tasks = [rtask,ltask] # list with right and left elements

        contrast = make_contrast(design_matrix.columns,tasks,[1,-1],num_cond=2)

        # save estimates in dir 
        stats_filename = os.path.join(out_dir,'z_right-left_'+key[0]+'_contrast_thresh-%0.2f_stats_rsq-%.2f.npz' %(z_threshold,rsq_threshold))
        zscore_file = os.path.join(out_dir,'z_right-left_'+key[0]+'_contrast_thresh-%0.2f_rsq-%.2f.npy'%(z_threshold,rsq_threshold))

        if not os.path.isfile(stats_filename): # if doesn't exist already

            # compute contrast-related statistics
            data4stat = data_lowmask.T  if key[0]=='leg' else data_upmask.T

            soma_stats = Parallel(n_jobs=16)(delayed(compute_stats)(vert, design_matrix.values,contrast,betas[w]) for w,vert in enumerate(data4stat))
            soma_stats = np.vstack(soma_stats) # t_val,p_val,zscore
            
            np.savez(stats_filename,
                      t_val = soma_stats[..., 0],
                      p_val = soma_stats[..., 1],
                      zscore = soma_stats[..., 2])

            zmaps_all['RL_'+key[0]]=soma_stats[..., 2]
            
            np.save(zscore_file,soma_stats[..., 2])
        else:
            zmaps_all['RL_'+key[0]] = np.load(zscore_file,allow_pickle=True)  


    # compare each finger with the others of same hand
    print('Contrast one finger vs all others of same hand')

    bhand_label = ['lhand','rhand']

    for j,lbl in enumerate(bhand_label):

        print('For %s' %lbl)

        if lbl == 'lhand': # estou aqui, pensar melhor nisto
            data_RLmask = mask_data(data_upmask,zmaps_all['RL_hand'],side='below')
        elif lbl == 'rhand':
            data_RLmask = mask_data(data_upmask,zmaps_all['RL_hand'],side='above')

        hand_label = [s for s in analysis_params['all_contrasts']['upper_limb'] if lbl in s] #list of all fingers in one hand  
        otherfings = leave_one_out_lists(hand_label) # list of lists with other fingers to contrast 

        for i,fing in enumerate(hand_label):

            contrast = make_contrast(design_matrix.columns,[[fing],otherfings[i]],[1,-1/4.0],num_cond=2)

            # save estimates in dir 
            stats_filename = os.path.join(out_dir,'z_%s-all_%s_contrast_thresh-%0.2f_stats_rsq-%.2f.npz' %(fing,lbl,z_threshold,rsq_threshold))
            zscore_file = os.path.join(out_dir,'z_%s-all_%s_contrast_thresh-%0.2f_rsq-%.2f.npy'%(fing,lbl,z_threshold,rsq_threshold))

            if not os.path.isfile(stats_filename): # if doesn't exist already

                soma_stats = Parallel(n_jobs=16)(delayed(compute_stats)(vert, design_matrix.values,contrast,betas[w]) for w,vert in enumerate(data_RLmask.T))
                soma_stats = np.vstack(soma_stats) # t_val,p_val,zscore
                
                np.savez(stats_filename,
                          t_val = soma_stats[..., 0],
                          p_val = soma_stats[..., 1],
                          zscore = soma_stats[..., 2])

                np.save(zscore_file,soma_stats[..., 2])  

    # compare each face region with the others
    print('Contrast one face part vs all others within face')

    face = analysis_params['all_contrasts']['face']
    otherface = leave_one_out_lists(face) # list of lists with other fingers to contrast 

    for i,part in enumerate(face):
        contrast = make_contrast(design_matrix.columns,[[part],otherface[i]],[1,-1/3.0],num_cond=2)

        # save estimates in dir 
        stats_filename = os.path.join(out_dir,'z_%s-other_face_areas_contrast_thresh-%0.2f_stats_rsq-%.2f.npz' %(part,z_threshold,rsq_threshold))
        zscore_file = os.path.join(out_dir,'z_%s-other_face_areas_contrast_thresh-%0.2f_rsq-%.2f.npy' %(part,z_threshold,rsq_threshold))

        if not os.path.isfile(stats_filename): # if doesn't exist already

            soma_stats = Parallel(n_jobs=16)(delayed(compute_stats)(vert, design_matrix.values,contrast,betas[w]) for w,vert in enumerate(data_facemask.T))
            soma_stats = np.vstack(soma_stats) # t_val,p_val,zscore
            
            np.savez(stats_filename,
                      t_val = soma_stats[..., 0],
                      p_val = soma_stats[..., 1],
                      zscore = soma_stats[..., 2])

            np.save(zscore_file,soma_stats[..., 2])


    ### SMOOOTH THEM #########
    
    # append all contrasts and smooth them
    general_contrasts = [x for x in os.listdir(out_dir) if 'contrast_rsq-%.2f.npy'%rsq_threshold in x]
    fine_contrasts = [x for x in os.listdir(out_dir) if '_contrast_thresh-%.2f_rsq-%.2f.npy'%(z_threshold,rsq_threshold) in x]
    contrast2smooth = general_contrasts + fine_contrasts; contrast2smooth.sort()

    # new output folder for smoothed files
    new_out = os.path.join(out_dir,'smooth%d'%analysis_params['smooth_fwhm'])

    # path to save smooth contrasts, for testing
    if not os.path.exists(new_out):  # check if path exists
        os.makedirs(new_out)
    
    #load contrast files and smooth one at a time
    for _,file in enumerate(contrast2smooth):
        
        zload = np.load(os.path.join(out_dir,file),allow_pickle=True)

        # check if smooth file exists, else make it
        if not os.path.exists(os.path.join(new_out,file)):
            smooth_nparray(os.path.join(analysis_params['post_fmriprep_outdir'], 'soma', 'sub-{sj}'.format(sj=str(sub).zfill(2))),
                           zload,
                           new_out,
                           'zscore',
                           file,
                            n_TR=141,
                           file_extension=file_extension,
                           sub_mesh='fsaverage',
                           smooth_fwhm=analysis_params['smooth_fwhm'])

        else:
            print('smoothed %s already exists'%file)


        # not smoothed, "normal" contrasts
        if sj == 'median': # save all in dict to later make median contrast
            
            if file not in all_contrasts:
                all_contrasts[file] = [zload]
            else:
                all_contrasts[file] = np.vstack((all_contrasts[file], zload))


# calculate median contrasts, save and then smooth
if sj == 'median':
    # soma out path (where to save contrasts)
    out_dir = os.path.join(analysis_params['soma_outdir'],'new_fits','sub-{sj}'.format(sj=sj))

    if not os.path.exists(out_dir):  # check if path exists
        os.makedirs(out_dir)

    print('files will be saved in %s'%out_dir)
    
    for _,file in enumerate(contrast2smooth):
        
        print('averaging and saving %s'%file)
        
        med_contrast = np.nanmedian(all_contrasts[file],axis = 0)
        
        # save average contrast
        np.save(os.path.join(out_dir,file),med_contrast)
        
        # new output folder for smoothed files
        new_out = os.path.join(out_dir,'smooth%d'%analysis_params['smooth_fwhm'])

        # path to save smooth contrasts, for testing
        if not os.path.exists(new_out):  # check if path exists
            os.makedirs(new_out)

            # check if smooth file exists, else make it
        if not os.path.exists(os.path.join(new_out,file)):
            smooth_nparray(os.path.join(analysis_params['post_fmriprep_outdir'], 'soma', 'sub-{sj}'.format(sj=str(all_subs[0]).zfill(2))),
                           med_contrast,
                           new_out,
                           'zscore',
                           file,
                            n_TR=141,
                           file_extension=file_extension,
                           sub_mesh='fsaverage',
                           smooth_fwhm=analysis_params['smooth_fwhm'])

        else:
            print('smoothed %s already exists'%file)
      

print('Success!')










