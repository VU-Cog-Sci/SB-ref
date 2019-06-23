#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Th Jun 22 11:13:11 2019

@author: inesverissimo

Plot eyetracking density maps for soma
As check for fixation

needs to be run in "eye" conda env
"""

import os, json
import sys, glob
import re 

import numpy as np
import pandas as pd

import hedfpy

from scipy.stats import gaussian_kde
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# define participant number and open json parameter file
if len(sys.argv)<2:	
    raise NameError('Please add subject number (ex:1) '	
                    'as 1st argument in the command line!')	
   	 
elif len(sys.argv)<3:	
    raise NameError('Please add session number (ex:1) '	
                    'as 2nd argument in the command line!')	

else:	
    sj = int(sys.argv[1])
    ses = int(sys.argv[2])
    print('making plots for sub-%d ses-%d'%(sj,ses))


with open('analysis_params.json','r') as json_file:	
        analysis_params = json.load(json_file)	


def convert_session(sj,ses,indir,outdir,pupil_hp,pupil_lp):
    if type(sj) == int:
        sj, ses = str(sj).zfill(2), str(ses).zfill(2)
    globstr = os.path.join(indir, 'sub-{sj}/ses-{ses}/eyetrack/sub-{sj}_ses-{ses}_task-*_run-*_eyetrack.edf'.format(sj=sj, ses=ses))
    edf_files = glob.glob(globstr);edf_files.sort() #list of absolute paths to all edf files in that session for that subject
    
    #single hdf5 file that contains all eye data for the runs of that session
    hdf_file = os.path.join(outdir, 'sub-{sj}/ses-{ses}/eyetrack.h5'.format(sj=sj, ses=ses))
    
    if not os.path.exists(os.path.split(hdf_file)[0]): # check if path to save hdf5 files exists
        os.makedirs(os.path.split(hdf_file)[0])      # if not create it
        
        ho = hedfpy.HDFEyeOperator(hdf_file)
        for ef in edf_files:
            alias = os.path.splitext(os.path.split(ef)[1])[0] #name of data for that run
            ho.add_edf_file(ef)
            ho.edf_message_data_to_hdf(alias = alias) #write messages ex_events to hdf5
            ho.edf_gaze_data_to_hdf(alias = alias, pupil_hp = pupil_hp, pupil_lp = pupil_lp) #add raw and preprocessed data to hdf5   
            
    else:
        print('%s already exists, skipping' %hdf_file)
        

# for linux computer
screen = [1920, 1080]
task='soma'

# alter screen res for laptop subjects
laptop_subs = [(1,1),(3,1),(4,1),(5,1),(5,2),(6,1),(7,1),(8,1),(9,1)]
for _,data in enumerate(laptop_subs):
    if (sj,ses)==data:
        screen = [1680,1050]
        
print('screen resolution for sub-0%d and ses-0%d is %s'%(sj,ses,screen))


# convert for both sessions of this subject
for session in [1,2]:  
    if sj==9 and ses==2:
        print('issue with sub-%s ses-%s eyetracking files, not saved'%(str(sj).zfill(2),str(session).zfill(2)))
    else:
        print('converting eyetracking edf to hdf5 file for sub-%s ses-%s '%(str(sj).zfill(2),str(session).zfill(2)))
        convert_session(sj, session,analysis_params['sourcedata_dir'],analysis_params['eyetrack_dir'],
                        analysis_params['high_pass_pupil_f'],analysis_params['low_pass_pupil_f'])




# plot density over image for all runs

for run in [1,2,3,4]:
    if sj == 5 and run == 3:
        print('no run %d'%run)
    else:
        # define hdf operator for specific run
        hdf_file = os.path.join(analysis_params['eyetrack_dir'], 
                        'sub-{sj}/ses-{ses}/eyetrack.h5'.format(sj=str(sj).zfill(2), ses=str(ses).zfill(2)))
        ho = hedfpy.HDFEyeOperator(hdf_file)

        alias = 'sub-{sj}_ses-{ses}_task-{task}_run-{run}_bold_eyetrack'.format(sj=str(sj).zfill(2), 
                ses=str(ses).zfill(2), task=task, run=str(run).zfill(2))
        
        # load table with timestamps for run
        with pd.HDFStore(ho.input_object) as h5_file:
            table_trl = h5_file['%s/trials'%alias] # load table with timestamps for run
            
        # compute array with all timestamps of start and end of trial, for whole run
        parameters = ho.read_session_data(alias, 'parameters') 
        trl_str_end = [] 

        for trial_nr in range(len(parameters)):
            trl_time = np.array(table_trl[table_trl['trial_start_index'] == trial_nr][['trial_start_EL_timestamp', 'trial_end_EL_timestamp']])
            trl_time = trl_time[0] #[trial start sample, trial end sample]

            trl_str_end.append(trl_time)

        # load table with positions for run (from start of 1st trial to end)
        with pd.HDFStore(ho.input_object) as h5_file:
            period_block_nr = ho.sample_in_block(sample = trl_str_end[0][0], block_table = h5_file['%s/blocks'%alias]) 
            table_pos = h5_file['%s/block_%i'%(alias, period_block_nr)]
            
        # compute array with all gaze x,y positions of each trial, for whole run
        gaze_alltrl = []
        # gaze can be from right or left eye, so set string first
        xgaz_srt = table_pos.columns[1]
        ygaz_str = table_pos.columns[2]

        for trial_nr in range(len(parameters)):
            x_pos = np.array(table_pos[np.logical_and(trl_str_end[trial_nr][0]<=table_pos['time'],table_pos['time']<=trl_str_end[trial_nr][1])][xgaz_srt])
            y_pos = np.array(table_pos[np.logical_and(trl_str_end[trial_nr][0]<=table_pos['time'],table_pos['time']<=trl_str_end[trial_nr][1])][ygaz_str])

            gaze_trial = np.array([x_pos.squeeze(),y_pos.squeeze()], dtype=object)

            gaze_alltrl.append(gaze_trial)

        # pixel coordinates, according to screen res
        x_pixels, y_pixels = np.mgrid[0:screen[0]:10, 0:screen[1]:10]
        pixel_coordinates = np.vstack([x_pixels.ravel(), y_pixels.ravel()])
        
        for trial in range(len(parameters)):
            # check name of stim for trial
            stim_name = os.path.splitext(analysis_params['soma_stimulus'][trial])[0]
            # pick corresponding background
            if stim_name in analysis_params['all_contrasts']['face']:
                backg_img = os.path.join(analysis_params['somaplot_dir'],'face.png')
            elif stim_name in analysis_params['all_contrasts']['upper_limb'] or 'bhand' in stim_name:
                backg_img = os.path.join(analysis_params['somaplot_dir'],'upper_limb.png')
            else:
                backg_img = os.path.join(analysis_params['somaplot_dir'],'lower_limb.png')
            print('trial stim is '+stim_name+', will be ploted over '+backg_img)

            # need to change type of numpy array back to float, to calculate density
            gaze = gaze_alltrl[trial].astype('float64') 
            gaze_kde = gaussian_kde(gaze)
            density = np.reshape(gaze_kde(pixel_coordinates).T, x_pixels.shape)

            fig, ax = plt.subplots()
            image = Image.open(backg_img)
            image.thumbnail((screen[0],screen[1]), Image.ANTIALIAS) 
            img = image
            ax.imshow(img, extent=[0, screen[0], 0, screen[1]])
            ax.imshow(density, cmap='inferno',
                       extent=[0, screen[0], 0, screen[1]],alpha=0.5)
            ax.set_xlim([0, screen[0]])
            ax.set_ylim([0, screen[1]])

            filepath = os.path.join(analysis_params['somaplot_dir'],'sub-%s'%str(sj).zfill(2))

            if not os.path.exists(filepath): # check if path to save median run exist
                os.makedirs(filepath) 

            filename = os.path.join(filepath,'trial-%s_run-%s_soma_%s.png'%(str(trial).zfill(2),str(run).zfill(2),stim_name))

            plt.savefig(filename)
