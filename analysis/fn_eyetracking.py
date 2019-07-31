
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Th Jul 19 11:13:11 2019
@author: inesverissimo

Get gaze data, make some plots
Compute arrays with saccade vector info
during movie watching.

needs to be run in "eye" conda env
"""

import os, json
import sys, glob
import re 

import numpy as np
import pandas as pd

import hedfpy
import matplotlib.pyplot as plt

# define participant number and open json parameter file
if len(sys.argv)<2: 
    raise NameError('Please add subject number (ex:1) ' 
                    'as 1st argument in the command line!') 

else:   
    sj = int(sys.argv[1])
    ses = 1 # it's always in the first session #int(sys.argv[2])
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
task='fn'

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



outdir_plots = os.path.join(analysis_params['eyetrack_dir'],'fn','sub-{sj}'.format(sj=str(sj).zfill(2)))

if not os.path.exists(outdir_plots): # check if path to save plots exists
    os.makedirs(outdir_plots)      # if not create it


# do loop for all runs
num_runs = np.arange(1,11)

for run in num_runs:
    try:
        # save dict with timings for all runs and gazedata
        timings_allruns = {'trl_str_end':[],'trial_phase_info':[],'run_num':[],'gaze_all_trl':[]}
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

        # FN only has 1 trial per run
        trl_time = np.array(table_trl[table_trl['trial_start_index'] == 0][['trial_start_EL_timestamp', 'trial_end_EL_timestamp']])
        trl_str_end = trl_time[0] #[trial start sample, trial end sample]

        # load table with positions for run (from start of 1st trial to end)
        with pd.HDFStore(ho.input_object) as h5_file:
            period_block_nr = ho.sample_in_block(sample = trl_str_end[0], block_table = h5_file['%s/blocks'%alias]) 
            table_pos = h5_file['%s/block_%i'%(alias, period_block_nr)]

        # compute array with all gaze x,y positions of each trial, for whole run
        gaze_alltrl = []
        # gaze can be from right or left eye, so set string first
        xgaz_srt = table_pos.columns[10]#[1] #now using gaze_x/y_int - I think it's the interpolated, but check
        ygaz_str = table_pos.columns[11]#[2]

        x_pos = np.array(table_pos[np.logical_and(trl_str_end[0]<=table_pos['time'],table_pos['time']<=trl_str_end[1])][xgaz_srt])
        y_pos = np.array(table_pos[np.logical_and(trl_str_end[0]<=table_pos['time'],table_pos['time']<=trl_str_end[1])][ygaz_str])

        gaze_alltrl = np.array([x_pos.squeeze(),y_pos.squeeze()], dtype=object)

        # get trial phase info
        trial_phase_info = ho.read_session_data(alias, 'trial_phases')
        #save info in dict
        timings_allruns['trl_str_end'].append(trl_str_end)
        timings_allruns['trial_phase_info'].append([trial_phase_info['trial_phase_EL_timestamp'][0],trial_phase_info['trial_phase_EL_timestamp'][1],trial_phase_info['trial_phase_EL_timestamp'][2]])
        timings_allruns['run_num'].append([str(run).zfill(2)])
        timings_allruns['gaze_all_trl'].append(gaze_alltrl)

        # save numpy array with timing info in sub-dir
        np.savez_compressed(os.path.join(outdir_plots,'gaze_timings_run-%s.npz'%str(run).zfill(2)),
            trl_str_end=timings_allruns['trl_str_end'],trial_phase_info=timings_allruns['trial_phase_info'],run_num=timings_allruns['run_num'],
            gazedata=timings_allruns['gaze_all_trl'])

        # plot gaze!
        plt.figure(num=None, figsize=(10, 6), dpi=100, facecolor='w', edgecolor='k')

        plt.plot(gaze_alltrl[0][:],c='k')
        plt.plot(gaze_alltrl[1][:],c='orange')
        # add lines for beggining and end of clips
        # subtract start time to make it valid
        #plt.axvline(x=trl_str_end[0]-trl_str_end[0],c='b') #start recording
        plt.axvline(x=trial_phase_info['trial_phase_EL_timestamp'][0]-trl_str_end[0],c='b',linestyle='--') #1st TR? fixation period
        plt.axvline(x=trial_phase_info['trial_phase_EL_timestamp'][1]-trl_str_end[0],c='b') #start movie clip
        plt.axvline(x=trial_phase_info['trial_phase_EL_timestamp'][2]-trl_str_end[0],c='r') #end movie clip

        plt.xlabel('Samples',fontsize=18)
        plt.ylabel('Position',fontsize=18)
        plt.legend(['xgaze','ygaze'], fontsize=10)
        plt.title('Gaze run-%s' %str(run).zfill(2), fontsize=20)

        plt.savefig(os.path.join(outdir_plots,'gaze_xydata_run-%s.png' %str(run).zfill(2)))
        plt.close()


        # get saccade info for that period (during movie clip) 
        movie_period = np.array([timings_allruns['trial_phase_info'][0][1],timings_allruns['trial_phase_info'][0][2]])
        try: 
            print('left eye was recorded')
            saccade_info = ho.saccades_during_period(movie_period,alias,requested_eye='L')
        except:
            print('right eye was recorded')
            saccade_info = ho.saccades_during_period(movie_period,alias,requested_eye='R')
        saccade_info = saccade_info

        trial_dur = int(trl_str_end[1]-trl_str_end[0])
        trial_arr = np.zeros((5,trial_dur)) # array of (5 x trial length), filled with sacc amplitude, x position and y position of vector and time of saccade

        #
        # save value and timing of interpolated samples, between the beginning and end of the movie
        interp_mov_time = np.array(table_pos[np.logical_and(movie_period[0]<=table_pos['time'],table_pos['time']<=movie_period[1])]['time'])
        interp_mov_val = np.array(table_pos[np.logical_and(movie_period[0]<=table_pos['time'],table_pos['time']<=movie_period[1])][table_pos.columns[6]]) # interpolated time points|

        interp_time = []
        for indice, interp in enumerate(interp_mov_time):
            if interp_mov_val[indice] != 0.0:
                interp_time.append(int(interp-trl_str_end[0])) # store timepoint - time of begining of trial
            else:
                interp_time.append(np.nan) #fill rest with nans

        # detect saccades during relevant interval (and exclude those in interpolated timepoints, to be more conservative)
        sac = 0 # initiate saccade counter
        init_mov = int(movie_period[0]-trl_str_end[0]) # initiate counter for interpolated values (only within movie)
        end_mov = int(movie_period[1]-trl_str_end[0])

        for i in range(trial_dur):
            if sac < len(saccade_info): #set saccade range to check if interpolated samples within it
                sacc_interval = np.arange(saccade_info[sac]['expanded_start_time'],saccade_info[sac]['expanded_end_time']+1)

            elif sac==len(saccade_info): # when all saccade info saved, break loop
                print('total of %d saccade info saved, no more saccades in run %s' %(sac,str(run).zfill(2)))
                break

            if i == (saccade_info[sac]['expanded_end_time']+1): #after saccade end time, increment counter to next saccade
                sac += 1
            elif i >= saccade_info[sac]['expanded_start_time'] and (init_mov <= i <= end_mov) and interp_time[i-init_mov] not in sacc_interval: 
                trial_arr[0][i]=np.sqrt(saccade_info[sac]['expanded_vector'][0]**2+saccade_info[sac]['expanded_vector'][1]**2) # amplitude (distance) of saccade
                trial_arr[1][i]=saccade_info[sac]['expanded_vector'][0] # x position relative to center (0,0)
                trial_arr[2][i]=saccade_info[sac]['expanded_vector'][1] # y position relative to center (0,0)
                trial_arr[3][i]=saccade_info[sac]['expanded_start_time']# start time relative to begining of trial 
                trial_arr[4][i]=saccade_info[sac]['expanded_end_time'] # end time relative to begining of trial


        # save numpy array with saccade vector info in sub-dir
        np.savez(os.path.join(outdir_plots,'sacc4dm_run-%s.npz'%str(run).zfill(2)),
             amplitude=trial_arr[0],xpos=trial_arr[1],ypos=trial_arr[2],startime=trial_arr[3],endtime=trial_arr[4])

        # just to see where saccade vector endpoint is on screen, as a sanity check
        # not really necessary
        save_plots = os.path.join(outdir_plots,'run-%s' %str(run).zfill(2))

        if not os.path.exists(save_plots): # check if path to save plots exists
            os.makedirs(save_plots)      # if not create it

        sac_counter = 0 #do counter because now the number of saccades will be less than detected by hedfpy
        for j in range(len(saccade_info)):

            smp_idx = saccade_info[j]['expanded_start_time']#13014 # index with sample number 

            if trial_arr[1][smp_idx] != 0:
                x_centered = trial_arr[1][smp_idx] + screen[0]/2.0
                y_centered = trial_arr[2][smp_idx] + screen[1]/2.0
                amp_pix = trial_arr[0][smp_idx]

                sac_endpoint = plt.Circle((x_centered, y_centered), radius=amp_pix/2.0, color='r') #circle diameter = amplitude of saccade
                fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
                ax.set_xlim((0, screen[0]))
                ax.set_ylim((0, screen[1]))
                ax.add_artist(sac_endpoint)

                plt.savefig(os.path.join(save_plots,'expanded_vector_run-%s_sac-%s.png' %(str(run).zfill(2),str(sac_counter).zfill(3))))
                plt.close()
                sac_counter +=1
        print('%d saccades chosen from original %d detected ones' %(sac_counter-1,sac))

    except:
        print('No object named %s' %alias) # not all of the eyetracking of the runs are saved?
        pass 



