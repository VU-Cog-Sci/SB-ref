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


# define participant number and open json parameter file
if len(sys.argv)<2: 
    raise NameError('Please add subject number (ex:1) ' 
                    'as 1st argument in the command line!') 

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


data_dir = os.path.join(analysis_params['eyetrack_dir'],'fn','sub-{sj}'.format(sj=str(sj).zfill(2)))

# do loop for all runs
num_runs = np.arange(1,11)

for run in num_runs:
    try:
        filename = os.path.join(data_dir,'sacc4dm_run-{run}.npz'.format(run=str(run).zfill(2)))
        fileinfo = os.path.join(data_dir,'gaze_timings_run-{run}.npz'.format(run=str(run).zfill(2)))

        sac_data = np.load(filename) # array of (3 x trial length), filled with sacc amplitude, x position and y position of vector       
        trial_info = np.load(fileinfo)#,allow_pickle=True)

        start_scan = int(trial_info['trial_phase_info'][0][0]-trial_info['trl_str_end'][0][0]) #start of scan? relative to begining of trial
        start_movie = int(trial_info['trial_phase_info'][0][1]-trial_info['trl_str_end'][0][0]) #beginning of movie relative to begining of trial
        end_movie = int(trial_info['trial_phase_info'][0][2]-trial_info['trl_str_end'][0][0]) #end of movie relative to begining of trial

        # takes way too long, need to run in cartesius
        for smp_idx in range(start_scan,end_movie+1):#start_scan,end_movie+1): #saves images from start fixation to end of movie duration
            # do loop over all samples to get numpy array with "screenshots"

            x_centered = sac_data['xpos'][smp_idx] + screen[0]/2.0
            y_centered = sac_data['ypos'][smp_idx] + screen[1]/2.0
            amp_pix = sac_data['amplitude'][smp_idx]/2.0 #then diameter will be the amplitude of saccade

            sac_endpoint = plt.Circle((x_centered, y_centered), radius = amp_pix, color='r',clip_on = False) #important to avoid clipping of circle
            fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
            ax.set_xlim((0, screen[0]))
            ax.set_ylim((0, screen[1]))
            ax.add_artist(sac_endpoint)
            plt.axis('off')
            fig.canvas.draw()
            
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            img_gray = color.rgb2gray(np.asarray(img))
            try:
                img_threshbin = cv2.threshold(img_gray,threshold_triangle(img_gray),255,cv2.THRESH_BINARY_INV)[1]
            except:
                img_threshbin = (img_gray*0).astype(np.uint8) # to make it black when no saccade
                pass
            
            if smp_idx==start_scan: #binary image all samples stacked
                img_bin = np.expand_dims(img_threshbin, axis=0)
            else:
                img_bin = np.concatenate((img_bin,np.expand_dims(img_threshbin, axis=0)),axis=0)

            plt.close()
          
        img_filename = os.path.join(data_dir,'fn_bin_sub-{sj}_run-{run}.npy'.format(sj=str(sj).zfill(2),run=str(run).zfill(2)))
        # save as numpy array
        np.save(img_filename, img_bin.astype(np.uint8))

        # resample binary image sequence to 100 Hz (acquired at 1000Hz with eyelink - but check)
        resmp_img = ndimage.zoom(img_bin, [0.1,1,1]) 

        resmp_img_filename = os.path.join(data_dir,'fn_bin_100Hz_sub-{sj}_run-{run}.npy'.format(sj=str(sj).zfill(2),run=str(run).zfill(2)))
        # save as numpy array
        np.save(resmp_img_filename, resmp_img.astype(np.uint8))

        # save binary image in gif file
        # just to see how it looks
        imageio.mimwrite(os.path.join(data_dir,'saccade_retino_sub-{sj}_run-{run}.gif'.format(sj=str(sj).zfill(2),run=str(run).zfill(2))), img_bin , 'GIF')
        imageio.mimwrite(os.path.join(data_dir,'saccade_retino_resampled_sub-{sj}_run-{run}.gif'.format(sj=str(sj).zfill(2),run=str(run).zfill(2))), resmp_img , 'GIF')

    except: 
        print('not found %s, skipping' %filename)
        pass







