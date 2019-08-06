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

from utils import * #import script to use relevante functions


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
        # load files
        filename = os.path.join(data_dir,'sacc4dm_run-{run}.npz'.format(run=str(run).zfill(2)))
        fileinfo = os.path.join(data_dir,'gaze_timings_run-{run}.npz'.format(run=str(run).zfill(2)))

        # scaling factor of screenshots
        fig_sfactor = 0.1 

        # absolute path to save binary image numpy array
        img_filename = os.path.join(data_dir,'fn_bin_sub-{sj}_run-{run}.npy'.format(sj=str(sj).zfill(2),run=str(run).zfill(2)))

        # get "long" DM(subsampled but not to TR)
        sacc2longDM(filename,fileinfo,img_filename,smp_freq=analysis_params['eyetrack_smp_freq'],subsmp_freq=analysis_params['eyetrack_subsmp_freq'],
                    nrTR=analysis_params['FN_TRs'],TR=analysis_params['TR'],fig_sfactor=fig_sfactor,screen=screen)

    except: 
        print('not found %s, skipping' %filename)
        pass







